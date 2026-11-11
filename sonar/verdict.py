"""The verdict table for suite runs: a candidate against a reference, judged against the
candidate's own repeat on another boot (and the reference's, when it has one).

Arms are given candidate first (it also heat-soaks the hardware), then the reference, then
the candidate again: ``sonar table CANDIDATE REFERENCE CANDIDATE-AGAIN``. With two or more
runs the second is the reference; it is compared with every other run, and the first with
the last as the run's own repeat. A reference may be a recorded run: a released build's
numbers do not change, so it need not be measured again.

With three or more runs the verdict block comes first: reference to candidate, each headline
metric judged against the repeat (``suite.with_floor``). The paired decode rows, and an
engine grid step's geometric mean, are judged against recorded A/A floors by step
(``--floors``): the 95% bound of one boot against another for that step, measured on the
caller's hardware. With the reference's own repeat boot (``--reference-repeat``) every
headline row is the mean over both boots of each, and the boot term shrinks to x0.707 of one
pair's (``suite.boot_term``). A rows-a-step block follows, then every step of every pair.
"""

import argparse
import json
import os
from pathlib import Path

from . import suite


def load_floors(value: str | None) -> dict:
    """A JSON object of step id to A/A floor in %, inline or in a file. A file may also be
    {"floors_pct": {...}, ...} so it can carry its provenance beside the numbers."""
    if not value:
        return {}
    text = Path(value).read_text(encoding="utf-8") if os.path.isfile(value) else value
    data = json.loads(text)
    floors = data.get("floors_pct", data) if isinstance(data, dict) else None
    if not isinstance(floors, dict) or not all(isinstance(v, (int, float)) for v in floors.values()):
        raise SystemExit(f"--floors {value}: expected an object of step id to percent")
    return {str(k): float(v) for k, v in floors.items()}


def pairs(names: list[str], reference_repeat: bool = False) -> list[tuple[int, int]]:
    """(base, candidate) index pairs: the reference (the second) against every other run,
    then the first against the last as the repeat; with a reference repeat (the index after
    the last), the reference against it."""
    if len(names) < 2:
        return []
    out = [(1, i) for i in range(len(names)) if i != 1]
    if len(names) > 2:
        out.append((0, len(names) - 1))
    if reference_repeat:
        out.append((1, len(names)))
    return out


def verdicts(directories: list[str], names: list[str], floors: dict, compare_results, read_json, *,
             reference_repeat: tuple[str, str] | None = None, floors_note: str | None = None,
             tolerance: float = 3.0) -> str:
    """The verdict block. ``reference_repeat`` is the reference's repeat boot (its directory
    and name)."""
    judge = lambda b, c: compare_results(b, c, tolerance)
    candidate, reference, again = directories[0], directories[1], directories[-1]
    reference_to_candidate = suite.compare(read_json, reference, candidate, judge)
    repeat = suite.compare(read_json, candidate, again, judge)
    reference_to_repeat = suite.compare(read_json, reference, again, judge)
    extra = {}
    if reference_repeat:
        second = reference_repeat[0]
        extra = {"reference_repeat": suite.compare(read_json, reference, second, judge),
                 "reference_again_to_candidate": suite.compare(read_json, second, candidate, judge),
                 "reference_again_to_repeat": suite.compare(read_json, second, again, judge)}
    rows = suite.with_floor(reference_to_candidate, repeat, floors=floors, reference_to_repeat=reference_to_repeat,
                            **extra)
    return suite.format_floor(rows, names[1], names[0], names[-1],
                              reference_repeat=reference_repeat[1] if reference_repeat else None,
                              floors_note=floors_note)


def table(directories: list[str], names: list[str], floors: dict, compare_results, read_json, compare_main,
          emit=print, tolerance: float = 3.0, reference_repeat: tuple[str, str] | None = None,
          floors_note: str | None = None, warning: str | None = None) -> None:
    """Print the verdict block, the rows-a-step block and every pair's comparison. ``warning``
    is printed under the verdicts' first line (why a reference repeat was not used, say)."""
    if len(directories) > 2:
        text = verdicts(directories, names, floors, compare_results, read_json, reference_repeat=reference_repeat,
                        floors_note=floors_note, tolerance=tolerance)
        if warning:
            text = text.replace("\n", f"\n{warning}\n", 1)
        emit(text)
    else:
        reference_repeat = None  # a repeat is averaged only beside a candidate repeat
    # The reference's repeat may be one of the runs already (a middle arm); then it is shown
    # and paired as one of them.
    extra = reference_repeat is not None and reference_repeat[0] not in directories
    shown = list(zip(names, directories)) + ([(reference_repeat[1], reference_repeat[0])] if extra else [])
    if directories:
        emit("\n" + suite.format_accounting({name: suite.step_accounting(read_json, directory)
                                             for name, directory in shown}))
    every = [directory for _, directory in shown]
    labels = [name for name, _ in shown]
    for base, candidate in pairs(names, extra):
        emit(f"\n######## {labels[base]} -> {labels[candidate]}", flush=True)
        compare_main(["compare", every[base], every[candidate], "--tolerance", str(tolerance)])


def add_parser(sub, command) -> None:
    parser = sub.add_parser("table", help="verdicts: candidate, reference, candidate again (suite runs)")
    parser.add_argument("runs", nargs="+", metavar="RUN",
                        help="suite run directories: candidate, reference, then the candidate's repeat")
    parser.add_argument("--names", help="comma-separated names for the runs (default: their directory names); "
                                        "with --reference-repeat, one more for it")
    parser.add_argument("--reference-repeat", metavar="RUN",
                        help="the reference's repeat boot (a suite run of the same suite): every headline row "
                             "becomes the mean over both boots of each")
    parser.add_argument("--floors", metavar="JSON",
                        help="A/A floors in %% by step id (a JSON object, or a file holding one), for the paired "
                             "decode rows and engine grid steps; without one a paired row is not judged")
    parser.add_argument("--floors-note", help="a line printed under the verdicts naming the floors' source")
    parser.add_argument("--warning", help="a line printed under the verdicts' first line")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="budget in %% for metrics without an interval")
    parser.set_defaults(func=command)


def names_for(args: argparse.Namespace) -> tuple[list[str], tuple[str, str] | None]:
    """The runs' names, and the reference repeat as (directory, name) or None."""
    count = len(args.runs) + (1 if args.reference_repeat else 0)
    if args.names:
        names = args.names.split(",")
        if len(names) != count:
            raise SystemExit(f"--names has {len(names)} names for {count} runs")
    else:
        names = [Path(run).name for run in args.runs] + ([Path(args.reference_repeat).name]
                                                          if args.reference_repeat else [])
    repeat = (args.reference_repeat, names[-1]) if args.reference_repeat else None
    if args.reference_repeat and len(args.runs) < 3:
        raise SystemExit("--reference-repeat needs the candidate's repeat as well (three runs)")
    return names[:len(args.runs)], repeat
