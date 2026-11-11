"""The verdict table over suite runs: the boot term, averaging both boots of each side, and
reading results recorded under earlier schema names."""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sonar
from sonar import __main__ as cli, canonical, kind, suite, verdict
from tests import test_sonar as fakes


def paired(change: float, half: float = 0.05) -> dict:
    return {"metric": "decode tok/s, paired", "change_pct": change, "ci95_pct": half,
            "low_pct": change - half, "high_pct": change + half, "higher_is_better": True}


def outcome(grid: float, tps: float, pair: float | None = None, grid_ci: float = 0.15) -> dict:
    rows = [{"metric": "output tok/s", "change_pct": tps, "ci95_pct": 0.9}]
    if pair is not None:
        rows.append(paired(pair))
    return {"steps": {"grid": ("sonar-engine/1", {"rows": [
                {"metric": "decode geometric mean (20)", "change_pct": grid, "ci95_pct": grid_ci}]}),
            "c16": ("sonar-run/1", {"rows": rows})}}


class BootTermTest(unittest.TestCase):
    def test_one_bound_for_both_sides(self):
        self.assertAlmostEqual(suite.boot_term(0.4, None, 1, 1), 0.4)
        self.assertAlmostEqual(suite.boot_term(0.4, None, 2, 1), 0.4 * math.sqrt(0.75))
        self.assertAlmostEqual(suite.boot_term(0.4, 0.4, 2, 2), 0.4 * math.sqrt(0.5))
        self.assertAlmostEqual(suite.boot_term(0.4, None, 2, 2), 0.4 * math.sqrt(0.5))

    def test_each_side_its_own_bound(self):
        self.assertAlmostEqual(suite.boot_term(0.4, 0.2, 2, 2), math.sqrt(0.4 ** 2 / 4 + 0.2 ** 2 / 4))
        self.assertAlmostEqual(suite.boot_term(0.4, 0.2, 1, 2), math.sqrt(0.4 ** 2 / 2 + 0.2 ** 2 / 4))


class AveragingTest(unittest.TestCase):
    floors = {"c16": 0.43, "grid": 0.43}

    def rows(self, **extra):
        return {(r["step"], r["metric"]): r for r in suite.with_floor(
            outcome(-1.0, 1.0, 0.40), outcome(0.10, -0.10, 0.10), floors=self.floors,
            reference_to_repeat=outcome(-0.9, 1.2, 0.30), **extra)}

    def test_without_a_reference_repeat_nothing_changes(self):
        rows = self.rows()
        pair = rows[("c16", "decode tok/s, paired")]
        self.assertEqual((pair["boots"], pair["reference_boots"]), (2, 1))
        self.assertAlmostEqual(pair["change_pct"], 0.35)
        self.assertAlmostEqual(pair["floor_pct"], 0.43 * math.sqrt(0.75))
        self.assertIsNone(pair["reference_repeat_pct"])
        # A window row keeps the first pairing and interval + |repeat|.
        window = rows[("c16", "output tok/s")]
        self.assertEqual((window["boots"], window["reference_boots"], window["change_pct"]), (1, 1, 1.0))
        self.assertAlmostEqual(window["margin_pct"], 0.9 + 0.10)
        # The grid's geometric mean takes its recorded floor over both candidate boots.
        grid = rows[("grid", "decode geometric mean (20)")]
        self.assertEqual(grid["boots"], 2)
        self.assertAlmostEqual(grid["change_pct"], -0.95)
        self.assertAlmostEqual(grid["floor_pct"], 0.43 * math.sqrt(0.75))
        self.assertAlmostEqual(grid["margin_pct"], math.hypot(0.15, 0.43 * math.sqrt(0.75)))
        self.assertEqual(grid["verdict"], "better")  # step times fell

    def test_a_grid_without_a_floor_keeps_interval_plus_repeat(self):
        rows = {(r["step"], r["metric"]): r for r in suite.with_floor(outcome(-0.3, 1.0), outcome(0.2, 0.1))}
        grid = rows[("grid", "decode geometric mean (20)")]
        self.assertIsNone(grid["floor_pct"])
        self.assertAlmostEqual(grid["margin_pct"], 0.15 + 0.2)
        self.assertEqual(grid["verdict"], "same")

    def test_both_boots_of_each(self):
        rows = self.rows(reference_repeat=outcome(0.05, 0.2, -0.1),
                         reference_again_to_candidate=outcome(-1.1, 0.8, 0.50),
                         reference_again_to_repeat=outcome(-1.0, 1.0, 0.40))
        pair = rows[("c16", "decode tok/s, paired")]
        self.assertEqual((pair["boots"], pair["reference_boots"]), (2, 2))
        self.assertAlmostEqual(pair["change_pct"], (0.40 + 0.30 + 0.50 + 0.40) / 4)
        self.assertAlmostEqual(pair["reference_repeat_pct"], -0.1)
        self.assertAlmostEqual(pair["floor_pct"], 0.43 * math.sqrt(0.5))
        self.assertAlmostEqual(pair["margin_pct"], math.hypot(0.05, 0.43 * math.sqrt(0.5)))
        window = rows[("c16", "output tok/s")]
        self.assertAlmostEqual(window["change_pct"], (1.0 + 0.8 + 1.2 + 1.0) / 4)
        # Window rows: interval + the repeats' boot term over two boots of each.
        self.assertAlmostEqual(window["boot_pct"], math.sqrt(0.10 ** 2 / 4 + 0.2 ** 2 / 4))
        self.assertAlmostEqual(window["margin_pct"], 0.9 + window["boot_pct"])
        grid = rows[("grid", "decode geometric mean (20)")]
        self.assertAlmostEqual(grid["change_pct"], (-1.0 - 0.9 - 1.1 - 1.0) / 4)
        self.assertAlmostEqual(grid["floor_pct"], 0.43 * math.sqrt(0.5))

    def test_a_noisy_repeat_on_either_side_widens_its_bound(self):
        rows = self.rows(reference_repeat=outcome(0.9, 0.2, -0.8),
                         reference_again_to_candidate=outcome(-1.1, 0.8, 0.50),
                         reference_again_to_repeat=outcome(-1.0, 1.0, 0.40))
        self.assertAlmostEqual(rows[("c16", "decode tok/s, paired")]["floor_pct"], math.sqrt(0.43 ** 2 / 4 + 0.8 ** 2 / 4))
        self.assertAlmostEqual(rows[("grid", "decode geometric mean (20)")]["floor_pct"],
                               math.sqrt(0.43 ** 2 / 4 + 0.9 ** 2 / 4))

    def test_a_missing_pairing_counts_only_the_boots_used(self):
        rows = self.rows(reference_repeat=outcome(0.05, 0.2, -0.1), reference_again_to_candidate=outcome(-1.1, 0.8, 0.5))
        pair = rows[("c16", "decode tok/s, paired")]
        self.assertEqual((pair["boots"], pair["reference_boots"]), (2, 2))
        self.assertAlmostEqual(pair["change_pct"], (0.40 + 0.50 + 0.30) / 3)

    def test_the_table_says_which_boots_were_averaged(self):
        rows = suite.with_floor(outcome(-1.0, 1.0, 0.4), outcome(0.1, -0.1, 0.1), floors=self.floors,
                                reference_to_repeat=outcome(-0.9, 1.2, 0.3), reference_repeat=outcome(0.05, 0.2, -0.1),
                                reference_again_to_candidate=outcome(-1.1, 0.8, 0.5),
                                reference_again_to_repeat=outcome(-1.0, 1.0, 0.4))
        text = suite.format_floor(rows, "ref", "cand", "cand-again", reference_repeat="ref-again",
                                  floors_note="floors (testbed A, screen): grid 0.37% (6 pairs)")
        head = text.splitlines()[0]
        self.assertTrue(head.startswith("verdicts: ref -> cand, against the run's own repeat (cand -> cand-again)"))
        self.assertIn("and the reference's (ref-again); each change is the mean over both boots of each "
                      "(cand, cand-again; ref, ref-again)", head)
        self.assertIn("(both boots of each, reference repeat -0.10%; window tok/s", text)
        self.assertIn("floors (testbed A, screen)", text)
        self.assertTrue(text.splitlines()[-1].startswith("paired rows: margin"))
        self.assertIn("in quadrature", text)
        # Without a reference repeat the header is the old one.
        plain = suite.format_floor(suite.with_floor(outcome(-1.0, 1.0, 0.4), outcome(0.1, -0.1, 0.1), floors=self.floors),
                                   "ref", "cand", "cand-again")
        self.assertEqual(plain.splitlines()[0], "verdicts: ref -> cand, against the run's own repeat (cand -> cand-again)")



class SchemaNameTest(unittest.TestCase):
    ALIASES = {"oldbench/1": "sonar-run/1", "oldbench-engine/1": "sonar-engine/1"}

    def setUp(self):
        saved = dict(sonar._ALIASES)
        self.addCleanup(lambda: (sonar._ALIASES.clear(), sonar._ALIASES.update(saved)))
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(sonar.SCHEMA_ALIASES_ENV, None)

    def test_sonar_names_no_other_tool(self):
        self.assertEqual(sonar.schema_aliases(), {})
        self.assertEqual(kind("sonar-model/1"), "model")
        self.assertIsNone(kind("oldbench/1"))

    def test_a_callers_aliases_map_to_sonar_kinds(self):
        sonar.add_schema_aliases(self.ALIASES)
        self.assertEqual(canonical("oldbench/1"), "sonar-run/1")
        self.assertEqual(kind("oldbench-engine/1"), "engine")
        self.assertIsNone(kind("something-else/1"))
        with self.assertRaises(ValueError):
            sonar.add_schema_aliases({"x/1": "not-a-sonar-schema/1"})

    def test_aliases_reach_child_processes_through_the_environment(self):
        os.environ[sonar.SCHEMA_ALIASES_ENV] = json.dumps(self.ALIASES)
        self.assertEqual(kind("oldbench/1"), "run")
        os.environ[sonar.SCHEMA_ALIASES_ENV] = '{"x/1": "y"}'
        with self.assertRaises(ValueError):
            kind("x/1")

    def test_a_recorded_result_compares_with_a_new_one(self):
        def run(batches):
            from sonar import stats
            return {"schema": "sonar-run/1", "workload": {"mode": "closed", "concurrency": 1, "prompts_sha256": "x"},
                    "summary": {"output_tps": stats.describe(batches) | {"batches": batches},
                                "latency": {"ttft_s": {"p50": 0.5}, "tpot_ms": {"p50": 60.0},
                                            "per_user_tps": {"p50": 16.0}}}}
        sonar.add_schema_aliases(self.ALIASES)
        recorded = dict(run([100, 101, 99, 100]), schema="oldbench/1")
        schema, result = cli.compare_results(recorded, run([110, 111, 109, 110]), 3.0)
        self.assertEqual(schema, "sonar-run/1")
        self.assertEqual({row["metric"]: row["verdict"] for row in result["rows"]}["output tok/s"], "better")
        with self.assertRaises(SystemExit):
            cli.compare_results(recorded, dict(run([1, 2]), schema="sonar-engine/1"), 3.0)


class FloorsTest(unittest.TestCase):
    def test_inline_file_and_provenance_forms(self):
        self.assertEqual(verdict.load_floors('{"c1": 0.35}'), {"c1": 0.35})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "floors.json")
            path.write_text(json.dumps({"floors_pct": {"c1": 0.35, "c16": 0.43}, "source": "16 A/A pairs"}))
            self.assertEqual(verdict.load_floors(str(path)), {"c1": 0.35, "c16": 0.43})
        self.assertEqual(verdict.load_floors(None), {})
        with self.assertRaises(SystemExit):
            verdict.load_floors('{"c1": "wide"}')

    def test_pairs(self):
        self.assertEqual(verdict.pairs(["cand", "ref", "again"]), [(1, 0), (1, 2), (0, 2)])
        self.assertEqual(verdict.pairs(["cand", "ref"]), [(1, 0)])
        self.assertEqual(verdict.pairs(["cand"]), [])


class TableTest(unittest.TestCase):
    def test_three_arm_table(self):
        definition = dict(fakes.SuiteTest.TINY)
        servers = [fakes.FakeServer(), fakes.FakeServer(variant=True), fakes.FakeServer()]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "tiny.json")
                Path(path).write_text(json.dumps(definition))
                runs = []
                for label, server in zip(("cand", "ref", "cand-again"), servers):
                    out = os.path.join(tmp, label)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(cli.main(["suite", server.url, "--suite", path, "--out", out,
                                                   "--label", label]), 0)
                    runs.append(out)
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    self.assertEqual(cli.main(["table", *runs, "--names", "cand,ref (recorded),cand-again",
                                               "--floors", '{"c2": 0.4}']), 0)
                text = printed.getvalue()
                self.assertIn("verdicts: ref (recorded) -> cand, against the run's own repeat (cand -> cand-again)", text)
                self.assertIn("rows a step", text)
                self.assertIn("######## ref (recorded) -> cand", text)
                self.assertIn("######## cand -> cand-again", text)
                self.assertIn(f"suite tiny ({suite.digest(definition)[:12]})", text)
                with self.assertRaises(SystemExit):
                    cli.main(["table", *runs, "--names", "a,b"])
                # The reference booted again: every headline row averages both boots of each.
                again = os.path.join(tmp, "ref-again")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    cli.main(["suite", servers[1].url, "--suite", path, "--out", again, "--label", "ref-again"])
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    cli.main(["table", *runs, "--reference-repeat", again, "--floors", '{"c2": 0.4, "grid": 0.4}',
                              "--floors-note", "floors: from 6 same-build pairs"])
                text = printed.getvalue()
                self.assertIn("and the reference's (ref-again); each change is the mean over both boots of each", text)
                self.assertIn("floors: from 6 same-build pairs", text)
                self.assertIn("######## ref -> ref-again", text)
                # Named among the runs (a middle arm), it is not shown or paired twice.
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    cli.main(["table", runs[0], runs[1], again, runs[2], "--reference-repeat", again])
                self.assertEqual(printed.getvalue().count("######## ref -> ref-again"), 1)
        finally:
            for server in servers:
                server.close()


if __name__ == "__main__":
    unittest.main()
