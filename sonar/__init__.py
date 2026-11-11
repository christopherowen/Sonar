"""Sonar: steady-state load, engine and model-behaviour measurements for OpenAI-compatible
LLM servers, with paired, boot-aware verdicts.

Standard library only, so it runs from any host Python against any server. Sonar knows
nothing about the deployment it measures; a caller that does stores its identity beside
the measurements (``suite --metadata``):

    python3 -m sonar run URL --mode closed --concurrency 16 --output run.json
    python3 -m sonar compare base.json candidate.json

See README.md and docs/methodology.md.
"""

import json
import os

__version__ = "1.1.0"
SCHEMA = "sonar-run/1"

# Result kinds Sonar writes. Records keep the schema name they were written with; readers
# compare kinds, so a recorded reference compares with a new run of its kind.
KINDS = {
    "sonar-run/1": "run",
    "sonar-engine/1": "engine",
    "sonar-model/1": "model",
    "sonar-prefill/1": "prefill",
    "sonar-stall/1": "stall",
    "sonar-suite/1": "suite",
    "sonar-model-evidence/1": "model-evidence",
}
# Other names a caller's records give these formats (an earlier client's, say), each mapped
# to a Sonar schema: add_schema_aliases in-process, or SONAR_SCHEMA_ALIASES (a JSON object)
# for the commands a caller runs as child processes.
SCHEMA_ALIASES_ENV = "SONAR_SCHEMA_ALIASES"
_ALIASES: dict[str, str] = {}


def _checked(aliases: dict) -> dict[str, str]:
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and v in KINDS for k, v in aliases.items()):
        raise ValueError(f"schema aliases must map names to Sonar schemas ({', '.join(KINDS)})")
    return dict(aliases)


def add_schema_aliases(aliases: dict) -> None:
    """Read results recorded under ``aliases``' names as the Sonar schemas they map to."""
    _ALIASES.update(_checked(aliases))


def schema_aliases() -> dict[str, str]:
    """Every alias in force: those added in-process, then SONAR_SCHEMA_ALIASES."""
    text = os.environ.get(SCHEMA_ALIASES_ENV)
    return {**_ALIASES, **(_checked(json.loads(text)) if text else {})}


def canonical(schema: str | None) -> str | None:
    """The Sonar name of a result's schema (an alias maps to the schema it names)."""
    return schema_aliases().get(schema, schema)


def kind(schema: str | None) -> str | None:
    """``run``, ``engine``, ``model``, ``prefill``, ``stall``, ``suite``, or None."""
    return KINDS.get(canonical(schema))
