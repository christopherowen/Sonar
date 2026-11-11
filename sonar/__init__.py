"""Sonar: steady-state load, engine and model-behaviour measurements for OpenAI-compatible
LLM servers, with paired, boot-aware verdicts.

Standard library only, so it runs from any host Python against any server. Sonar knows
nothing about the deployment it measures; a caller that does stores its identity beside
the measurements (``suite --metadata``):

    python3 -m sonar run URL --mode closed --concurrency 16 --output run.json
    python3 -m sonar compare base.json candidate.json

See README.md and docs/methodology.md.
"""

__version__ = "1.0.0"
SCHEMA = "sonar-run/1"

# Result kinds Sonar writes, and the names earlier releases of this code wrote for the
# same formats (it began as "servebench"). Records keep the name they were written with;
# readers compare kinds, so a recorded reference compares with a new run of its kind.
KINDS = {
    "sonar-run/1": "run",
    "sonar-engine/1": "engine",
    "sonar-model/1": "model",
    "sonar-prefill/1": "prefill",
    "sonar-stall/1": "stall",
    "sonar-suite/1": "suite",
    "sonar-model-evidence/1": "model-evidence",
}
LEGACY_SCHEMAS = {
    "servebench/1": "sonar-run/1",
    "servebench-engine/1": "sonar-engine/1",
    "servebench-model/1": "sonar-model/1",
    "servebench-prefill/1": "sonar-prefill/1",
    "servebench-stall/1": "sonar-stall/1",
    "servebench-suite/1": "sonar-suite/1",
    "servebench-model-evidence/1": "sonar-model-evidence/1",
}


def canonical(schema: str | None) -> str | None:
    """The current name of a result's schema (a legacy name maps to its successor)."""
    return LEGACY_SCHEMAS.get(schema, schema)


def kind(schema: str | None) -> str | None:
    """``run``, ``engine``, ``model``, ``prefill``, ``stall``, ``suite``, or None."""
    return KINDS.get(canonical(schema))
