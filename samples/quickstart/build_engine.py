"""Engine factory for the evaluation CLI example in the README.

Run from the repository root:

    PYTHONPATH=samples/quickstart python -m ragsearch.evaluation \
        --engine-factory build_engine.build_engine \
        --cases samples/quickstart/eval_cases.json --summary-only
"""

import os
from pathlib import Path

from ragsearch import setup

SAMPLE = Path(__file__).resolve().parent / "insurance_claims.csv"


def build_engine():
    return setup(SAMPLE, os.environ["COHERE_API_KEY"])
