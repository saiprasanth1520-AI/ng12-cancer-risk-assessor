"""Pytest configuration for the evaluation harness.

Adds a --run-eval CLI flag. Tests marked with @pytest.mark.eval are skipped
unless --run-eval is passed (they require a live LLM and take minutes to run).
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: mark test as requiring a live LLM (deselected by default)")


def pytest_addoption(parser):
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="Run evaluation tests that require a live LLM connection",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-eval"):
        return  # Don't skip eval tests

    skip_eval = pytest.mark.skip(reason="Need --run-eval to run")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip_eval)
