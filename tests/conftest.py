"""Tests whose function is not implemented yet are skipped instead of failing.

This way ``pytest`` always shows what is done (passed), what is missing
(skipped) and what is wrong (failed).
"""

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")
