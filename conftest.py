import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run live integration tests that make real API calls",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip = pytest.mark.skip(reason="pass --live to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)
