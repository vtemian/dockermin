"""Shared pytest fixtures."""


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: tests that require a running Docker daemon")
