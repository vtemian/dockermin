"""Shared pytest fixtures."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: tests that require a running Docker daemon")
