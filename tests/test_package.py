from __future__ import annotations

import importlib.metadata

import py_spharm_pdm as m


def test_version():
    assert importlib.metadata.version("py_spharm_pdm") == m.__version__
