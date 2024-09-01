from __future__ import annotations

import importlib.metadata

import pytest
import vtk

import py_spharm_pdm as m
from py_spharm_pdm import core


def make_duck():
    SET = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 1),
        (0, 0, 2),
    ]

    im = vtk.vtkImageData()
    im.SetDimensions(3, 3, 3)
    im.AllocateScalars(vtk.VTK_INT, 1)
    im.SetSpacing(10, 10, 10)

    arr: vtk.vtkDataArray = im.GetPointData().GetScalars()

    arr.Fill(0)

    for ijk in SET:
        arr.SetComponent(im.GetScalarIndex(ijk), 0, 1)

    return im


@pytest.fixture()
def duck() -> vtk.vtkImageData:
    return make_duck()


def test_version():
    assert importlib.metadata.version("py_spharm_pdm") == m.__version__


def test_duck(duck: vtk.vtkImageData):
    print(duck)

    init = core.initial_parameterization(duck)
    print(init)
