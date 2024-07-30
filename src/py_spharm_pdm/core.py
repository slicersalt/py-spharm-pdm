from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import vtk
from numpy.typing import ArrayLike
from scipy.sparse import csgraph


def neighbors(data: vtk.vtkPolyData, pt: int) -> set:
    """Returns the set of point ids that share an edge with the given point id."""
    cell_ids = vtk.vtkIdList()
    data.GetPointCells(pt, cell_ids)
    point_ids = set()
    for cell_id_idx in range(cell_ids.GetNumberOfIds()):
        cell: vtk.vtkCell = data.GetCell(cell_ids.GetId(cell_id_idx))
        cell_point_ids: vtk.vtkIdList = cell.GetPointIds()
        for cell_point_id_idx in range(cell_point_ids.GetNumberOfIds()):
            point_ids.add(cell_point_ids.GetId(cell_point_id_idx))
    point_ids.remove(pt)
    return point_ids


def extract_points(data: vtk.vtkPolyData) -> np.ndarray:
    """Returns point coordinates in each row of a matrix."""
    res = np.zeros((data.GetNumberOfPoints(), 3))
    for idx in range(data.GetNumberOfPoints()):
        data.GetPoint(idx, res[idx])
    return res


def extract_normals(data: vtk.vtkPolyData) -> np.ndarray:
    res = np.zeros((data.GetNumberOfPoints(), 3))
    for idx in range(data.GetNumberOfPoints()):
        pts: vtk.vtkPointData = data.GetPointData()
        normals: vtk.vtkFloatArray = pts.GetNormals()
        res[idx] = normals.GetTuple(idx)
    return res


def build_adjacency_matrix(mesh: vtk.vtkPolyData):
    edges = vtk.vtkExtractEdges()
    edges.SetInputData(mesh)
    edges.UseAllPointsOn()

    edges.Update()

    result: vtk.vtkPolyData = edges.GetOutput()
    result.BuildLinks()

    count = edges.GetNumberOfPoints()
    matrix = sp.dok_array((count, count))
    for pt in range(count):
        for nb in neighbors(edges, pt):
            matrix[pt, nb] = 1

    return matrix.tocsr()


def build_mesh(data: vtk.vtkImageData) -> vtk.vtkPolyData:
    ext = np.array(data.GetExtent())
    ext[0::2] -= 1
    ext[1::2] += 1

    pad = vtk.vtkImageConstantPad()
    pad.SetConstant(0)
    pad.SetOutputWholeExtent(ext)
    pad.SetInputData(data)

    net = vtk.vtkSurfaceNets3D()
    net.SetInputConnection(pad.GetOutputPort())
    net.SetSmoothing(False)
    net.SetOutputStyleToBoundary()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(net.GetOutputPort())
    clean.PointMergingOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.FlipNormalsOff()
    normals.ConsistencyOn()

    normals.Update()

    return normals.GetOutput()


def initial_parameterization(data: vtk.vtkImageData) -> vtk.vtkPolyData:
    mesh = build_mesh(data)
    adjacency = build_adjacency_matrix(mesh)

    lat = solve_latitude(adjacency)


def solve_latitude(adjacency: ArrayLike) -> ArrayLike:
    # assumes the first vertex is the north pole and the last vertex is the south pole.

    laplacian = csgraph.laplacian(adjacency)

    lat = np.zeros((len(adjacency),))
    lat[-1] = np.pi

    # `[1:-1]` mask avoids the poles; they are the boundary conditions. `values` is `pi` for nodes adjacent to the south
    # pole and 0 elsewhere. Be careful to keep `values` sparse.
    values = adjacency[:, [-1]] * lat[-1]

    lat[1:-1] = sp.linalg.spsolve(laplacian[1:-1, 1:-1], values[1:-1])

    return lat
