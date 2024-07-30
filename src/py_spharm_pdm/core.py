from __future__ import annotations

import numpy as np
import vtk
from scipy import sparse as sp
from vtkmodules.util import numpy_support


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


def build_adjacency_matrix(edges: vtk.vtkPolyData):
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


def build_edges(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    extract = vtk.vtkExtractEdges()
    extract.SetInputData(mesh)
    extract.UseAllPointsOn()

    extract.Update()

    edges: vtk.vtkPolyData = extract.GetOutput()
    edges.BuildLinks()

    return edges


def initial_parameterization(data: vtk.vtkImageData) -> vtk.vtkPolyData:
    mesh = build_mesh(data)
    edges = build_edges(mesh)
    adjacency = build_adjacency_matrix(edges)

    # region Latitude problem
    laplacian = sp.csgraph.laplacian(adjacency).tocsr()

    lat = np.zeros((adjacency.shape[0],))
    lat[-1] = np.pi

    # `[1:-1]` mask avoids the poles; they are the boundary conditions. `values` is `pi` for nodes adjacent to the south
    # pole and 0 elsewhere. Be careful to keep `values` sparse.
    values = adjacency[:, [-1]] * lat[-1]

    lat[1:-1] = sp.linalg.spsolve(laplacian[1:-1, 1:-1], values[1:-1])
    # endregion

    # region Longitude problem
    lon = np.zeros((adjacency.shape[0],))

    geo = vtk.vtkDijkstraGraphGeodesicPath()
    geo.SetInputData(edges)
    geo.SetStartVertex(0)
    geo.SetEndVertex(edges.GetNumberOfPoints() - 1)
    geo.Update()
    short_path = np.array(
        [geo.GetIdList().GetId(idx) for idx in range(geo.GetIdList().GetNumberOfIds())]
    )

    verts = extract_points(edges)
    norms = extract_normals(edges)

    values = np.zeros((adjacency.shape[0],))
    for _, idx, nxt in np.lib.stride_tricks.sliding_window_view(short_path, 3):
        row_idxs, _, _ = sp.find(adjacency[:, [idx]])
        for row_idx in row_idxs:
            if row_idx in short_path:
                # don't alter the path itself
                continue

            # if west
            if (
                np.dot(
                    norms[idx],
                    np.cross(verts[nxt] - verts[idx], verts[row_idx] - verts[idx]),
                )
                > 0
            ):
                values[row_idx] += 2 * np.pi
                values[idx] -= 2 * np.pi

    lon_laplacian_matrix = laplacian.copy()
    row_idxs, _, _ = sp.find(adjacency[:, [0, -1]])
    for row_idx in row_idxs:
        lon_laplacian_matrix[row_idx, row_idx] -= 1
    lon_laplacian_matrix[0, 0] += 2

    lon[1:-1] = sp.linalg.spsolve(lon_laplacian_matrix[1:-1, 1:-1], values[1:-1])
    # endregion

    pd: vtk.vtkPointData = mesh.GetPointData()

    arr = numpy_support.numpy_to_vtk(lat)
    arr.SetName("Latitude")
    pd.AddArray(arr)

    arr = numpy_support.numpy_to_vtk(lon)
    arr.SetName("Longitude")
    pd.AddArray(arr)

    return mesh
