from __future__ import annotations

import numpy as np
import torch
import vtk
from scipy import sparse as sp
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkCommonDataModel import vtkPolyData


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


def build_adjacency_matrix(edges: vtk.vtkPolyData) -> sp.dok_array:
    count = edges.GetNumberOfPoints()
    matrix = sp.dok_array((count, count))
    for pt in range(count):
        for nb in neighbors(edges, pt):
            matrix[pt, nb] = 1

    return matrix


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

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(net.GetOutputPort())
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
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
    adjacency: sp.csr_array = build_adjacency_matrix(edges).tocsr()

    # writer = vtk.vtkPolyDataWriter()
    # writer.SetInputData(edges)
    # writer.SetFileName("debug/pyedges.vtk")
    # writer.Update()

    NORTH = 0
    SOUTH = edges.GetNumberOfPoints() - 1

    # region Latitude problem
    lat = np.zeros(adjacency.shape[0])
    # set all neighbors of south pole to PI
    lat[adjacency[:, [SOUTH]].todense().flatten().astype(bool)] = np.pi

    A: sp.csr_array = sp.csgraph.laplacian(adjacency).tocsr()[1:-1, 1:-1]
    b = lat[1:-1]

    x = sp.linalg.spsolve(A, b)
    lat[NORTH] = 0
    lat[SOUTH] = np.pi
    lat[1:-1] = x

    arr = numpy_support.numpy_to_vtk(lat)
    arr.SetName("Latitude")
    mesh.GetPointData().AddArray(arr)
    # endregion

    # region Longitude problem
    A: sp.csr_array = sp.csgraph.laplacian(adjacency[1:-1, 1:-1]).tocsr()

    # Arbitrarily increase a diagonal element to make the matrix non-singular.
    A[0, 0] += 2.0

    lon = np.zeros((adjacency.shape[0],))

    verts = extract_points(edges)
    norms = extract_normals(edges)

    # todo this is where the bug is. this does not correctly set the rhs for the longitude problem
    # refer to EqualAreaParametricMeshNewtonIterator::set_longi_rhs
    values = np.zeros((adjacency.shape[0],))

    idx = sp.find(adjacency[:, [NORTH]])[0][0]
    while idx != SOUTH:
        nbs, _, _ = sp.find(adjacency[:, [idx]])
        argnext = np.argmax(lat[nbs])

        for nb in nbs:
            # noinspection PyUnreachableCode
            if np.dot(norms[idx], np.cross([0, 0, 1], verts[nb] - verts[idx])) < 0:
                values[nb] -= 2 * np.pi
                values[idx] += 2 * np.pi

        idx = nbs[argnext]

    b = values[1:-1]
    x = sp.linalg.spsolve(A, b)
    lon[1:-1] = x

    arr = numpy_support.numpy_to_vtk(lon)
    arr.SetName("Longitude")
    mesh.GetPointData().AddArray(arr)
    # endregion

    return mesh


def torch_refine_parameterization(
    mesh: vtkPolyData,
    maxiter=2500,
):
    """Projected Gradient Descent"""

    edge_mesh = build_edges(mesh)

    pd: vtk.vtkPointData = mesh.GetPointData()
    lat = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Latitude"))
    lon = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Longitude"))

    quads = []
    for cid in range(mesh.GetNumberOfCells()):
        pts = vtkIdList()
        mesh.GetCellPoints(cid, pts)
        quads.append([pts.GetId(k) for k in range(pts.GetNumberOfIds())])

    edges = []
    for cid in range(edge_mesh.GetNumberOfCells()):
        pts = vtkIdList()
        edge_mesh.GetCellPoints(cid, pts)
        edges.append([pts.GetId(k) for k in range(pts.GetNumberOfIds())])

    A, B, C, D = np.transpose(quads)

    U, V = np.transpose(edges)

    ideal_cell_area = 4 / 3 * torch.pi / mesh.GetNumberOfCells()

    lat = torch.tensor(lat, requires_grad=True)
    lon = torch.tensor(lon, requires_grad=True)

    optimizer = torch.optim.Adam([lat, lon], lr=1e-2)
    for i in range(maxiter):
        optimizer.zero_grad()

        coords = torch.stack(
            [
                torch.sin(lat) * torch.cos(lon),
                torch.sin(lat) * torch.sin(lon),
                torch.cos(lat),
            ]
        ).T

        _ = i

        a = coords[A]
        b = coords[B]
        c = coords[C]
        d = coords[D]

        ab = torch.linalg.vecdot(a, b, dim=1)
        ac = torch.linalg.vecdot(a, c, dim=1)
        ad = torch.linalg.vecdot(a, d, dim=1)
        bc = torch.linalg.vecdot(b, c, dim=1)
        bd = torch.linalg.vecdot(b, d, dim=1)
        cd = torch.linalg.vecdot(c, d, dim=1)

        Ca = bd - ad * ab
        Cb = ac - ab * bc
        Cc = bd - bc * cd
        Cd = ac - cd * ad

        spata = torch.stack([d, a, b], dim=2).det()
        spatb = torch.stack([a, b, c], dim=2).det()
        spatc = torch.stack([b, c, d], dim=2).det()
        spatd = torch.stack([c, d, a], dim=2).det()

        area = -(
            torch.atan2(Ca, spata)
            + torch.atan2(Cb, spatb)
            + torch.atan2(Cc, spatc)
            + torch.atan2(Cd, spatd)
        )
        areas = torch.fmod(area + 8.5 * torch.pi, torch.pi) - 0.5 * torch.pi

        AREA_POW = 2
        AREA_FACTOR = 50
        area_error = (
            ((areas.abs() - ideal_cell_area) * AREA_FACTOR)
            .pow(AREA_POW)
            .sum()
            .pow(1 / AREA_POW)
        )

        crossings = -torch.stack(
            [
                torch.linalg.vecdot(a - b, c - d, dim=1),  # should be -1
                torch.linalg.vecdot(b - c, d - a, dim=1),  # should be -1
            ]
        )
        crossings[crossings < 0] *= 10
        crossings_error = (1 - crossings).pow(2).sum().pow(1 / 2)

        u = coords[U]
        v = coords[V]

        goal = (1 - (u * v).sum(dim=1)).sum() / 2

        metric = area_error + crossings_error + goal

        metric.backward()
        optimizer.step()

        arr = numpy_support.numpy_to_vtk(lat.detach())
        arr.SetName("Latitude")
        pd.AddArray(arr)

        arr = numpy_support.numpy_to_vtk(lon.detach())
        arr.SetName("Longitude")
        pd.AddArray(arr)

        if i % 50 == 0:
            yield i, mesh

        # with torch.no_grad():
        #     # enforce sphere constraint
        #     sphere /= sphere.norm(dim=0)
        #
        #     # project gradients into tangent space
        #     sphere.grad -= sphere * (sphere * sphere.grad).sum(dim=0)

    arr = numpy_support.numpy_to_vtk(lat.detach())
    arr.SetName("Latitude")
    pd.AddArray(arr)

    arr = numpy_support.numpy_to_vtk(lon.detach())
    arr.SetName("Longitude")
    pd.AddArray(arr)


def fit_spharms(mesh: vtk.vtkPolyData):
    pd: vtk.vtkPointData = mesh.GetPointData()

    lat = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Latitude"))
    lon = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Longitude"))

    _ = lat, lon
