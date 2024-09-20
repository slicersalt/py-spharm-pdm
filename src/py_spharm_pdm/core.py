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
    normals.ComputeCellNormalsOff()
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
    maxiter=1000,
):
    """Projected Gradient Descent"""

    pd: vtk.vtkPointData = mesh.GetPointData()
    lat = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Latitude"))
    lon = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Longitude"))

    # todo instead of EDGES, maybe get tuples from sparse adjacency matrix?
    #  maybe some multiplication by the adjacency matrix?

    # cd: vtk.vtkCellData = mesh.GetCellData()

    # AREA_WEIGHT = 1
    # ANGLE_WEIGHT = 0

    quads = []
    for cid in range(mesh.GetNumberOfCells()):
        pts = vtkIdList()
        mesh.GetCellPoints(cid, pts)
        quads.append([pts.GetId(k) for k in range(pts.GetNumberOfIds())])

    A, B, C, D = np.transpose(quads)

    # print(connections)

    ideal_cell_area = 4 * torch.pi / mesh.GetNumberOfCells()

    # cells = np.zeros((mesh.GetNumberOfCells(), 3), dtype="i")
    # for idx in range(mesh.GetNumberOfCells()):
    #     cell: vtk.vtkCell = mesh.GetCell(idx)
    #     ids: vtk.vtkIdList = cell.GetPointIds()
    #     cells[idx] = [ids.GetId(k) for k in range(ids.GetNumberOfIds())]

    # state = torch.tensor(np.array([
    #     lat,
    #     lon,
    # ]), requires_grad=True)

    lat = torch.tensor(lat, requires_grad=True)
    lon = torch.tensor(lon, requires_grad=True)

    # sphere = torch.tensor(np.array([
    #     np.sin(lat) * np.cos(lon),
    #     np.sin(lat) * np.sin(lon),
    #     np.cos(lat),
    # ]), requires_grad=True)

    optimizer = torch.optim.Adam([lat, lon], lr=9e-4)
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

        A1 = torch.stack([coords[C], coords[B], coords[A]], dim=2).det()
        A2 = torch.stack([coords[A], coords[D], coords[C]], dim=2).det()
        areas = (A1 + A2) / 2
        area_rms_error = (areas - ideal_cell_area).pow(2).sum().sqrt()

        # diagonal = torch.tensordot(coords[A], coords[C], dims=([0], [0]))
        # diagonal = coords[A].T @ coords[C]
        AC = (coords[A] * coords[C]).sum(dim=1)
        BD = (coords[B] * coords[D]).sum(dim=1)

        diagonals = torch.stack([AC, BD])
        # print(diagonals)
        diagonals_rms_error = (1 - diagonals).pow(2).sum().sqrt()

        # metric = area_rms_error + diagonals_rms_error
        metric = diagonals_rms_error + area_rms_error

        # metric = torch.var(areas)
        # print(metric)

        # print(torch.stack([coords[A], coords[B], coords[C]], dim=1).det().shape)
        # print(torch.stack([coords[A], coords[B], coords[C]]).shape)
        # coords[C], coords[D], coords[A],
        # sys.exit()

        # torch.linalg.det()
        # delta = torch.norm(coords[U] - coords[V], dim=1)

        # metric = torch.var(delta)
        # print(metric)

        metric.backward()
        optimizer.step()

        # corners = sphere.T[cells, :]
        # areas = torch.linalg.det(corners)
        # area_error = torch.abs(areas - ideal_cell_area)
        #
        # # total_area_error = torch.sum(area_error)  # todo is there a better metric for this?
        # total_area_error = area_error.norm()
        #
        # l0 = corners[:, 2] - corners[:, 1]
        # l0 = l0 / l0.norm(dim=1)[:, None]
        # l1 = corners[:, 2] - corners[:, 0]
        # l1 = l1 / l1.norm(dim=1)[:, None]
        # l2 = corners[:, 1] - corners[:, 0]
        # l2 = l2 / l2.norm(dim=1)[:, None]
        #
        # dots = torch.stack(
        #     [
        #         l1 * l2,
        #         -l2 * l0,
        #         l0 * l1,
        #     ]
        # )
        # angle = torch.acos(dots)
        #
        # angle_error = torch.abs(angle - torch.pi / 3)
        # total_angle_error = angle_error.norm()
        #
        # metric = total_area_error * AREA_WEIGHT + total_angle_error * ANGLE_WEIGHT
        #
        # metric.backward()
        #
        # # print(i, '|||', *(f'{v:0.1f}' for v in area_error), '|||', total_area_error.item())
        # print(i, "|||", metric.item())
        #
        # optimizer.step()
        #
        # with torch.no_grad():
        #     # enforce sphere constraint
        #     sphere /= sphere.norm(dim=0)
        #
        #     # project gradients into tangent space
        #     sphere.grad -= sphere * (sphere * sphere.grad).sum(dim=0)

    # sphere_f = sphere.detach().numpy()
    # print(sphere_f[2])

    # lat_f = np.arccos(sphere_f[2])
    # lon_f = np.atan2(sphere_f[1], sphere_f[0])

    # arr = numpy_support.numpy_to_vtk(lat)
    # arr.SetName("Latitude_0")
    # pd.AddArray(arr)

    # arr = numpy_support.numpy_to_vtk(lon)
    # arr.SetName("Longitude_0")
    # pd.AddArray(arr)

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
