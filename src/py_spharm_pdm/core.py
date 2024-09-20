from __future__ import annotations

import numpy as np
import scipy.optimize
import vtk
from scipy import sparse as sp
from scipy.optimize import NonlinearConstraint
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


def refine_parameterization(mesh: vtk.vtkPolyData):
    # edges = build_edges(mesh)
    # matrix = build_adjacency_matrix(edges)

    pd: vtk.vtkPointData = mesh.GetPointData()
    lat = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Latitude"))
    lon = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Longitude"))

    # todo instead of EDGES, maybe get tuples from sparse adjacency matrix?
    #  maybe some multiplication by the adjacency matrix?

    # cd: vtk.vtkCellData = mesh.GetCellData()

    cells = np.zeros((mesh.GetNumberOfCells(), 4), dtype="i")
    for idx in range(mesh.GetNumberOfCells()):
        cell: vtk.vtkCell = mesh.GetCell(idx)
        ids: vtk.vtkIdList = cell.GetPointIds()
        cells[idx] = [ids.GetId(k) for k in range(ids.GetNumberOfIds())]

    sphere = np.array(
        [
            np.sin(lat) * np.cos(lon),
            np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ]
    ).T

    ideal_cell_area = 4 * np.pi / mesh.GetNumberOfCells()

    edge_indices = [[0, 1], [1, 2], [2, 3], [3, 0]]

    angle_det_indices = [
        [3, 0, 1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
    ]

    diag_a_indices = [1, 0, 1, 0]
    diag_b_indices = [3, 2, 3, 2]

    # todo make a class, not closures.

    # todo let `x` be lat/lon, then there is no need for norm constraint. would need to recompute cartesian coordinates
    #  in goal func and area constraint, so better merge `goal_func` and `gradient` together with `jac=True`.
    #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    def goal_func(x) -> float:
        """See EqualAreaParametricMeshNewtonIterator::goal_func."""
        # for vertex in vertices:
        #     for neighbor in neighbors(vertex):
        #         goal += 1 - dot(vertex, neighbor)

        points = x.reshape(sphere.shape)

        prod = np.prod(points[cells[:, edge_indices]], axis=-2).reshape(-1, 3)

        return (len(prod) - prod.sum()) / 2

    def gradient(x) -> np.ndarray:
        """See EqualAreaParametricMeshNewtonIterator::calc_gradient."""
        # for vertex in vertices:
        #     nbsum = [0, 0, 0]
        #
        #     for neighbor in neighbors(vertex):
        #         nbsum += neighbor
        #
        #     gradient[vertex] = dot(vertex, nbsum) * vertex - nbsum

        points = x.reshape(sphere.shape)

        nbsum = np.zeros_like(points)
        for u, v in edge_indices:
            nbsum[cells[:, u]] += points[cells[:, v]]
            nbsum[cells[:, v]] += points[cells[:, u]]

        dot = (nbsum * points).sum(-1)[:, None]
        grad = dot * points - nbsum

        return grad.ravel()

    # def hessian(_):
    #     """To replace 2-point Hessian in minimize() and reduce gradient count."""
    #     raise NotImplementedError

    def norms(x) -> np.ndarray:
        """To constrain norm of each vectors to 1."""
        points = x.reshape(sphere.shape)

        return np.linalg.norm(points, axis=1)

    def norms_jac(x) -> sp.dok_array:
        """To constrain norm of each vector to 1."""
        # Jacobian at any point is normal to the sphere at that point.

        points = x.reshape(sphere.shape)
        norm = np.linalg.norm(points, axis=1)
        normalized = points / norm[:, None]

        idxs = np.arange(len(points))

        res = sp.dok_array((len(points), len(x)))
        for k in range(points.shape[1]):
            res[idxs, points.shape[1] * idxs + k] = normalized[:, k]

        return res

    # def norms_hess(_):
    #     """To replace 2-point Hessian in NonlinearConstraint and reduce norms_jac count."""
    #     raise NotImplementedError

    def areas(x) -> np.ndarray:
        """See EqualAreaParametricMeshNewtonIterator::spher_area4."""
        points = x.reshape(sphere.shape)

        corners = points[cells, :]

        diag_a = corners[:, diag_a_indices]
        diag_b = corners[:, diag_b_indices]
        dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (
            diag_b * corners
        ).sum(-1)

        spats = np.linalg.det(corners[:, angle_det_indices])

        areas = -np.arctan2(dots, spats).sum(-1)
        areas = np.fmod(areas + 8.5 * np.pi, np.pi) - 0.5 * np.pi

        return areas  # noqa: RET504

    # def areas_jac(_) -> np.ndarray:
    #     """
    #     To allow sparse Jacobian and 2-point Hessian in NonlinearConstraint.
    #
    #     EqualAreaParametricMeshNewtonIterator must compute it somehow...
    #     """
    #     raise NotImplementedError

    # def areas_hess(_) -> np.ndarray:
    #     """To replace 2-point Hessian in NonlinearConstraint and reduce areas_jac count."""
    #     raise NotImplementedError

    result = scipy.optimize.minimize(
        fun=goal_func,
        x0=sphere.ravel(),
        jac=gradient,
        hess="2-point",
        constraints=[
            NonlinearConstraint(
                norms,
                1,
                1,
                jac=norms_jac,
                hess="2-point",
            ),
            NonlinearConstraint(
                areas,
                ideal_cell_area,
                ideal_cell_area,
                # jac=areas_jac,
                # hess='2-point',
            ),
        ],
        method="trust-constr",
        options={
            "xtol": 1e-3,
            "verbose": 2,
            "sparse_jacobian": True,
        },
    )
    final = result.x.reshape(sphere.shape)

    sphere = final.T
    lat = np.acos(sphere[2])
    lon = np.atan2(sphere[1], sphere[2])

    arr = numpy_support.numpy_to_vtk(lat)
    arr.SetName("Latitude")
    mesh.GetPointData().AddArray(arr)

    arr = numpy_support.numpy_to_vtk(lon)
    arr.SetName("Longitude")
    mesh.GetPointData().AddArray(arr)
