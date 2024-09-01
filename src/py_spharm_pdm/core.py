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


def initial_parameterization(
    data: vtk.vtkImageData,
) -> vtk.vtkPolyData:
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

            # if current idx is west of the meridian
            # noinspection PyUnreachableCode
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


def refine_parameterization(
    mesh: vtk.vtkPolyData,
    maxiter: int = 1000,
):
    # edges = build_edges(mesh)
    # matrix = build_adjacency_matrix(edges)

    pd: vtk.vtkPointData = mesh.GetPointData()
    lat = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Latitude"))
    lon = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Longitude"))

    # todo instead of EDGES, maybe get tuples from sparse adjacency matrix?
    #  maybe some multiplication by the adjacency matrix?

    # cd: vtk.vtkCellData = mesh.GetCellData()

    cells = np.zeros((mesh.GetNumberOfCells(), 3), dtype="i")
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

    edge_indices = [[0, 1], [1, 2], [2, 0]]

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

        areas = np.linalg.det(corners)

        # diag_a = corners[:, diag_a_indices]
        # diag_b = corners[:, diag_b_indices]
        # dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (
        #     diag_b * corners
        # ).sum(-1)

        # spats = np.linalg.det(corners[:, angle_det_indices])

        # areas = -np.arctan2(dots, spats).sum(-1)

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
            # "verbose": 2,
            "sparse_jacobian": True,
            "maxiter": maxiter,
        },
    )

    sphere_f = result.x.reshape(sphere.shape).T
    lat_f = np.arccos(sphere_f[2])
    lon_f = np.atan2(sphere_f[1], sphere_f[0])

    arr = numpy_support.numpy_to_vtk(lat)
    arr.SetName("Latitude_0")
    pd.AddArray(arr)

    arr = numpy_support.numpy_to_vtk(lon)
    arr.SetName("Longitude_0")
    pd.AddArray(arr)

    arr = numpy_support.numpy_to_vtk(lat_f)
    arr.SetName("Latitude")
    pd.AddArray(arr)

    arr = numpy_support.numpy_to_vtk(lon_f)
    arr.SetName("Longitude")
    pd.AddArray(arr)

    return result


def fit_spharms(mesh: vtk.vtkPolyData):
    pd: vtk.vtkPointData = mesh.GetPointData()

    lat = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Latitude"))
    lon = numpy_support.vtk_to_numpy(pd.GetAbstractArray("Longitude"))
