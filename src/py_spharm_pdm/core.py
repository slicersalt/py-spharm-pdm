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

    # todo make a class, not closures.

    # todo let `x` be lat/lon, then there is no need for norm constraint. would need to recompute cartesian coordinates
    #  in goal func and area constraint, so better merge `goal_func` and `gradient` together with `jac=True`.
    #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    refiner = Refiner(cells)

    result = scipy.optimize.minimize(
        fun=refiner.goal_func,
        x0=sphere.ravel(),
        jac=True,
        hess="2-point",
        constraints=[
            NonlinearConstraint(
                refiner.norms,
                1,
                1,
                jac=refiner.norms_jac,
                hess="2-point",
            ),
            NonlinearConstraint(
                refiner.areas,
                ideal_cell_area,
                ideal_cell_area,
                jac=refiner.areas_jac,
                hess="2-point",
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

    _ = final
    # todo reconstruct lat, lon from final; apply to the mesh.


class Refiner:
    EDGE_INDICES = ((0, 1), (1, 2), (2, 3), (3, 0))

    ANGLE_DET_INDICES = (
        (3, 0, 1),
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 0),
    )

    DIAG_A_INDICES = (1, 0, 1, 0)
    DIAG_B_INDICES = (3, 2, 3, 2)

    def __init__(self, cells: np.typing.NDArray):
        self.cells = cells

    def goal_func(self, x) -> float:
        """See EqualAreaParametricMeshNewtonIterator::goal_func."""
        # for vertex in vertices:
        #     for neighbor in neighbors(vertex):
        #         goal += 1 - dot(vertex, neighbor)

        import torch

        x = torch.tensor(x, requires_grad=True)
        points = x.reshape(-1, 3)
        prod = torch.prod(points[self.cells[:, self.EDGE_INDICES]], axis=-2).reshape(
            -1, 3
        )
        goal = (len(prod) - prod.sum()) / 2
        goal.backward()
        return goal.detach(), x.grad.detach()

    def hessian(self, _):
        """To replace 2-point Hessian in minimize() and reduce gradient count."""
        raise NotImplementedError

    def torch_norms(self, x):
        import torch

        return torch.linalg.norm(x.reshape(-1, 3), axis=1)

    def norms(self, x) -> np.ndarray:
        """To constrain norm of each vectors to 1."""
        import torch

        x = torch.tensor(x)
        return self.torch_norms(x).detach()

    def norms_jac(self, x) -> sp.dok_array:
        """To constrain norm of each vector to 1."""
        # Jacobian at any point is normal to the sphere at that point.

        import torch

        x = torch.tensor(x, requires_grad=True)
        (jac,) = torch.autograd.functional.jacobian(self.torch_norms, (x,))
        return jac.detach()

    # def norms_hess(self, x, v):
    #     """To replace 2-point Hessian in NonlinearConstraint and reduce norms_jac count."""
    #     # since norms is technically vector-valued function, not sure the hessian can really be computed here.
    #     # the hessian tensor of a vector-valued function could in theory be computed, but not easily with torch.
    #     raise NotImplemented

    def torch_areas(self, x):
        import torch

        points = x.reshape(-1, 3)
        corners = points[self.cells, :]

        diag_a = corners[:, self.DIAG_A_INDICES]
        diag_b = corners[:, self.DIAG_B_INDICES]
        dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (
            diag_b * corners
        ).sum(-1)

        spats = torch.linalg.det(corners[:, self.ANGLE_DET_INDICES])
        areas = -torch.arctan2(dots, spats).sum(dim=-1)
        return torch.fmod(areas + 8.5 * torch.pi, torch.pi) - 0.5 * torch.pi

    def areas(self, x) -> np.ndarray:
        """See EqualAreaParametricMeshNewtonIterator::spher_area4."""
        import torch

        x = torch.tensor(x)
        return self.torch_areas(x)

    def areas_jac(self, x) -> np.ndarray:
        """
        To allow sparse Jacobian and 2-point Hessian in NonlinearConstraint.

        EqualAreaParametricMeshNewtonIterator must compute it somehow...
        """

        import torch

        x = torch.tensor(x, requires_grad=True)
        (jac,) = torch.autograd.functional.jacobian(self.torch_areas, (x,))
        return jac.detach()

    # def areas_hess(self, x, v) -> np.ndarray:
    #     """To replace 2-point Hessian in NonlinearConstraint and reduce areas_jac count."""
    #     # since areas is technically vector-valued function, not sure the hessian can really be computed here.
    #     # the hessian tensor of a vector-valued function could in theory be computed, but not easily with torch.
    #     raise NotImplemented
