"""Triangulated interface surfaces (3D isosurfaces).

ASCII legacy-VTK POLYDATA (what OpenFOAM's `surfaces` function object and
rtlbm write) is read natively.  Points declared `float` are rounded to float32
first, exactly as the vtk package's reader does, then promoted to float64, so
the geometry is bit-identical to the legacy fractal3d.load_mesh.  Binary or
XML files fall back to the optional vtk package.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class BBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def characteristic_length(self):
        """Geometric mean of the extents (sqrt of area if flat, length if a line)."""
        d = sorted([self.max_x - self.min_x, self.max_y - self.min_y, self.max_z - self.min_z], reverse=True)
        if d[2] > 1e-10:
            return (d[0] * d[1] * d[2]) ** (1 / 3)
        if d[1] > 1e-10:
            return (d[0] * d[1]) ** 0.5
        return d[0]


@dataclass
class TriangleMesh:
    vertices: np.ndarray      # (N, 3) float64
    triangles: np.ndarray     # (M, 3) int64
    bbox: BBox

    @classmethod
    def from_arrays(cls, vertices, triangles):
        v = np.asarray(vertices, dtype=np.float64)
        t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        bb = BBox(v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), v[:, 2].min(), v[:, 2].max())
        return cls(v, t, bb)

    @property
    def n_triangles(self):
        return len(self.triangles)

    def triangle_areas(self):
        v0, v1, v2 = (self.vertices[self.triangles[:, k]] for k in range(3))
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    @property
    def surface_area(self):
        return np.sum(self.triangle_areas())

    def triangle_vertices(self):
        """(M, 3, 3) float64: vertices of each triangle."""
        return self.vertices[self.triangles]


def _fan(conn, offsets):
    """Fan-triangulate polygons (connectivity + offsets); drop polygons with < 3 vertices."""
    tris = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        n = b - a
        if n == 3:
            tris.append(conn[a:b])
        elif n > 3:
            for j in range(1, n - 1):
                tris.append([conn[a], conn[a + j], conn[a + j + 1]])
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def _read_legacy_ascii(path):
    with open(path, 'rb') as fh:
        head = fh.read(256)
    if b'ASCII' not in head.split(b'\n', 3)[2].upper():
        return None
    with open(path) as fh:
        tokens = fh.read().split()
    i = tokens.index('POINTS')
    n_pts, dtype = int(tokens[i + 1]), tokens[i + 2].lower()
    pts = np.array(tokens[i + 3:i + 3 + 3 * n_pts], dtype=np.float64)
    if dtype == 'float':
        pts = pts.astype(np.float32).astype(np.float64)
    pts = pts.reshape(-1, 3)
    j = tokens.index('POLYGONS', i + 3 + 3 * n_pts)
    n_poly, size = int(tokens[j + 1]), int(tokens[j + 2])
    raw = np.array(tokens[j + 3:j + 3 + size], dtype=np.int64)
    conn, offsets, k = [], [0], 0
    for _ in range(n_poly):
        m = raw[k]
        conn.extend(raw[k + 1:k + 1 + m])
        offsets.append(offsets[-1] + m)
        k += m + 1
    if not conn:
        raise ValueError(f'no polygons in {path}')
    return TriangleMesh.from_arrays(pts, _fan(np.array(conn, dtype=np.int64), offsets))


def _read_with_vtk(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLPolyDataReader() if path.endswith('.vtp') else vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    pd = reader.GetOutput()
    polys = pd.GetPolys()
    if polys.GetNumberOfCells() == 0:
        raise ValueError(f'no polygons in {path}')
    return TriangleMesh.from_arrays(vtk_to_numpy(pd.GetPoints().GetData()),
                                    _fan(vtk_to_numpy(polys.GetConnectivityArray()),
                                         vtk_to_numpy(polys.GetOffsetsArray())))


def load_mesh(path):
    mesh = None if path.endswith('.vtp') else _read_legacy_ascii(path)
    if mesh is None:
        try:
            mesh = _read_with_vtk(path)
        except ImportError as e:
            raise ImportError(f'{path} is binary/XML VTK; install the optional vtk package') from e
    return mesh
