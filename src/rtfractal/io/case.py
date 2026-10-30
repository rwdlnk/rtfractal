"""One interface to every supported case layout.

    case = open_case(path)
    case.dim            2 or 3
    case.format         'solavof' | 'openfoam'
    case.physics        RTPhysics
    case.H0             initial interface height
    case.snapshots      [Snapshot(t, key, ...)]
    case.read_2d(s)     dict(x, y, f[, U, V, P]) on cell centres, arrays (nx, ny)
    case.read_alpha_3d(s), case.read_U_3d(s), case.interface_mesh(s)

Conventions are inherited from the legacy analyzers so results reproduce:
SOLA-VOF H0 comes from the int*.in disturbance height; OpenFOAM 2D uses
H0 = H/2; OpenFOAM 3D uses the blockMesh mid-height.  rtlbm output is an
OpenFOAM-layout case.
"""
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..physics import RTPhysics
from . import openfoam as of
from . import solavof as sv
from .mesh import load_mesh


@dataclass
class Snapshot:
    t: float
    time_idx: int                 # key used in output file names
    path: Optional[str] = None    # VTK file (SOLA-VOF) or time directory (OpenFOAM)
    iface: Optional[str] = None   # 3D interface mesh


@dataclass
class Grid3D:
    nx: int
    ny: int
    nz: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.y_max - self.y_min) / self.ny
        self.dz = (self.z_max - self.z_min) / self.nz

    @property
    def y_centers(self):
        return np.linspace(self.y_min + self.dy / 2, self.y_max - self.dy / 2, self.ny)


def detect(path):
    if glob.glob(os.path.join(path, 'RT*-*.vtk')):
        return 'solavof', 2
    bmd = os.path.join(path, 'system', 'blockMeshDict')
    if os.path.isfile(bmd):
        nz = of.read_blockmesh(path)[2]
        if nz > 1:
            return 'openfoam', 3
    if os.path.isfile(os.path.join(path, 'constant', 'polyMesh', 'points')):
        return 'openfoam', 2
    return None, None


def open_case(path):
    path = os.path.abspath(path)
    fmt, dim = detect(path)
    if fmt == 'solavof':
        return SolaVofCase(path)
    if fmt == 'openfoam' and dim == 2:
        return OpenFOAM2DCase(path)
    if fmt == 'openfoam' and dim == 3:
        return OpenFOAM3DCase(path)
    raise ValueError(f'{path}: not a SOLA-VOF (RT*-*.vtk) or OpenFOAM-layout case')


class SolaVofCase:
    format, dim = 'solavof', 2

    def __init__(self, path, input_file=None):
        self.path = path
        files, (self.nx, self.ny) = sv.find_vtk_files(path)
        inp = input_file or sv.find_input_file(path)
        if inp is None:
            raise FileNotFoundError(f'no int*.in in {path}')
        self.params = sv.derive_physics(sv.parse_input(inp))
        p = self.params
        self.physics = RTPhysics(A=p['A'], g=p['g'], H=p['H'], L=p['L'])
        self.H0 = p['H0']
        self.snapshots = [Snapshot(t, step, fp) for t, step, fp in files]

    def read_2d(self, snap):
        return sv.read_vtk(snap.path)


class OpenFOAM2DCase:
    format, dim = 'openfoam', 2

    def __init__(self, path):
        self.path = path
        xn, yn = of.read_points_nodes(path)
        self.x_nodes, self.y_nodes = xn, yn
        self.nx, self.ny = len(xn) - 1, len(yn) - 1
        L, H = float(xn[-1] - xn[0]), float(yn[-1] - yn[0])
        rho_h, rho_l, nu_h, nu_l, g = of.read_fluid_properties(path)
        A = (rho_h - rho_l) / (rho_h + rho_l) if rho_h + rho_l > 0 else 0.0
        self.params = dict(A=A, g=g, H=H, L=L, NX=self.nx, NY=self.ny, dx=L / self.nx, dy=H / self.ny,
                           RHOF=rho_h, RHOFC=rho_l, NU=nu_h, NUC=nu_l)
        self.physics = RTPhysics(A=A, g=g, H=H, L=L)
        self.H0 = H / 2.0
        self.snapshots = [Snapshot(t, int(t * 1000), d) for t, d in of.time_dirs(path)]

    def read_2d(self, snap):
        xc = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        yc = 0.5 * (self.y_nodes[:-1] + self.y_nodes[1:])
        n = self.nx * self.ny
        f = of.read_scalar_field(os.path.join(snap.path, 'alpha.water'), n)
        xg, yg = np.meshgrid(xc, yc)
        out = {'x': xg.T, 'y': yg.T, 'f': f.reshape(self.ny, self.nx).T}
        up = os.path.join(snap.path, 'U')
        if os.path.isfile(up):
            u = of.read_vector_field(up, n)
            out['U'] = u[:, 0].reshape(self.ny, self.nx).T
            out['V'] = u[:, 1].reshape(self.ny, self.nx).T
        pp = os.path.join(snap.path, 'p_rgh')
        if os.path.isfile(pp):
            out['P'] = of.read_scalar_field(pp, n).reshape(self.ny, self.nx).T
        return out


class OpenFOAM3DCase:
    format, dim = 'openfoam', 3

    def __init__(self, path):
        self.path = path
        nx, ny, nz, b = of.read_blockmesh(path)
        self.grid = Grid3D(nx, ny, nz, *b)
        g3 = self.grid
        H, L, W = g3.y_max - g3.y_min, g3.x_max - g3.x_min, g3.z_max - g3.z_min
        rho_h, rho_l, nu_h, nu_l, g = of.read_fluid_properties(path)
        A = (rho_h - rho_l) / (rho_h + rho_l) if rho_h + rho_l > 0 else 0.0
        self.params = dict(A=A, g=g, H=H, L=L, W=W, H0=(g3.y_min + g3.y_max) / 2.0,
                           rho_heavy=rho_h, rho_light=rho_l, nu_heavy=nu_h, nu_light=nu_l)
        self.physics = RTPhysics(A=A, g=g, H=H, L=L, W=W)
        self.H0 = self.params['H0']
        if os.path.isdir(os.path.join(path, 'processor0')) and not of.time_dirs(path):
            raise NotImplementedError('decomposed 3D case: run reconstructPar first')
        dirs = {round(t, 9): d for t, d in of.time_dirs(path)}
        self.snapshots = [Snapshot(t, int(round(t * 1000)), dirs.get(round(t, 9)), vtk)
                          for t, vtk in of.interface_vtks(path)]

    def read_alpha_3d(self, snap):
        g3 = self.grid
        return of.read_scalar_field(os.path.join(snap.path, 'alpha.water'),
                                    g3.nx * g3.ny * g3.nz).reshape(g3.nz, g3.ny, g3.nx)

    def read_U_3d(self, snap):
        p = os.path.join(snap.path, 'U')
        if not os.path.isfile(p):
            return None
        g3 = self.grid
        return of.read_vector_field(p, g3.nx * g3.ny * g3.nz).reshape(g3.nz, g3.ny, g3.nx, 3)

    def interface_mesh(self, snap):
        return load_mesh(snap.iface)
