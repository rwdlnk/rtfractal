"""OpenFOAM-layout readers (OpenFOAM cases and rtlbm output).

Structured blockMesh grids only: x varies fastest, then y, then z.
Values are parsed to float64 exactly as float() would, so results match the
legacy line-by-line parsers bit for bit.
"""
import os
import re

import numpy as np


def _list_block(text, start):
    """Return the text between the '(' that follows `start` and its matching
    top-level ')' for an OpenFOAM `N ( ... )` list."""
    i = text.index('(', start)
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError('unterminated list')


def _numbers(block):
    return np.array(block.translate({ord('('): ' ', ord(')'): ' '}).split(), dtype=np.float64)


def _header_end(text):
    """Offset just past the FoamFile header dictionary (or 0)."""
    m = re.search(r'FoamFile\s*\{', text)
    if not m:
        return 0
    return text.index('}', m.end()) + 1


def read_points_nodes(case):
    """Unique sorted x and y node coordinates from constant/polyMesh/points."""
    path = os.path.join(case, 'constant', 'polyMesh', 'points')
    with open(path) as fh:
        text = fh.read()
    body = text[_header_end(text):]
    body = re.sub(r'//[^\n]*', '', body)
    m = re.search(r'^\s*(\d+)\s*$', body, re.M)
    if not m:
        raise ValueError(f'cannot parse {path}')
    xyz = _numbers(_list_block(body, m.end())).reshape(-1, 3)
    return np.unique(xyz[:, 0]), np.unique(xyz[:, 1])


def read_scalar_field(path, n_expected=None):
    """internalField of a volScalarField (nonuniform list or uniform value)."""
    with open(path) as fh:
        text = fh.read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)', text)
    if m:
        vals = _numbers(_list_block(text, m.end()))
    else:
        m = re.search(r'internalField\s+uniform\s+([0-9.eE+-]+)', text)
        if not m or n_expected is None:
            raise ValueError(f'cannot parse internalField in {path}')
        vals = np.full(n_expected, float(m.group(1)))
    if n_expected is not None and vals.size != n_expected:
        raise ValueError(f'{path}: {vals.size} values, expected {n_expected}')
    return vals


def read_vector_field(path, n_expected=None):
    """internalField of a volVectorField as an (n, 3) array."""
    with open(path) as fh:
        text = fh.read()
    m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*(\d+)', text)
    if m:
        vals = _numbers(_list_block(text, m.end())).reshape(-1, 3)
    else:
        m = re.search(r'internalField\s+uniform\s+\(([^)]*)\)', text)
        if not m or n_expected is None:
            raise ValueError(f'cannot parse internalField in {path}')
        vals = np.tile(np.array(m.group(1).split(), dtype=np.float64), (n_expected, 1))
    if n_expected is not None and len(vals) != n_expected:
        raise ValueError(f'{path}: {len(vals)} vectors, expected {n_expected}')
    return vals


def read_blockmesh(case):
    """(nx, ny, nz, bounds) from a single-block system/blockMeshDict, where
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)."""
    path = os.path.join(case, 'system', 'blockMeshDict')
    with open(path) as fh:
        content = fh.read()
    vm = re.search(r'vertices\s*\(\s*([\s\S]*?)\s*\)\s*;', content)
    bm = re.search(r'hex\s*\([^)]+\)\s*\((\d+)\s+(\d+)\s+(\d+)\)', content)
    if not vm or not bm:
        raise ValueError(f'cannot parse {path}')
    nx, ny, nz = (int(g) for g in bm.groups())
    coords = re.findall(r'\(\s*([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s*\)', vm.group(1))
    c = np.array(coords, dtype=float)
    bounds = (c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max(), c[:, 2].min(), c[:, 2].max())
    return nx, ny, nz, tuple(float(b) for b in bounds)


def read_fluid_properties(case):
    """(rho_heavy, rho_light, nu_heavy, nu_light, g).  'water' is the heavy
    (alpha = 1) fluid and 'air' the light one, whatever their real names."""
    rho_h, rho_l, nu_h, nu_l = 998.0, 998.0, 1.0e-6, 1.0e-6
    for name in ('water', 'air'):
        p = os.path.join(case, 'constant', f'physicalProperties.{name}')
        if not os.path.isfile(p):
            continue
        with open(p) as fh:
            content = fh.read()
        m_rho = re.search(r'rho\s+([\d.eE+-]+)', content)
        m_nu = re.search(r'nu\s+([\d.eE+-]+)', content)
        if name == 'water':
            rho_h = float(m_rho.group(1)) if m_rho else rho_h
            nu_h = float(m_nu.group(1)) if m_nu else nu_h
        else:
            rho_l = float(m_rho.group(1)) if m_rho else rho_l
            nu_l = float(m_nu.group(1)) if m_nu else nu_l
    g = 9.81
    p = os.path.join(case, 'constant', 'g')
    if os.path.isfile(p):
        with open(p) as fh:
            m = re.search(r'value\s*\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)', fh.read())
        if m:
            g = abs(float(m.group(2)))
    return rho_h, rho_l, nu_h, nu_l, g


def time_dirs(case, field='alpha.water'):
    """Sorted [(t, path)] of numeric time directories containing `field`."""
    out = []
    for item in os.listdir(case):
        p = os.path.join(case, item)
        if not os.path.isdir(p):
            continue
        try:
            t = float(item)
        except ValueError:
            continue
        if os.path.isfile(os.path.join(p, field)):
            out.append((t, p))
    return sorted(out, key=lambda e: e[0])


def interface_vtks(case):
    """Sorted [(t, path)] of postProcessing/extractInterface/<t>/interface.vtk."""
    d = os.path.join(case, 'postProcessing', 'extractInterface')
    if not os.path.isdir(d):
        return []
    out = []
    for item in os.listdir(d):
        try:
            t = float(item)
        except ValueError:
            continue
        p = os.path.join(d, item, 'interface.vtk')
        if os.path.isfile(p):
            out.append((t, p))
    return sorted(out, key=lambda e: e[0])
