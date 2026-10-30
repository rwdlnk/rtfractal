"""SOLA-VOF readers: RT<nx>x<ny>-<step>.vtk rectilinear grids and int*.in input files.

Snapshot time is step/1000 s (the legacy convention).  The int*.in parser is
the legacy solvof_input.py logic, kept verbatim in behaviour.
"""
import glob
import os
import re

import numpy as np

_VTK_RE = re.compile(r'RT(\d+)x(\d+)-(\d+)\.vtk$')


def find_vtk_files(case):
    """Sorted [(t, step, path)] and (nx, ny) from the file names."""
    files = []
    for p in glob.glob(os.path.join(case, 'RT*-*.vtk')):
        m = _VTK_RE.search(os.path.basename(p))
        if m:
            files.append((int(m.group(3)) / 1000.0, int(m.group(3)), p, int(m.group(1)), int(m.group(2))))
    if not files:
        raise FileNotFoundError(f'no RT*-*.vtk files in {case}')
    files.sort(key=lambda e: e[1])
    return [(t, s, p) for t, s, p, _, _ in files], (files[0][3], files[0][4])


def read_vtk(path, fields=None):
    """Rectilinear-grid VTK -> dict(x, y, f[, P, U, V]) on cell centres, arrays
    shaped (nx, ny) with x[:, j] constant in j (legacy layout).  `fields`
    limits which SCALARS are parsed (e.g. ('F',)); default all."""
    with open(path) as fh:
        text = fh.read()

    def block(start, count):
        return np.array(text[start:].split(None, count)[:count], dtype=np.float64)

    m = re.search(r'DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', text)
    dims = tuple(int(g) for g in m.groups())
    coords = {}
    for axis in 'XY':
        m = re.search(axis + r'_COORDINATES\s+(\d+)\s+\w+', text)
        coords[axis] = block(m.end(), int(m.group(1)))
    if 'CELL_DATA' not in text:
        raise ValueError(f'{path}: only CELL_DATA SOLA-VOF files are supported')
    x, y = coords['X'], coords['Y']
    nxc, nyc = dims[0] - 1, dims[1] - 1
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    xg, yg = np.meshgrid(xc, yc)
    out = {'x': xg.T, 'y': yg.T}
    for m in re.finditer(r'SCALARS\s+(\S+)[^\n]*\n\s*LOOKUP_TABLE[^\n]*\n', text):
        name = m.group(1)
        if fields is not None and name not in fields:
            continue
        v = block(m.end(), nxc * nyc)
        out['f' if name == 'F' else name] = v.reshape(nyc, nxc).T
    return out


# --------------------------------------------------------------------- int*.in
_KV_RE = re.compile(r'^([A-Za-z][A-Za-z0-9]*)\s*(?:\([^)]*\))?\s*=\s*(.+?)\s*$', re.IGNORECASE)
_FLOAT_KEYS = {'FLENG', 'FTIME', 'NU', 'NUC', 'EPSI', 'GX', 'GY', 'UI', 'VI', 'VELMX', 'OMG', 'ALPHA',
               'CSQ', 'AUTOT', 'SIGMA', 'CANGLE', 'RHOF', 'RHOFC', 'FLHT', 'XPL', 'YPB', 'XPR', 'YPT',
               'VORIG', 'SHVEL', 'DELT', 'TWFIN', 'PRTDT', 'PLTST1', 'PLTST2', 'PLTDT1', 'PLTDT2'}
_INT_KEYS = {'IPLOT', 'ICYL', 'IMOVY', 'WL', 'WR', 'WT', 'WB', 'ISYMPLT', 'ISURF10', 'NMAT', 'NPX',
             'NPY', 'NOBS', 'NDIS', 'IDIS', 'ISHVEL', 'IARE'}


def find_input_file(case):
    m = sorted(glob.glob(os.path.join(case, 'int*.in')))
    return m[0] if m else None


def parse_input(path):
    """Parse a SOLA-VOF int*.in file into a dict (upper-case keys)."""
    with open(path) as fh:
        lines = fh.readlines()
    p = {'_filepath': path}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        up = s.upper()
        if not s or s.startswith('!'):
            if 'VALUES FOR XL' in up or 'VALUES FOR YL' in up:
                key = 'XL' if 'XL' in up else 'YL'
                i += 2
                if i < len(lines):
                    p[key] = [float(v) for v in lines[i].split()]
                i += 1
                continue
            if 'DISTURBANCE PARAMETERS' in up:
                i += 2
                if i < len(lines):
                    v = lines[i].split()
                    if len(v) >= 4:
                        p['DIST_R'], p['DIST_H'] = float(v[0]), float(v[1])
                        p['DIST_ILOW'], p['DIST_IHIGH'] = int(v[2]), int(v[3])
                i += 1
                continue
            i += 1
            continue
        low = s.lower()
        if low.startswith(('-nkx', '-xc', '-yc', '-ifl')):
            i += 1
            if i < len(lines):
                v = lines[i].split()
                if low.startswith('-nkx'):
                    p['NKX'], p['NKY'], p['NONU'] = int(v[0]), int(v[1]), int(v[2])
                elif low.startswith('-xc'):
                    p['XC'], p['NXL'], p['NXR'], p['DXMN'] = float(v[0]), int(v[1]), int(v[2]), float(v[3])
                elif low.startswith('-yc'):
                    p['YC'], p['NYL'], p['NYR'], p['DYMN'] = float(v[0]), int(v[1]), int(v[2]), float(v[3])
                else:
                    p['IFL'], p['IFR'], p['JFB'], p['JFT'] = (int(x) for x in v[:4])
            i += 1
            continue
        m = _KV_RE.match(s)
        if m:
            key, val = m.group(1).upper(), m.group(2).strip()
            try:
                p[key] = float(val) if key in _FLOAT_KEYS else int(val) if key in _INT_KEYS else val
            except ValueError:
                p[key] = val
        i += 1
    return p


def derive_physics(p):
    rhof, rhofc = p['RHOF'], p['RHOFC']
    xl, yl = p.get('XL', [0.0, 0.4]), p.get('YL', [0.0, 0.5])
    H = yl[-1] - yl[0]
    return dict(A=(rhof - rhofc) / (rhof + rhofc), g=abs(p['GY']), H=H, L=xl[-1] - xl[0],
                H0=p.get('DIST_H', p.get('YC', H / 2.0)),
                NX=p['NXL'] + p['NXR'], NY=p['NYL'] + p['NYR'], dx=p['DXMN'], dy=p['DYMN'],
                RHOF=rhof, RHOFC=rhofc, NU=p.get('NU', 1.0e-6))
