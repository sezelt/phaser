"""
Hooks for initializing a reconstruction from a Cyclomatic export.

`Cyclomatic` (a separate package) fits a scanned dataset and can export the
corrected scan positions and the fitted probe aberrations into a single HDF5
file (format `"cyclomatic-scan-positions"`, version 1). The two hook variants
here — a `cyclomatic` scan (reading `/positions`) and a `cyclomatic` probe
(reading `/aberrations`) — both point at the same file, and are used together to
initialize a reconstruction directly from it.
"""

import logging
import re
import typing as t

import h5py
from frozendict import frozendict

from phaser.state import ScanState, PixelatedProbeState
from phaser.types import KrivanekComplex
from phaser.utils.num import cast_array_module
from phaser.utils.optics import make_focused_probe

from . import CyclomaticScanProps, CyclomaticProbeProps, ScanHookArgs, ProbeHookArgs

_FORMAT = "cyclomatic-scan-positions"
_FORMAT_VERSION = 1

_COS_RE = re.compile(r'C(\d+),(\d+)_cos')
_SIN_RE = re.compile(r'C(\d+),(\d+)_sin')


def _open_cyclomatic(path) -> h5py.File:
    """
    Open a Cyclomatic export and verify its root attributes before any parsing.

    Use as a context manager. Raises `ValueError` if the file is not a
    recognized Cyclomatic scan-positions file (format or version mismatch).
    """
    f = h5py.File(path, 'r')
    try:
        if f.attrs.get("format") != _FORMAT:
            raise ValueError(
                f"Not a Cyclomatic scan-positions file: root 'format' attr is "
                f"{f.attrs.get('format')!r}, expected {_FORMAT!r}"
            )
        if f.attrs.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Cyclomatic scan-positions format_version "
                f"{f.attrs.get('format_version')}, expected {_FORMAT_VERSION}"
            )
    except Exception:
        f.close()
        raise
    return f


def _read_aberrations(f: h5py.File) -> tuple[tuple[KrivanekComplex, ...], t.Optional[float]]:
    """
    Read the fitted Krivanek aberrations from the `/aberrations` group.

    The group holds one `C{n,m}_cos` / `C{n,m}_sin` attribute pair per fitted
    coefficient, plus `units` and `fit_residual`. The set of coefficients is
    fit-basis-dependent, so the attributes are enumerated rather than hard-coded.

    Returns the aberrations as `KrivanekComplex` values (a missing cos/sin part
    defaults to 0.0) and the `fit_residual` (or None if not present).
    """
    if "aberrations" not in f:
        return (), None

    a = f["aberrations"].attrs
    # (n, m) -> [cos, sin]
    coefs: dict[tuple[int, int], list[float]] = {}

    for key in a:
        if (match := _COS_RE.fullmatch(key)) is not None:
            n, m = int(match.group(1)), int(match.group(2))
            coefs.setdefault((n, m), [0.0, 0.0])[0] = float(a[key])
        elif (match := _SIN_RE.fullmatch(key)) is not None:
            n, m = int(match.group(1)), int(match.group(2))
            coefs.setdefault((n, m), [0.0, 0.0])[1] = float(a[key])

    aberrations = tuple(
        KrivanekComplex(n, m, val=complex(cos, sin))
        for ((n, m), (cos, sin)) in sorted(coefs.items())
    )

    residual = float(a["fit_residual"]) if "fit_residual" in a else None
    return aberrations, residual


def cyclomatic_scan(args: ScanHookArgs, props: CyclomaticScanProps) -> ScanState:
    """
    Load corrected scan positions from a Cyclomatic export.

    The `/positions` dataset has shape `(Nx, Ny, 2)` with the last axis holding
    `(x, y)` pairs, in Å. It is reoriented to phaser's convention —
    `(ny, nx, 2)` with the last axis `(y, x)` — before being used as the scan.
    """
    xp = cast_array_module(args['xp'])
    logger = logging.getLogger(__name__)

    with _open_cyclomatic(props.path) as f:
        if "positions" not in f:
            raise ValueError(f"Cyclomatic file {props.path} has no 'positions' dataset")
        ds = f["positions"]
        if ds.attrs.get("units") != "A":
            raise ValueError(
                f"Cyclomatic 'positions' must have units 'A', got {ds.attrs.get('units')!r}"
            )
        pos = ds[...]

    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"Cyclomatic 'positions' must have shape (Nx, Ny, 2), got {pos.shape}")

    # (Nx, Ny, 2) with last axis (x, y) -> (ny, nx, 2) with last axis (y, x)
    pts = pos.transpose(1, 0, 2)
    pts = pts[:, :, [1, 0]]

    if props.remove_offset:
        pts = pts - pts.mean(axis=(0, 1), keepdims=True)
    if props.scale is not None:
        pts = pts * props.scale

    logger.info(f"Making Cyclomatic scan, shape {pts.shape[:-1]},"
                f" offset removed {props.remove_offset},"
                f" scale {props.scale}")

    scan = xp.asarray(pts, dtype=args['dtype'])

    return ScanState(
        scan, xp.asarray(scan, copy=True), tilt=None, meta=frozendict(
            type='cyclomatic',
        )
    )


def cyclomatic_probe(args: ProbeHookArgs, props: CyclomaticProbeProps) -> PixelatedProbeState:
    """
    Build a focused probe initialized from the aberrations fitted by Cyclomatic.

    Each `C{n,m}` pair from the `/aberrations` group is used as a Krivanek
    aberration. `C1,0` acts as the defocus (its contribution to the wavefront
    error is `theta2 * C1,0 / 2`), so the probe is built with `defocus=0.0` and
    `C1,0` passed as an ordinary aberration. `conv_angle` (not stored in the
    file) is required to set the probe aperture.
    """
    logger = logging.getLogger(__name__)

    if props.conv_angle is None:
        raise ValueError("Probe 'conv_angle' must be specified by metadata or manually")

    with _open_cyclomatic(props.path) as f:
        aberrations, residual = _read_aberrations(f)

    if not aberrations:
        logger.warning("Cyclomatic probe: no aberrations found in file; "
                       "building an aberration-free probe")
    else:
        residual_s = f", fit residual {residual} A" if residual is not None else ""
        s = '\n'.join(f"  {ab!r}" for ab in aberrations)
        logger.info(f"Making Cyclomatic probe, conv_angle {props.conv_angle} mrad{residual_s}\n"
                    f"Aberrations:\n{s}")

    sampling = args['sampling']
    ky, kx = sampling.recip_grid(dtype=args['dtype'], xp=args['xp'])
    probe = make_focused_probe(
        ky, kx, args['wavelength'],
        props.conv_angle, defocus=0.0, aberrations=aberrations
    )
    return PixelatedProbeState(sampling, probe)
