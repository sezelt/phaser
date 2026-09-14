
from pathlib import Path
import logging
import typing as t

import h5py
import numpy

from phaser.utils.image import apply_flips
from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from phaser.types import cast_length
from .. import LoadScanomaticProps, RawData


def load_scanomatic(args: None, props: LoadScanomaticProps) -> RawData:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    if not path.exists():
        raise ValueError(f"Couldn't find raw data at path {path}")

    with h5py.File(path, 'r') as f:
        # EMD data layout (py4DSTEM): (scan_y, scan_x, det_y, det_x)
        if (data := f.get('datacube_root/datacube/data')) is None:
            raise ValueError(f"Couldn't find data at 'datacube_root/datacube/data' in {path}")
        patterns = numpy.asarray(data[()])

        cal = f.get('datacube_root/metadatabundle/calibration')
        som = f.get('datacube_root/datacube/metadatabundle/SoM2k')

        raw_voltage = _leaf(cal, 'voltage')  # V
        if _leaf(cal, 'QR_flip'):
            logger.warning("EMD metadata has QR_flip=True, but the scanomatic reader ignores it")
        diff_step = props.diff_step or _leaf(cal, 'Q_pixel_size')  # mrad
        conv_angle = props.conv_angle or _leaf(cal, 'convergence_semiangle_mrad')  # mrad
        step_size = props.step_size or _leaf(cal, 'R_pixel_size')  # A
        meta_adu = _leaf(cal, 'ADU per electron')  # electrons/ADU
        som_rotation = _leaf(som, 'scan rotation')
        scan_rotation = props.scan_rotation or (
            float(numpy.rad2deg(-som_rotation))  # radians -> degrees, and fix sign
            if som_rotation is not None else None
        )

    voltage = props.kv * 1e3 if props.kv is not None else raw_voltage

    if voltage is None:
        raise ValueError("voltage must be present in EMD metadata or passed to 'raw_data' as 'kv'")
    if diff_step is None:
        raise ValueError("'diff_step' must be specified by metadata or passed to 'raw_data'")
    if conv_angle is None:
        logger.warning("Convergence angle not found in EMD metadata; specify 'conv_angle' in raw_data or 'init.probe'")

    wavelength = Electron(voltage).wavelength

    det_flips = props.det_flips or (True, False, False)  # defaults to typical EMPAD orientation
    logger.info(f"Loading with detector flips: {list(map(int, det_flips))} [y, x, transpose]")
    patterns = numpy.fft.ifftshift(apply_flips(patterns, det_flips), axes=(-1, -2))

    adu = props.adu or meta_adu or 12.11 * voltage / 1e3  # electrons/ADU; prop, metadata, or inferred: 12.11 per kV
    logger.info(f"Scaling patterns by ADU ({adu:.1f})")
    patterns /= adu

    a = wavelength / (diff_step * 1e-3)  # recip. pixel size -> 1 / real space extent
    sampling = Sampling(cast_length(patterns.shape[-2:], 2), extent=(a, a))

    mask = numpy.ones_like(patterns, shape=patterns.shape[-2:])

    probe_hook = {
        'type': 'focused',
        'conv_angle': conv_angle,
        'defocus': None,
    }
    scan_hook = {
        'type': 'raster',
        'shape': patterns.shape[:2],
        'step_size': step_size,
        'affine': None,
        'rotation': scan_rotation,
    }
    tilt_hook = None

    return {
        'patterns': patterns,
        'mask': mask,
        'sampling': sampling,
        'wavelength': wavelength,
        'probe_hook': probe_hook,
        'scan_hook': scan_hook,
        'tilt_hook': tilt_hook,
        'seed': None,
    }


def _leaf(group: t.Optional[h5py.Group], key: str) -> t.Any:
    """Read a scalar dataset from a group, or None if missing."""
    if group is None:
        return None
    d = group.get(key)
    if d is None:
        return None
    return d[()]
