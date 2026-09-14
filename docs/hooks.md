# Input files and hooks

This page documents how phaser reads and parses its input files, and lists the
arguments accepted by every built-in hook.

## How input files are parsed

Plans are YAML files (a file may contain multiple YAML documents separated by
`---`, each one an independent plan). The CLI entry points are:

```sh
phaser run <plan.yaml> [--raise-on-warn]   # parse and execute
phaser validate <plan.yaml> [--json]       # parse only, report errors
```

The YAML is parsed strictly: every field is type-checked, unknown fields are
rejected, and all errors in the file are collected and reported at once
(`--raise-on-warn` turns warnings into errors).

### Top-level structure

Every plan must have `file_type: phaser_plan` (the default, so usually omitted)
and `version` exactly `1.0` (the default).

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `file_type` | `'phaser_plan'` | `'phaser_plan'` | |
| `version` | str | `'1.0'` | Must be exactly `1.0`. |
| `name` | str | **required** | Name of the reconstruction. |
| `backend` | `'cupy' \| 'jax' \| 'torch' \| 'numpy'` | auto | Compute backend. |
| `device` | str | auto | Device string passed to the backend. |
| `dtype` | `'float32' \| 'float64'` | `'float32'` | Working precision. |
| `wavelength` | float (Å) | inferred | Overrides the value from the raw data metadata. |
| `raw_data` | raw data hook | **required** | Loads the 4D-STEM dataset (see below). |
| `post_load` | list of post-load hooks | `[]` | Applied to the raw patterns, in order. |
| `init` | init plan | `{}` | Initializes probe, object, scan, and tilt (see below). |
| `post_init` | list of post-init hooks | `[]` | Applied to the state after initialization. |
| `slices` | [slices](#slices) | `None` | Default slices for all engines. |
| `engines` | list of engine hooks | **required** | Engines run in sequence. |

### The init plan

`init` has these fields:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `state` | path | `None` | Path to an HDF5 state file to resume from (a previously saved partial state). |
| `scan` | [scan hook](#scan-raster), empty dict `{}`, or `None` | `None` | An empty dict means "use the scan from the raw data metadata". |
| `tilt` | [tilt hook](#tilt-global), empty dict `{}`, or `None` | `None` | Same as above. |
| `probe` | [probe hook](#probe-focused), empty dict `{}`, or `None` | `None` | Same as above. |
| `object` | [object hook](#object-random) | `None` | The object to reconstruct (defaults to a random object if omitted). |

The raw-data loaders (e.g. `empad`, `gatan`, `nion`) usually fill in `probe`,
`scan`, and `tilt` from the file's metadata; when they do, the corresponding
`init` entry can be left as `{}` (or omitted). The `manual` loader fills in
none of them, so `init` must supply them explicitly.

### Writing hooks

Most fields in a plan are *hooks* — references to a function plus its
arguments. In YAML a hook can be written as:

1. **A bare string** naming a built-in hook, with no arguments:
   ```yaml
   post_init:
     - drop_nans
   ```
2. **A mapping** with a `type` key (the hook name) and the remaining keys as
   its arguments:
   ```yaml
   raw_data:
     type: empad
     path: data.raw
   ```
3. **A bare string or `type` value of the form `module.path:func`** — a custom
   user-defined function. It is imported and called as
   `func(args, props=...)`; its arguments are *not* type-checked, so only
   built-in hook names get argument validation:
   ```yaml
   post_load:
     - type: my_pkg.my_module:my_hook
       foo: 1
   ```

For built-in hooks, unknown or mistyped arguments are rejected at parse time.
A hook mapping without a `type` key is an error.

Every built-in hook is invoked as `f(args, props=...)`, where `args` depends on
the hook's position in the pipeline (e.g. `PostLoadHook`s receive the raw-data
dict, `PostInitHook`s and engine hooks receive the reconstruction state).

---

## Raw data hooks

`raw_data` selects which loader reads the dataset. All loaders return a dict
with at least `patterns` (float32, shape `(ny, nx, ky, kx)`, y decreasing
downwards, zero-frequency centered), `mask` (a 2D detector mask), `sampling`
(diffraction pixel size in mrad/px), and `wavelength` (Å); empad/gatan/nion/
scanomatic additionally fill in `probe_hook`, `scan_hook`, and sometimes
`tilt_hook` from the metadata.

### `empad`

```yaml
raw_data:
  type: empad
  path: ...
```

**File format.** An EMPAD dataset is a raw binary file of float32 values where
each pattern is 130 rows × 128 columns (128 data rows plus 2 junk rows, which
are cropped on load). The file must contain a whole number of patterns
(file size must be divisible by `130 × 128`). If `path` is a `.json` file it
is the EMPAD metadata sidecar (EMPAD v1 metadata only — v2 is not supported);
otherwise `path` is the binary itself and the metadata JSON is found next to
it. Scan shape is inferred from an `x{nx}_y{ny}` pattern in the filename. An
EMPAD XML file can be converted to the JSON metadata with
`EmpadMetadata.from_xml`.

**Arguments.**

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | path | **required** | Path to the raw binary or its `.json` metadata. |
| `diff_step` | float (mrad/px) | from metadata | Overrides the metadata value. |
| `kv` | float (kV) | from metadata | Overrides the accelerating voltage (V = kv × 1000). |
| `adu` | float (electrons/count) | from metadata | ADU scaling factor. |
| `det_flips` | [bool, bool, bool] | `(True, False, False)` | `(flip_y, flip_x, transpose)` applied to the detector patterns. |

The metadata file (`.json`, `EmpadMetadata`) contains: `voltage` (V),
`conv_angle` (mrad), `defocus` (m, positive = overfocus), `camera_length` (m),
`diff_step` (mrad/px), `scan_rotation` (deg), `scan_shape` (x, y),
`scan_fov` (m), `scan_step` (m/px), `exposure_time` (s), `beam_current` (A),
`adu` (electrons/count), `det_flips`, `det_rotation` (deg), optional
`scan_correction` 2×2 matrix, optional `scan_positions` override, and an
optional `crop` region `(min_y, max_y, min_x, max_x)`.

### `gatan`

```yaml
raw_data:
  type: gatan
  path: data.dm4
```

**File format.** A single Gatan Digital Micrograph `.dm4` file, read with the
`rsciio` package (installed as an extra dependency of this hook). Metadata —
voltage, defocus, camera length, diffraction and real-space calibrations, scan
rotation, exposure, and single-electron scaling (`e_scaling`) and
`background_offset` from the ImageData brightness calibration — is extracted
from the file's tags.

**Arguments.**

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | path | **required** | Path to the `.dm4` file. |
| `diff_step` | float (mrad/px) | from metadata | Overrides the value read from the file. |
| `kv` | float (kV) | from metadata | Overrides the accelerating voltage. |
| `adu` | float (electrons/count) | from metadata | ADU scaling factor (overrides `e_scaling`). |

### `nion`

```yaml
raw_data:
  type: nion
  path: data.nion.zip
  diff_step: 0.22
```

**File format.** A zip archive containing `data.npy` (the 4D data) and
`metadata.json` (scan size, scan rotation, field of view, high tension,
defocus, spatial calibrations, and camera processing flags).

**Arguments.**

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | path | **required** | Path to the zip file. |
| `diff_step` | float (mrad/px) | **required** | Nion files do not record the diffraction pixel size. |
| `detector_rotation_offset` | float (deg) | `0` | Additional detector rotation applied to the scan rotation. |

The accelerating voltage comes from the metadata's `high_tension`, so no
`probe` initialization is provided by this loader — `init.probe` is required.

### `manual`

```yaml
raw_data:
  type: manual
  path: patterns.npy
  det_shape: [128, 128]
  diff_step: 0.22
  kv: 200
```

**File format.** By file extension:

- `.npy` / `.npz` — loaded with `numpy.load`
- `.tif` / `.tiff` — loaded with `tifffile`
- `.h5` / `.hdf5` / `.emd` — HDF5, read with `h5py`. The dataset key is
  `key` if given, otherwise inferred as the first of `dp`, `data`, or
  `datacube_root/datacube/data` that is at least 3-dimensional.
- any other extension — raw binary. Requires `det_shape`; the file size (minus
  `offset`) must be divisible by `pattern_size + gap`.

The array must be at least 3-dimensional and float (non-float input is cast to
float32; complex is rejected).

**Arguments.**

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | path | **required** | Path to the data file. |
| `det_shape` | [int, int] | `None` | Detector shape `(ny, nx)` after flips. Required for raw binary files. |
| `dtype` | str | `'float32'` | NumPy dtype (e.g. `'float32'`). Raw binary files only. |
| `gap` | int (bytes) | `0` | Bytes between patterns. Raw binary files only. |
| `offset` | int (bytes) | `0` | Bytes before the first pattern. Raw binary files only. |
| `key` | str | inferred | HDF5 dataset key (split on both `.` and `/`). |
| `diff_step` | float (mrad/px) | **required** | Diffraction pixel size. |
| `kv` | float (kV) | — | Accelerating voltage in kV. |
| `wavelength` | float (Å) | — | Wavelength. Exactly one of `kv` or `wavelength` is required. |
| `adu` | float (electrons/count) | — | ADU scaling factor (a warning is issued if omitted). |
| `det_flips` | [bool, bool, bool] | `(False, False, False)` | `(flip_y, flip_x, transpose)`. |
| `fftshifted` | bool | `False` | `True` if the patterns already have zero-frequency in the center (otherwise an ifftshift is applied). |

This loader provides no probe, scan, or tilt, so `init.probe`, `init.scan`,
and `init.tilt` must all be specified in the plan.

### `scanomatic`

```yaml
raw_data:
  type: scanomatic
  path: scan.emd
  adu: 1.0
```

**File format.** A py4DSTEM EMD file (HDF5) as produced by scan-o-matic
(scanomatic2000). The 4D data is read from `datacube_root/datacube/data`
(shape `(scan_y, scan_x, det_y, det_x)`). Calibration values are read from
the `datacube_root/metadatabundle/calibration` group: `voltage` (volts),
`Q_pixel_size` (mrad/px), `convergence_semiangle_mrad` (mrad),
`R_pixel_size` (Å/px), `ADU per electron`, and `QR_flip`. The scan rotation
is read from the
`SoM2k` metadata group (`.../datacube/metadatabundle/SoM2k/scan rotation`,
in radians, converted to degrees). If the calibration metadata has
`QR_flip = True`, a warning is logged noting that the reader ignores it.

**Arguments.**

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | path | **required** | Path to the `.emd` file. |
| `diff_step` | float (mrad/px) | from metadata | Overrides `Q_pixel_size`. |
| `kv` | float (kV) | from metadata | Overrides the accelerating voltage (V = kv × 1000; the metadata voltage is in volts). |
| `adu` | float (electrons/count) | from metadata | ADU scaling factor. Overrides the metadata `ADU per electron`; defaults to 12.11 per kV of accelerating voltage if that is also absent. |
| `det_flips` | [bool, bool, bool] | `(True, False, False)` | `(flip_y, flip_x, transpose)`. Defaults to the typical EMPAD orientation. |
| `conv_angle` | float (mrad) | from metadata | Overrides `convergence_semiangle_mrad`. |
| `step_size` | float (Å) | from metadata | Overrides `R_pixel_size`. |
| `scan_rotation` | float (deg CCW) | from metadata | Overrides `scan rotation`. |

The loader fills in `probe_hook` (`focused`, with the metadata convergence
angle) and `scan_hook` (`raster`, with the metadata step size and scan
rotation). The defocus is not recorded in the EMD metadata, so it is left
unspecified — `init.probe` must supply a `defocus` (any other probe fields
not set there fall back to the metadata values).

---

## Post-load hooks

Hooks in the `post_load` list transform the raw patterns. Each receives the
raw-data dict and returns the modified patterns (and updated mask where
relevant).

### `crop_data`

Crops the detector patterns.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `crop` | [int, int, int, int] | **required** | `(y_i, y_f, x_i, x_f)`, Python-style slicing. |

### `poisson`

Adds Poisson + Gaussian noise (deterministic given the plan's seed).

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `scale` | float | auto | Counting scale. If omitted, estimated from the data. |
| `gaussian` | float | `1e-3` | Gaussian read noise variance. |

### `scale`

Scales pattern intensities.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `scale` | float | **required** | Multiplicative factor. |

### `offset`

Offsets pattern intensities.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `offset` | float | **required** | Additive value. |

### `bin`

Bins (downsamples) the patterns by averaging.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `bin` | int | **required** | Binning factor. |

### `apply_mtf`

Applies a detector MTF filter to the patterns (in chunks of 128 patterns).

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `mtf` | [filter hook](#filter-hooks) | **required** | The filter to apply. |
| `domain` | `'real' \| 'recip'` | `'recip'` | Domain in which to apply the filter. |

---

## Initialization hooks

### Probe: `focused`

Initializes a focused (single-mode) probe.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `defocus` | float (Å) | `None` | Positive is overfocus. |
| `conv_angle` | float (mrad) | `None` | Convergence (semiconvergence) angle. |
| `aberrations` | list of [aberrations](#aberrations) | `[]` | CTF aberrations to apply to the probe. |

### Object: `random`

Initializes the object with small random values.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `sigma` | float | `1e-6` | Std. deviation of the random initialization. |

### Scan: `raster`

Defines a raster scan.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `shape` | [int, int] | inferred | Scan shape `(ny, nx)` (total, before any cropping). |
| `step_size` | float or [float, float] (Å) | `None` | Raster step size. |
| `rotation` | float (deg) | `None` | Scan rotation, counter-clockwise. |
| `affine` | 2×2 array | `None` | Scan correction matrix, `[x', y'] = affine @ [x, y]`. |

### Scan: `custom`

Loads user-defined scan positions.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | str | **required** | Path to a `.npy` file containing a `(ny, nx, 2)` array of (y, x) positions. |
| `scale` | float | `None` | Scale factor to multiply the positions by. Use the step size (Å) when positions are given in units of scan steps. |
| `remove_offset` | bool | `True` | Subtract the mean position so the scan is centered on the origin, removing any arbitrary global offset. |

### Tilt: `global`

A constant tilt applied to the whole scan.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `tilt` | [float, float] (mrad) | **required** | Global `[ty, tx]`, broadcast to every probe position. |

### Tilt: `custom`

Loads a per-position tilt map.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | str | **required** | Path to a `.npy` file containing a `(ny, nx, 2)` or `(N, 2)` tilt array (converted to float32). |

---

## Post-init hooks

Hooks in the `post_init` list operate on the initialized reconstruction state.

### `drop_nans`

Masks out probe positions (patterns, scan positions, and tilt) that are mostly
NaN.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `threshold` | float | `0.9` | Fraction of NaN pixels above which a pattern is dropped. |

### `diffraction_align`

Shifts each pattern by the first moment of the mean pattern, so that all
patterns are aligned to the mean. No arguments.

```yaml
post_init:
  - diffraction_align
```

---

## Engine hooks

`engines` is a list of engine hooks; engines run in sequence, each receiving
the state left by the previous one.

```yaml
engines:
  - type: conventional
    noise_model: {type: amplitude}
    solver: {type: lsqml}
    ...
```

### `conventional`

Iterative LSQML / EPie solver (see [Conventional solvers](#conventional-solvers)).

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `noise_model` | [noise model hook](#noise-model-hooks) | **required** | |
| `solver` | [conventional solver hook](#conventional-solvers) | **required** | |
| `position_solver` | [position solver hook](#position-solvers) | `None` | Optionally corrects probe positions. |
| `group_constraints` | list of [group constraints](#group-constraints) | **required** | Constraints applied between groups. May be `[]`. |
| `iter_constraints` | list of [iteration constraints](#iteration-constraints) | **required** | Constraints applied each iteration. May be `[]`. |
| *(all fields below)* | | | See [Common engine fields](#common-engine-fields). |

### `gradient`

Gradient-descent solver with per-variable solvers and cost regularizers.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `noise_model` | [noise model hook](#noise-model-hooks) | **required** | |
| `solvers` | mapping from [recons vars](#reconsvars) to [gradient solver hooks](#gradient-solvers) | **required** | One solver per variable being updated, e.g. `object: {type: adam}`. |
| `regularizers` | list of [cost regularizer hooks](#cost-regularizers) | **required** | Added to the cost. May be `[]`. |
| `group_constraints` | list of [group constraints](#group-constraints) | **required** | |
| `iter_constraints` | list of [iteration constraints](#iteration-constraints) | **required** | |
| *(all fields below)* | | | See [Common engine fields](#common-engine-fields). |

### Common engine fields

Fields shared by both engines:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `sim_shape` | [int, int] | `None` | `[ny, nx]` simulation shape. |
| `resize_method` | `'pad_crop' \| 'resample'` | `'pad_crop'` | How to fit the data to `sim_shape`. |
| `probe_modes` | int | `1` | Number of incoherent probe modes. |
| `base_mode_power` | float | `0.7` | Intensity assigned to the base mode when creating incoherent probe modes. |
| `bwlim_frac` | float | `2/3` | Fraction of the bandwidth limit. |
| `obj_pad_px` | float | `5.0` | Padding around the object, in pixels. |
| `slices` | [slices](#slices) | from plan | Overrides the plan-level `slices`. |
| `niter` | int | `10` | Number of iterations. |
| `grouping` | int | auto | Number of probes per group. |
| `compact` | bool | `False` | |
| `shuffle_groups` | [flag](#flags-and-schedules) | `None` | Whether to shuffle the group order (each iteration). |
| `buffer_n_groups` | int | `2` | Groups of patterns buffered on the device. `0` disables buffering; `~` (None) preloads the entire dataset. |
| `jit_unroll_slices` | bool or int | engine default | JAX backend only. Slices to unroll during JIT compilation (`True`/`0` = all, `False`/`1` = none). Currently `10` for the gradient engine. |
| `update_probe` | [flag](#flags-and-schedules) | `True` | Which iterations to update the probe on. |
| `update_object` | [flag](#flags-and-schedules) | `True` | Which iterations to update the object on. |
| `update_positions` | [flag](#flags-and-schedules) | `False` | Which iterations to update probe positions on. |
| `update_tilt` | [flag](#flags-and-schedules) | `False` | Which iterations to update tilt on. |
| `calc_error` | [flag](#flags-and-schedules) | every iteration | Which iterations to compute the error on. |
| `calc_error_fraction` | float | `0.1` | Fraction of probes used when computing the error. |
| `save` | [flag](#flags-and-schedules) | `False` | Which iterations to save the state on. |
| `save_images` | [flag](#flags-and-schedules) | `False` | Which iterations to save images on. |
| `save_options` | save options | see below | |
| `early_termination` | int | `None` | Terminate after this many iterations without improvement. |
| `early_termination_smoothing` | float | `0.9` | Smoothing factor on the error for early termination. **Low value = more smoothing** (smooths over ~`1/smoothing` iterations). |
| `check_every_group` | bool | `False` | Check early-termination criteria after every group. |
| `send_every_group` | bool | `False` | Send progress updates after every group. |
| `mtf` | [filter hook](#filter-hooks) or MTF plan | `None` | Detector MTF applied to simulated patterns. An MTF plan is a mapping with `filter` (a filter hook) and `domain` (`'real'` or `'recip'`, default `'recip'`). |

**Save options** (`save_options`):

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `images` | list of save types | `['probe', 'object_phase_stack']` | See [save types](#save-types). |
| `crop_roi` | bool | `True` | Crop saved images to the ROI. |
| `unwrap_phase` | bool | `False` | Save unwrapped object phase. |
| `img_dtype` | `'float' \| '8bit' \| '16bit' \| '32bit'` | `'16bit'` | |
| `plot_ext` | str | `'svg'` | Extension for matplotlib figures. |
| `plot_dpi` | int | `300` | |
| `out_dir` | str | `'{name}'` | Output directory (format string). |
| `img_fmt` | str | `'{type}_iter{iter.total_iter:03}.{ext}'` | Filename format for images. |
| `hdf5_fmt` | str | `'iter{iter.total_iter:03}.h5'` | Filename format for HDF5 state files. |

#### Save types

The valid entries for `save_options.images` are: `probe`, `probe_mag`,
`probe_recip`, `probe_recip_mag`, `object_phase_stack`, `object_phase_sum`,
`object_mag_stack`, `object_mag_sum`, `scan`, `tilt`.

---

## Noise model hooks

### `amplitude`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `gaussian_variance` | float | `0.1` | Gaussian noise variance. |
| `eps` | float | `1e-3` | Regularization epsilon. |
| `offset` | float | `0.0` | Intensity offset. |

### `anscombe`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `gaussian_variance` | float | `0.1` | Gaussian noise variance. |
| `eps` | float | `1e-3` | Regularization epsilon. |
| `offset` | float | `0.375` | Anscombe offset. |

### `poisson`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `gaussian_variance` | float | `0.1` | Gaussian read-noise variance. |
| `eps` | float | `1e-3` | Regularization epsilon. |
| `offset` | float | `0.0` | Intensity offset. |

---

## Conventional solvers

### `lsqml`

Least-squares multigrid solver.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `stochastic` | bool | `True` | Use stochastic (per-group) updates. |
| `beta_object` | [schedule](#flags-and-schedules) | `1.0` | Object regularization strength. |
| `beta_probe` | [schedule](#flags-and-schedules) | `1.0` | Probe regularization strength. |
| `illum_reg_object` | [schedule](#flags-and-schedules) | `1e-2` | Object illumination regularization. |
| `illum_reg_probe` | [schedule](#flags-and-schedules) | `1e-2` | Probe illumination regularization. |
| `gamma` | [schedule](#flags-and-schedules) | `1e-4` | Step-size relaxation factor. |

### `epie`

EPie (extended ptychographic iterative engine) solver.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `beta_object` | [schedule](#flags-and-schedules) | `1.0` | Object regularization strength. |
| `beta_probe` | [schedule](#flags-and-schedules) | `1.0` | Probe regularization strength. |

---

## Position solvers

Used by the conventional engine's optional `position_solver` to refine probe
positions.

### `steepest_descent`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `step_size` | float | `1e-2` | Step size as a fraction of the optimal step. |
| `max_step_size` | float (Å) | `None` | Maximum step size. |

### `momentum`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `step_size` | float | `1e-2` | Step size as a fraction of the optimal step. |
| `max_step_size` | float (Å) | `None` | Maximum step size. |
| `momentum` | float | `0.9` | Momentum decay rate. |

---

## Gradient solvers

One entry per variable in the `gradient` engine's `solvers` mapping.

### `sgd`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `learning_rate` | [schedule](#flags-and-schedules) | **required** | |
| `momentum` | [schedule](#flags-and-schedules) | `None` | Momentum coefficient. |
| `nesterov` | bool | `True` | Use Nesterov momentum. |

### `adam`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `learning_rate` | [schedule](#flags-and-schedules) | **required** | |
| `b1` | float | `0.9` | First-moment decay. |
| `b2` | float | `0.999` | Second-moment decay. |
| `eps` | float | `1e-8` | Numerical stability. |
| `eps_root` | float | `0.0` | Square-root epsilon. |
| `nesterov` | bool | `False` | Use Nesterov momentum. |

### `polyak_sgd`

Polyak averaging SGD (anneals the learning rate toward an estimate of the
optimal step size).

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `max_learning_rate` | [schedule](#flags-and-schedules) | **required** | Upper bound on the learning rate. |
| `f_min` | float | **required** | Estimate of the minimum of the cost. |
| `scaling` | [schedule](#flags-and-schedules) | `1.0` | Scaling of the estimated step size. |
| `eps` | float | `0.0` | Numerical stability. |

---

## Iteration constraints

Hooks in an engine's `iter_constraints` list, applied at the end of each
iteration.

| Hook | Arguments | Notes |
| --- | --- | --- |
| `clamp_object_amplitude` | `amplitude`: float or list of float/`~` (default `1.1`) | Clamps the object amplitude magnitude. A list clamps each slice individually. |
| `limit_probe_support` | `max_angle`: float (**required**) | Limits the probe support to this angle. |
| `layers` | `sigma`: float (Å, default `50.0`), `weight`: float (default `0.9`) | Layer (through-focus) regularization. |
| `obj_low_pass` | `max_freq`: float (1/px, default `0.4`) | Low-pass filters the object (Nyquist is 0.5). |
| `obj_gaussian` | `sigma`: float (Å, **required**), `weight`: float (default `0.9`) | Gaussian smoothing of the object. |
| `opr_gaussian` | `sigma`: float (**required**), `weight`: float (default `0.9`) | Gaussian smoothing of the optical potential. (`attr_path` is fixed to `opr.data`.) |
| `tilt_gaussian` | `sigma`: float (**required**), `weight`: float (default `0.9`) | Gaussian smoothing of the tilt field. (`attr_path` is fixed to `scan.tilt`.) |
| `remove_phase_ramp` | free-form dict, passed to the function untyped | Removes the mean phase ramp. |
| `nonneg_object_phase` | `weight`: float (default `1.0`) | Penalizes negative object phase. |

## Group constraints

Hooks in an engine's `group_constraints` list, applied between groups. The
supported hooks are a subset of the iteration constraints:

`clamp_object_amplitude`, `limit_probe_support`, `obj_low_pass`,
`obj_gaussian`, `remove_phase_ramp`, `nonneg_object_phase` (same arguments as
above).

## Cost regularizers

Hooks in the gradient engine's `regularizers` list; each adds a term to the
cost being minimized.

| Hook | Arguments | Notes |
| --- | --- | --- |
| `obj_l1` | `cost`: float (**required**) | L1 norm of the object. |
| `obj_l2` | `cost`: float (**required**) | L2 norm of the object. |
| `obj_phase_l1` | `cost`: float (**required**) | L1 norm of the object phase. |
| `obj_recip_l1` | `cost`: float (**required**) | L1 norm of the reciprocal-space object. |
| `obj_tv` | `cost`: float (**required**), `eps`: float (default `1e-8`) | Total variation of the object. |
| `obj_tikh` (alias `obj_tikhonov`) | `cost`: float (**required**) | Tikhonov regularization of the object. |
| `layers_tv` | `cost`: float (**required**) | Total variation between slices. |
| `layers_tikh` (alias `layers_tikhonov`) | `cost`: float (**required**) | Tikhonov regularization between slices. |
| `probe_phase_tikh` (alias `probe_phase_tikhonov`) | `cost`: float (**required**) | Tikhonov regularization of the probe phase. |
| `probe_recip_tv` | `cost`: float (**required**), `eps`: float (default `1e-8`) | Total variation of the reciprocal-space probe. |
| `probe_recip_tikh` (alias `probe_recip_tikhonov`) | `cost`: float (**required**) | Tikhonov regularization of the reciprocal-space probe. |

---

## Filter hooks

Used in `apply_mtf` (post-load) and `mtf` (engines).

### `gaussian`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `sigma` | float or [float, float] | **required** | Std. deviation of the blur, in detector pixels. |
| `psf_sigma` | float | `3.0` | Std. deviation of the sampling PSF. |

### `ideal_sq`

Ideal square-pixel response.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `psf_radius` | int | `10` | Radius of the PSF kernel. |

### `empad`

The measured EMPAD detector MTF, parameterized by accelerating voltage.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `kv` | float | from data | Accelerating voltage (kV). Supported values: 80, 120, 200, 300. |
| `square` | bool | `True` | Additionally convolve with the square-pixel response. |

The underlying MTF is a Gaussian at 80 and 120 kV, and a Gaussian mixture at
200 and 300 kV. Any other `kv` is an error.

---

## Flags and schedules

Many engine fields accept *flags* (which iterations to perform an action on)
and some accept *schedules* (a value that varies with iteration).

**Flags** can be:

- a plain `true`/`false`,
- a `SimpleFlag` mapping:
  | Argument | Type | Default | Notes |
  | --- | --- | --- | --- |
  | `after` | int | `0` | Only true for iterations greater than this. |
  | `every` | int | `1` | Only true every `n`th iteration (measured from `after`). |
  | `before` | int | `None` | Only true for iterations less than this. |
  
  Example: `update_probe: {after: 5}` — update the probe on iterations 6, 7, 8, …
- or a [schedule hook](#schedule-hooks) of the `expr` type.

**Schedules** can be a plain number, or a [schedule hook](#schedule-hooks).

### Schedule hooks

#### `constant`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `value` | float | **required** | Constant value. |

#### `piecewise`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `init` | [schedule](#flags-and-schedules) | **required** | Value before any threshold. |
| `steps` | mapping from int to [schedule](#flags-and-schedules) | **required** | Iteration thresholds. The value in effect at iteration `i` is the one for the largest threshold `<= i`, or `init` if none match. |

```yaml
beta_object:
  type: piecewise
  init: 1.0
  steps:
    5: 0.5
    10: 0.1
```

#### `expr`

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `expr` | str | **required** | Python expression, evaluated with `i` (current engine iteration), `iter` (iteration state), `state`, `niter`, and `np` (numpy) in scope. |

```yaml
learning_rate:
  type: expr
  expr: "0.01 / (1 + 0.1 * i)"
```

---

## Shared types

### Slices

The `slices` field (top-level or per-engine) describes how to slice the object
through focus. Exactly one of these forms:

```yaml
# explicit thicknesses (Å)
slices:
  thicknesses: [10.0, 10.0, 10.0]

# n slices of a fixed thickness (Å)
slices:
  n: 10
  slice_thickness: 10.0

# n slices filling a total thickness (Å)
slices:
  n: 10
  total_thickness: 100.0
```

### Aberrations

A list of aberrations (e.g. in `focused` probe initialization). Each entry is
either a named aberration or a Krivanek (n, m) coefficient.

Named aberrations — a mapping from a known name to a complex value:

```yaml
aberrations:
  - c1: -5000        # Å
  - c3: 10.0e-6      # Å
```

The value may be a number, or a mapping: `{re: <float>, im: <float>=0}`
(cartesian) or `{mag: <float>, angle: <float>=0}` (polar, degrees).

Known names and their `(n, m)` Krivanek indices: `c1` (1,0), `a1` (1,2),
`b2` (2,1, ×3), `a2` (2,3), `c3` (3,0), `s3` (3,2, ×3), `a3` (3,4), `b4` (4,1,
×4), `d4` (4,3, ×4), `a4` (4,5), `c5` (5,0). (The scale factor is applied
internally; e.g. `C_21 = 3·B2`.)

Krivanek form — a mapping with `n`, `m`, and the coefficient value:

```yaml
aberrations:
  - n: 3
    m: 2
    scale_factor: 1.0
    val: {re: 1e-6, im: 0}   # or {mag: ..., angle: ...}, or a plain number
```

Valid when `n >= 0`, `m >= 0`, `m <= n + 1`, and `m % 2 + n % 2 == 1`.

### ReconsVars

A set of reconstruction variables, used as the keys of the gradient engine's
`solvers` mapping. Valid variables: `object`, `probe`, `positions`, `tilt`,
`positions_affine`, `positions_line`. Given as a list or a comma-separated
string:

```yaml
solvers:
  object: {type: adam, learning_rate: 0.01}
  probe,tilt: {type: sgd, learning_rate: 0.001}
```
