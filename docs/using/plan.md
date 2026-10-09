# Reconstruction plan files

A 'plan' is a file specifying a series of reconstruction steps ('engines') to take. Plan files are specified in the [YAML](https://yaml.org) data specification language, to allow files to be easily read by both humans and machines.

## Plan options

A reconstruction plan has the following top-level keys:

```yaml
# name of reconstruction
name: my_recons
# (optional) computational backend to use, 'cupy', 'jax', 'numpy', or None (default)
backend: 'cupy'
# (optional) datatype to perform reconstructions with 'float32' (default) or 'float64'
dtype: float32

# Hook to load raw data. See section below
raw_data: ~

# (optional) Sequence of hooks to run after loading data. See section below
post_load: []

# wavelength of radiation, in angstroms. Inferred from raw data if not specified
wavelength: 0.0251

# How to initialize probe, object, and probe positions
init:
  probe: ~
  object: ~
  scan: ~

# (optional) For multislice reconstructions, specify initial slices to use
slices:
  n: 20
  total_thickness: 200  # can also specify `slice_thickness` instead

# (optional) Sequence of hooks to run after initialization but before reconstructions
post_init: []

# list of engines to run in sequence
engines: []

```

## Engine plan options

Each engine supports the following keys:

```yaml
# number of iterations to run
niter: 100
# (optional) number of probe positions to simulate simultaneously
grouping: 64
# (optional) whether to group probe positions
# compactly (as in LSQ-MLc) or sparsely (the default)
compact: False
# (optional) flag indicating whether to shuffle groups at a given iteration.
# Defaults to `True` if using sparse positions, `False` otherwise.
shuffle_groups: True

# (optional) size of simulation [n_y, n_x]
sim_shape: [256, 256]
# (optional) how to resize probe, object, and patterns to `sim_shape`.
# 'pad_crop' (default) or 'resample'. `pad_crop` means patterns are
# padded or cropped (and therefore real-space is resampled).
resize_method: 'pad_crop'

# (optional) Number of incoherent probe modes to simulate.
probe_modes: 4

# (optional) flags indicating whether to update the object, probe,
# or probe positions at a given iteration
update_object: True
update_probe: {after: 10}
update_positions: {before: 80}

# noise model to use for reconstruction.
noise_model: 'poisson'

# (optional) flag indicating whether to calculate detector error
# at a given iteration
calc_error: {every: 5}
# (optional) fraction of groups to calculate error at
calc_error_fraction: 0.1

# (optional) flag indicating whether to save output
# at a given iteration
save: {every: 10}
save_images: {every: 10}

# (optional) options for output
save_options:
  # (optional) list of image types to save. Supports 'probe[_mag]',
  # 'probe_recip[_mag]', 'object_(phase|mag)_(stack|sum)'
  images: ['probe', 'probe_mag', 'object_phase_stack', 'object_mag_stack']
  # (optional) whether to crop images to the scan bounding box
  crop_roi: True
  # (optional) whether to phase unwrap phase images
  unwrap_phase: True
  # (optional) Datatype to store images at. Floating point images
  # are stored unscaled, other images are scaled to saturation.
  # Either a single dtype applied to all images, or a per-image map
  # of image type to dtype (types not listed default to 16bit)
  img_dtype: 16bit  # float, 8bit, 16bit, or 32bit
  #img_dtype:
  #  probe: 8bit
  #  object_phase_stack: 32bit

  # (optional) Python format strings controlling the name of
  # HDF5 and image outputs.
  # 'name' is resolved to the name of the reconstruction, 'type'
  # is the type of image, and 'iter' is an IterState object.
  out_dir: "{name}"
  img_fmt: "{type}_iter{iter.total_iter}.tiff"
  hdf5_fmt: "iter{iter.total_iter}.h5"
```

## Hooks

A common theme throughout `phaser` is the use of 'hooks'. Hooks are modular
components, customizable by the end user. For instance, `post_load` is passed a list
of hooks, each of which can arbitrarily modify the raw data. Each hook type has
a defined function signature, which all instances of that hook should obey. Hooks
can also take properties, which are passed to the hook function for configuration.

Here are some example hook instantiations:
```yaml
post_load:
   # calls the built-in hook 'poisson'
 - 'poisson'      
   # same as previous, but can specify properties as well
 - type: poisson   
   scale: 1.0e+7
   # call a user-defined hook 'function' from package 'user_package.subpkg'
 - user_package.subpkg:function
   # same as previous, but pass properties as well
 - type: user_package.subpkg:function
   myprop: value
```

Properties are type-checked for built-in hooks, but not user defined hooks.
In the case of `post_load`, this is signature of each hook function:

```python
def hook(args: RawData, props: t.Dict[str, t.Any]) -> RawData:
    ...
```

## Known hooks

Post-load hooks have an opportunity to modify the loaded raw data prior
to state initialization.
```yaml
post_load:
  # crop the raw data (in real space), (y_i, y_f, x_i, x_f). Python-like slicing
  - type: crop_data
    crop: [0, 50, 0, -10]
  # scale the raw data by `scale`, then apply Poisson noise
  - type: poisson
    scale: 1.0e+7
  # just scale the raw data
  - type: scale
    scale: 1.0e+4
```

Post-init hooks modify data after initialization but before reconstruction.
They can modify the raw patterns, the state, or both.

```yaml
post_init:
  # drop patterns which are mostly NaNs. Flattens scan
  - drop_nans
  # diffraction align patterns (applies bilinear resampling)
  - diffraction_align
```

Noise models are hooks as well. When called, they return an instance of `NoiseModel`.
Noise models are discussed in more detail in the engines section.

```yaml
noise_model:
  - type: amplitude
    eps: 1.0e-3
  - anscombe
  - poisson
```

Flags are hooks which return a boolean value each iteration:

```yaml
engines:
 - niter: 100
   # return True after 10 iterations, False before
   update_probe: {after: 10}
   # return True every 5 iterations
   update_positions: {every: 5}
   # return True every even iteration between 3 and 79
   update_object: {after: 2, before: 80, every: 2}
   # or call a user-defined hook
   save_images: 'mypkg:user_defined_hook'
```

Solvers & regularizers are hooks as well. These are discussed in more detail
in the engines section.