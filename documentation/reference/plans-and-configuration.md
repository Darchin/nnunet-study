# Plans and Configuration Reference

This page is the concise reference for `nnUNetPlans.json` and configuration-level customization.

## What a plans file does

The plans file defines how nnU-Net preprocesses the dataset and how each configuration is trained.

It contains:

- global dataset-level settings
- one or more named configurations under `configurations`

## Important global fields

- `image_reader_writer`: reader/writer class used for this dataset
- `label_manager`: label-handling class
- `transpose_forward` and `transpose_backward`: axis reordering used internally
- `dataset_name`: dataset identity metadata

## Important configuration fields

- `spacing`: target spacing
- `patch_size`: training patch size
- `batch_size`: batch size
- `data_identifier`: name of the corresponding preprocessed data folder
- `preprocessor_name`: preprocessor class
- `normalization_schemes`: normalization mapping per channel
- `network_arch_class_name`: architecture class
- `conv_kernel_sizes`: encoder kernel layout
- `n_conv_per_stage_encoder`
- `n_conv_per_stage_decoder`
- `num_pool_per_axis`
- `pool_op_kernel_sizes`

## Special configuration relationships

- `inherits_from`: reuse one or more configurations and override selected fields. A string selects one parent. A list
  composes fully resolved parents from left to right, so later parents override earlier parents and the child overrides
  all of them.
- `previous_stage`: previous stage of a cascade
- `next_stage`: next stage of a cascade

## When you must rerun preprocessing

You generally need to rerun preprocessing if you change anything that affects the prepared data, for example:

- `spacing`
- `preprocessor_name`
- `normalization_schemes`
- resampling functions
- any change that requires a new `data_identifier`

You generally do not need to rerun preprocessing for training-only changes such as:

- `batch_size`
- some architecture-only settings that still reuse the same prepared data

## Common customization patterns

### Increase batch size

```json
"3d_fullres_bs40": {
  "inherits_from": "3d_fullres",
  "batch_size": 40
}
```

### Compose an architecture configuration and a training recipe

```json
"mn-2x-t-recipe": {
  "inherits_from": ["mn-2x", "mobile-training-recipe"],
  "patch_size_multiplier": 6,
  "architecture": {
    "arch_kwargs": {
      "channels": [16, 32, 64, 128, 256]
    }
  }
}
```

Each parent is resolved before it is merged. This is ordered composition rather than an implicit inheritance order:
parents later in the list take precedence over earlier parents, and fields in the child take precedence over every
parent. Nested dictionaries follow the same configuration merge behavior as single-parent inheritance.

### Add a custom preprocessor

```json
"3d_fullres_custom_preproc": {
  "inherits_from": "3d_fullres",
  "preprocessor_name": "MyPreprocessor",
  "data_identifier": "3d_fullres_custom_preproc"
}
```

### Change target spacing

```json
"3d_fullres_custom_spacing": {
  "inherits_from": "3d_fullres",
  "spacing": [1.0, 1.0, 2.5],
  "data_identifier": "3d_fullres_custom_spacing"
}
```

## Related pages

- [Train models](../how-to/train-models.md)
- [Intensity normalization in nnU-Net](../explanation_normalization.md)
- [Extending nnU-Net](../extending_nnunet.md)

## Detailed legacy page

- [Modifying the nnU-Net configurations](../explanation_plans_files.md)
