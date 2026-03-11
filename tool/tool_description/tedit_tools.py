"""Auto-generated tool descriptions for TEdit-based editing functions.

This module exposes a list named ``description`` containing metadata for each
TEdit editing tool. These descriptions are intended to guide a language model
agent when invoking TEdit-based editing tools for time series modification.
"""

description = [
    {
        "description": "Edit time series using TEdit diffusion model based on attribute conditions. This tool uses a pre-trained diffusion model to transform time series by changing their attributes (e.g., trend type, seasonality, volatility). It's particularly useful for semantic-level editing where you want to change the overall characteristics of a time series.",
        "name": "tedit_edit",
        "optional_parameters": [
            {
                "name": "n_samples",
                "type": "int",
                "default": 1,
                "description": "Number of edited samples to generate. Higher values provide more options but take longer."
            },
            {
                "name": "sampler",
                "type": "str",
                "default": "ddim",
                "description": "Sampling method: 'ddim' for faster deterministic sampling, 'ddpm' for standard diffusion sampling."
            },
            {
                "name": "edit_steps",
                "type": "int",
                "default": 50,
                "description": "Number of diffusion steps for editing. More steps may improve quality but are slower."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series to edit."
            },
            {
                "name": "src_attrs",
                "type": "array_like",
                "default": None,
                "description": "Source attribute indices describing the current time series characteristics (e.g., [trend_type, seasonality_type])."
            },
            {
                "name": "tgt_attrs",
                "type": "array_like",
                "default": None,
                "description": "Target attribute indices describing the desired time series characteristics."
            }
        ],
        "tag": ["editing", "diffusion", "semantic", "tedit"]
    },
    {
        "description": "Edit a specific region of time series using TEdit diffusion model. This tool applies TEdit's semantic editing capabilities to a localized region, allowing you to change characteristics of a specific segment while preserving the rest of the time series.",
        "name": "tedit_edit_region",
        "optional_parameters": [
            {
                "name": "n_samples",
                "type": "int",
                "default": 1,
                "description": "Number of edited samples to generate."
            },
            {
                "name": "sampler",
                "type": "str",
                "default": "ddim",
                "description": "Sampling method: 'ddim' or 'ddpm'."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of the region to edit (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of the region to edit (exclusive, 0-based)."
            },
            {
                "name": "src_attrs",
                "type": "array_like",
                "default": None,
                "description": "Source attribute indices for the region."
            },
            {
                "name": "tgt_attrs",
                "type": "array_like",
                "default": None,
                "description": "Target attribute indices for the region."
            }
        ],
        "tag": ["editing", "diffusion", "semantic", "region", "tedit"]
    },
    {
        "description": "Change the trend type of a time series using TEdit. This tool transforms the trend characteristics (e.g., from linear to exponential, or change the slope direction) while preserving other aspects like seasonality and noise patterns.",
        "name": "tedit_change_trend",
        "optional_parameters": [
            {
                "name": "n_samples",
                "type": "int",
                "default": 1,
                "description": "Number of samples to generate."
            },
            {
                "name": "sampler",
                "type": "str",
                "default": "ddim",
                "description": "Sampling method."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "trend_type_idx",
                "type": "int",
                "default": None,
                "description": "Target trend type index (depends on model's attribute encoding)."
            }
        ],
        "tag": ["editing", "diffusion", "trend", "semantic", "tedit"]
    },
    {
        "description": "Change the seasonality pattern of a time series using TEdit. This tool modifies the seasonal component (e.g., change period, amplitude, or waveform type) while maintaining the trend and other characteristics.",
        "name": "tedit_change_seasonality",
        "optional_parameters": [
            {
                "name": "n_samples",
                "type": "int",
                "default": 1,
                "description": "Number of samples to generate."
            },
            {
                "name": "sampler",
                "type": "str",
                "default": "ddim",
                "description": "Sampling method."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "seasonality_type_idx",
                "type": "int",
                "default": None,
                "description": "Target seasonality type index."
            }
        ],
        "tag": ["editing", "diffusion", "seasonality", "semantic", "tedit"]
    },
    {
        "description": "Change the volatility characteristics of a time series using TEdit. This tool adjusts the noise/fluctuation patterns (e.g., from low to high volatility) while preserving the underlying trend and seasonality.",
        "name": "tedit_change_volatility",
        "optional_parameters": [
            {
                "name": "n_samples",
                "type": "int",
                "default": 1,
                "description": "Number of samples to generate."
            },
            {
                "name": "sampler",
                "type": "str",
                "default": "ddim",
                "description": "Sampling method."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "volatility_type_idx",
                "type": "int",
                "default": None,
                "description": "Target volatility type index."
            }
        ],
        "tag": ["editing", "diffusion", "volatility", "semantic", "tedit"]
    },
    {
        "description": "Select a region of the time series for editing based on semantic or statistical criteria. This is the first step in the two-stage editing process (region selection -> specific editing). It identifies the most appropriate region to apply edits based on the user's intent.",
        "name": "select_editing_region",
        "optional_parameters": [
            {
                "name": "method",
                "type": "str",
                "default": "semantic",
                "description": "Selection method: 'semantic' for intent-based, 'statistical' for anomaly-based, 'manual' for user-specified."
            },
            {
                "name": "threshold",
                "type": "float",
                "default": 0.5,
                "description": "Threshold for statistical methods (e.g., anomaly detection threshold)."
            },
            {
                "name": "min_length",
                "type": "int",
                "default": 5,
                "description": "Minimum region length to select."
            },
            {
                "name": "max_length",
                "type": "int",
                "default": 50,
                "description": "Maximum region length to select."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "intent",
                "type": "str",
                "default": None,
                "description": "User's editing intent (e.g., 'increase trend', 'reduce noise', 'smooth outliers')."
            }
        ],
        "tag": ["editing", "region_selection", "preprocessing", "tedit"]
    },
    {
        "description": "Apply a two-stage editing process: first select a region, then apply TEdit editing to that region. This combines region selection with semantic editing for targeted modifications.",
        "name": "tedit_two_stage_edit",
        "optional_parameters": [
            {
                "name": "region_method",
                "type": "str",
                "default": "semantic",
                "description": "Region selection method."
            },
            {
                "name": "n_samples",
                "type": "int",
                "default": 1,
                "description": "Number of samples to generate."
            },
            {
                "name": "sampler",
                "type": "str",
                "default": "ddim",
                "description": "Sampling method."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "intent",
                "type": "str",
                "default": None,
                "description": "User's editing intent for region selection."
            },
            {
                "name": "src_attrs",
                "type": "array_like",
                "default": None,
                "description": "Source attributes for TEdit."
            },
            {
                "name": "tgt_attrs",
                "type": "array_like",
                "default": None,
                "description": "Target attributes for TEdit."
            }
        ],
        "tag": ["editing", "diffusion", "two_stage", "semantic", "tedit"]
    }
]
