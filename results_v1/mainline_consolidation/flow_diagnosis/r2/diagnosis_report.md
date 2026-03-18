# XTraffic PatchTST Negative-Gain Diagnosis

## Dataset Health

```json
{
  "total_samples": 24,
  "applicable_count": 15,
  "non_applicable_count": 9,
  "applicable_ratio": 0.625,
  "effect_family_distribution": {
    "impulse": 10,
    "level": 5
  },
  "shape_distribution": {
    "hump": 10,
    "step": 5
  },
  "duration_distribution": {
    "short": 15
  },
  "region_length": {
    "mean": 8.133333333333333,
    "min": 6,
    "max": 21
  },
  "direction_consistency_rate": 0.3333333333333333
}
```

## A/B Summary (`tedit_hybrid` vs `profile`)

```json
{
  "tedit_summary": {
    "total": 24,
    "successful": 24,
    "failed": 0,
    "avg_base_mae_vs_revision_target": 53.72947748819987,
    "avg_edited_mae_vs_revision_target": 52.77045174564563,
    "avg_base_mae_vs_future_gt": 103.16665416655871,
    "avg_edited_mae_vs_future_gt": 102.2076284240045,
    "avg_edited_mse_vs_revision_target": 7596.839390795832,
    "avg_edited_smape_vs_revision_target": 0.48791251312043055,
    "avg_future_t_iou": 0.21180555555555555,
    "avg_revision_gain": 0.959025742554239,
    "avg_magnitude_calibration_error": 39.177413734463876,
    "avg_outside_region_preservation": -0.3243886102065195,
    "avg_over_edit_rate": 0.5504240394932111,
    "avg_normalized_parameter_error": 0.3565012604289377,
    "avg_peak_delta_error": 46.352685797380275,
    "avg_signed_area_error": 308.4661711495719,
    "avg_duration_error": 2.4166666666666665,
    "avg_recovery_slope_error": 9.54452414076243,
    "avg_revision_needed_match": 1.0,
    "avg_effect_family_match": 1.0,
    "avg_direction_match": 1.0,
    "avg_shape_match": 1.0,
    "avg_duration_match": 1.0,
    "avg_strength_match": 0.375,
    "avg_intent_match_score": 0.875,
    "applicable_count": 15,
    "applicable_avg_base_mae_vs_revision_target": 85.96716398111978,
    "applicable_avg_edited_mae_vs_revision_target": 84.43272279303301,
    "applicable_avg_base_mae_vs_future_gt": 85.96716398111978,
    "applicable_avg_edited_mae_vs_future_gt": 84.43272279303301,
    "applicable_avg_future_t_iou": 0.3388888888888889,
    "applicable_avg_revision_gain": 1.534441188086782,
    "applicable_avg_magnitude_calibration_error": 62.68386197514222,
    "applicable_avg_outside_region_preservation": -1.1190217763304313,
    "applicable_avg_over_edit_rate": 0.8806784631891377,
    "applicable_avg_normalized_parameter_error": 0.5704020166863003,
    "applicable_avg_peak_delta_error": 74.16429727580845,
    "applicable_avg_signed_area_error": 493.54587383931505,
    "applicable_avg_duration_error": 3.8666666666666667,
    "applicable_avg_recovery_slope_error": 15.271238625219889,
    "applicable_avg_revision_needed_match": 1.0,
    "non_applicable_count": 9,
    "non_applicable_avg_base_mae_vs_revision_target": 0.0,
    "non_applicable_avg_edited_mae_vs_revision_target": 0.0,
    "non_applicable_avg_base_mae_vs_future_gt": 131.83247114229027,
    "non_applicable_avg_edited_mae_vs_future_gt": 131.83247114229027,
    "non_applicable_avg_future_t_iou": 0.0,
    "non_applicable_avg_revision_gain": 0.0,
    "non_applicable_avg_magnitude_calibration_error": 0.0,
    "non_applicable_avg_outside_region_preservation": 1.0,
    "non_applicable_avg_over_edit_rate": 0.0,
    "non_applicable_avg_normalized_parameter_error": 0.0,
    "non_applicable_avg_peak_delta_error": 0.0,
    "non_applicable_avg_signed_area_error": 0.0,
    "non_applicable_avg_duration_error": 0.0,
    "non_applicable_avg_recovery_slope_error": 0.0,
    "non_applicable_avg_revision_needed_match": 1.0
  },
  "profile_summary": {
    "total": 24,
    "successful": 24,
    "failed": 0,
    "avg_base_mae_vs_revision_target": 53.72947748819987,
    "avg_edited_mae_vs_revision_target": 58.54861468200903,
    "avg_base_mae_vs_future_gt": 103.16665416655871,
    "avg_edited_mae_vs_future_gt": 107.9857913603679,
    "avg_edited_mse_vs_revision_target": 9363.376771433219,
    "avg_edited_smape_vs_revision_target": 0.5382539158861211,
    "avg_future_t_iou": 0.21180555555555555,
    "avg_revision_gain": -4.819137193809159,
    "avg_magnitude_calibration_error": 75.62240749955338,
    "avg_outside_region_preservation": -1.816839838966218,
    "avg_over_edit_rate": 0.07236258089171384,
    "avg_normalized_parameter_error": 0.3565012604289377,
    "avg_peak_delta_error": 26.852349022798393,
    "avg_signed_area_error": 685.2229574209122,
    "avg_duration_error": 2.4166666666666665,
    "avg_recovery_slope_error": 15.764073148947555,
    "avg_revision_needed_match": 1.0,
    "avg_effect_family_match": 1.0,
    "avg_direction_match": 1.0,
    "avg_shape_match": 1.0,
    "avg_duration_match": 1.0,
    "avg_strength_match": 0.375,
    "avg_intent_match_score": 0.875,
    "applicable_count": 15,
    "applicable_avg_base_mae_vs_revision_target": 85.96716398111978,
    "applicable_avg_edited_mae_vs_revision_target": 93.67778349121446,
    "applicable_avg_base_mae_vs_future_gt": 85.96716398111978,
    "applicable_avg_edited_mae_vs_future_gt": 93.67778349121446,
    "applicable_avg_future_t_iou": 0.3388888888888889,
    "applicable_avg_revision_gain": -7.710619510094655,
    "applicable_avg_magnitude_calibration_error": 120.9958519992854,
    "applicable_avg_outside_region_preservation": -3.5069437423459475,
    "applicable_avg_over_edit_rate": 0.11578012942674214,
    "applicable_avg_normalized_parameter_error": 0.5704020166863003,
    "applicable_avg_peak_delta_error": 42.96375843647743,
    "applicable_avg_signed_area_error": 1096.3567318734597,
    "applicable_avg_duration_error": 3.8666666666666667,
    "applicable_avg_recovery_slope_error": 25.222517038316088,
    "applicable_avg_revision_needed_match": 1.0,
    "non_applicable_count": 9,
    "non_applicable_avg_base_mae_vs_revision_target": 0.0,
    "non_applicable_avg_edited_mae_vs_revision_target": 0.0,
    "non_applicable_avg_base_mae_vs_future_gt": 131.83247114229027,
    "non_applicable_avg_edited_mae_vs_future_gt": 131.83247114229027,
    "non_applicable_avg_future_t_iou": 0.0,
    "non_applicable_avg_revision_gain": 0.0,
    "non_applicable_avg_magnitude_calibration_error": 0.0,
    "non_applicable_avg_outside_region_preservation": 1.0,
    "non_applicable_avg_over_edit_rate": 0.0,
    "non_applicable_avg_normalized_parameter_error": 0.0,
    "non_applicable_avg_peak_delta_error": 0.0,
    "non_applicable_avg_signed_area_error": 0.0,
    "non_applicable_avg_duration_error": 0.0,
    "non_applicable_avg_recovery_slope_error": 0.0,
    "non_applicable_avg_revision_needed_match": 1.0
  },
  "tedit_minus_profile": {
    "avg_revision_gain": 5.778162936363398,
    "avg_base_mae_vs_revision_target": 0.0,
    "avg_edited_mae_vs_revision_target": -5.778162936363401,
    "avg_base_mae_vs_future_gt": 0.0,
    "avg_edited_mae_vs_future_gt": -5.778162936363401,
    "avg_outside_region_preservation": 1.4924512287596985,
    "avg_over_edit_rate": 0.47806145860149724
  }
}
```

## TEdit Result Breakdown

```json
{
  "by_shape": {
    "none": {
      "revision_gain": 0.0,
      "base_mae_vs_revision_target": 0.0,
      "edited_mae_vs_revision_target": 0.0,
      "base_mae_vs_future_gt": 131.83247114229027,
      "edited_mae_vs_future_gt": 131.83247114229027,
      "outside_region_preservation": 1.0,
      "over_edit_rate": 0.0
    },
    "hump": {
      "revision_gain": 2.3961948843692977,
      "base_mae_vs_revision_target": 101.29981847829433,
      "edited_mae_vs_revision_target": 98.90362359392503,
      "base_mae_vs_future_gt": 101.29981847829433,
      "edited_mae_vs_future_gt": 98.90362359392503,
      "outside_region_preservation": -2.1556733168111224,
      "over_edit_rate": 0.8930503457421615
    },
    "step": {
      "revision_gain": -0.18906620447824807,
      "base_mae_vs_revision_target": 55.30185498677074,
      "edited_mae_vs_revision_target": 55.49092119124898,
      "base_mae_vs_future_gt": 55.30185498677074,
      "edited_mae_vs_future_gt": 55.49092119124898,
      "outside_region_preservation": 0.9542813046309522,
      "over_edit_rate": 0.8559346980830901
    }
  },
  "by_duration": {
    "none": {
      "revision_gain": 0.0,
      "base_mae_vs_revision_target": 0.0,
      "edited_mae_vs_revision_target": 0.0,
      "base_mae_vs_future_gt": 131.83247114229027,
      "edited_mae_vs_future_gt": 131.83247114229027,
      "outside_region_preservation": 1.0,
      "over_edit_rate": 0.0
    },
    "short": {
      "revision_gain": 1.534441188086782,
      "base_mae_vs_revision_target": 85.96716398111978,
      "edited_mae_vs_revision_target": 84.43272279303301,
      "base_mae_vs_future_gt": 85.96716398111978,
      "edited_mae_vs_future_gt": 84.43272279303301,
      "outside_region_preservation": -1.1190217763304313,
      "over_edit_rate": 0.8806784631891377
    }
  },
  "executor_checks": {
    "editor_region_within_bounds_rate": 1.0,
    "avg_editor_pred_iou": 1.0
  },
  "worst_samples_topk": [
    {
      "sample_id": "028",
      "revision_gain": -0.6486118354640666,
      "over_edit_rate": 0.8861788617886179,
      "outside_region_preservation": 0.908751274468189,
      "tool_name": "hybrid_down",
      "pred_region": [
        0,
        24
      ],
      "gt_region": [
        0,
        21
      ],
      "editor_region": [
        64,
        88
      ],
      "future_offset_resampled": 64,
      "intent_match_score": 0.8
    },
    {
      "sample_id": "009",
      "revision_gain": -0.2792327865501534,
      "over_edit_rate": 0.8582677165354331,
      "outside_region_preservation": 0.8907945077038363,
      "tool_name": "hybrid_down",
      "pred_region": [
        0,
        24
      ],
      "gt_region": [
        0,
        17
      ],
      "editor_region": [
        64,
        88
      ],
      "future_offset_resampled": 64,
      "intent_match_score": 0.8
    },
    {
      "sample_id": "046",
      "revision_gain": -0.015722109786565852,
      "over_edit_rate": 0.855072463768116,
      "outside_region_preservation": 0.9861193213417256,
      "tool_name": "hybrid_down",
      "pred_region": [
        0,
        24
      ],
      "gt_region": [
        0,
        6
      ],
      "editor_region": [
        64,
        88
      ],
      "future_offset_resampled": 64,
      "intent_match_score": 0.8
    },
    {
      "sample_id": "034",
      "revision_gain": -0.008794133673966087,
      "over_edit_rate": 0.8188405797101449,
      "outside_region_preservation": 0.99284876974264,
      "tool_name": "hybrid_down",
      "pred_region": [
        0,
        24
      ],
      "gt_region": [
        0,
        6
      ],
      "editor_region": [
        64,
        88
      ],
      "future_offset_resampled": 64,
      "intent_match_score": 0.8
    },
    {
      "sample_id": "NA_010",
      "revision_gain": 0.0,
      "over_edit_rate": 0.0,
      "outside_region_preservation": 1.0,
      "tool_name": "none",
      "pred_region": [
        0,
        0
      ],
      "gt_region": [
        0,
        0
      ],
      "editor_region": null,
      "future_offset_resampled": null,
      "intent_match_score": 1.0
    },
    {
      "sample_id": "NA_011",
      "revision_gain": 0.0,
      "over_edit_rate": 0.0,
      "outside_region_preservation": 1.0,
      "tool_name": "none",
      "pred_region": [
        0,
        0
      ],
      "gt_region": [
        0,
        0
      ],
      "editor_region": null,
      "future_offset_resampled": null,
      "intent_match_score": 1.0
    },
    {
      "sample_id": "NA_012",
      "revision_gain": 0.0,
      "over_edit_rate": 0.0,
      "outside_region_preservation": 1.0,
      "tool_name": "none",
      "pred_region": [
        0,
        0
      ],
      "gt_region": [
        0,
        0
      ],
      "editor_region": null,
      "future_offset_resampled": null,
      "intent_match_score": 1.0
    },
    {
      "sample_id": "NA_001",
      "revision_gain": 0.0,
      "over_edit_rate": 0.0,
      "outside_region_preservation": 1.0,
      "tool_name": "none",
      "pred_region": [
        0,
        0
      ],
      "gt_region": [
        0,
        0
      ],
      "editor_region": null,
      "future_offset_resampled": null,
      "intent_match_score": 1.0
    },
    {
      "sample_id": "NA_002",
      "revision_gain": 0.0,
      "over_edit_rate": 0.0,
      "outside_region_preservation": 1.0,
      "tool_name": "none",
      "pred_region": [
        0,
        0
      ],
      "gt_region": [
        0,
        0
      ],
      "editor_region": null,
      "future_offset_resampled": null,
      "intent_match_score": 1.0
    },
    {
      "sample_id": "NA_003",
      "revision_gain": 0.0,
      "over_edit_rate": 0.0,
      "outside_region_preservation": 1.0,
      "tool_name": "none",
      "pred_region": [
        0,
        0
      ],
      "gt_region": [
        0,
        0
      ],
      "editor_region": null,
      "future_offset_resampled": null,
      "intent_match_score": 1.0
    }
  ]
}
```

