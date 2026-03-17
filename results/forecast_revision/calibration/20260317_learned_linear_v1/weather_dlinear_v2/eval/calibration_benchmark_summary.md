# Forecast Revision Calibration Benchmark

| Method | NPE | Peak Delta | Signed Area | Duration | Recovery Slope | Revision Gain | Edited MAE | Outside Preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_direct_numeric | 0.2932 | 0.9792 | 2.1661 | 1.0000 | 0.8303 | 0.1137 | 0.1324 | 1.0000 |
| discrete_strength_table | 0.2910 | 0.9798 | 2.1678 | 1.0000 | 0.8308 | 0.1137 | 0.1324 | 1.0000 |
| rule_local_stats | 0.3082 | 0.9960 | 2.3344 | 1.0000 | 0.8221 | 0.1070 | 0.1391 | 1.0000 |
| learned_linear | 0.0487 | 0.9393 | 2.3803 | 0.0000 | 0.8075 | 0.0980 | 0.1482 | 1.0000 |
| oracle_calibration | 0.0000 | 0.3260 | 0.0269 | 0.0000 | 0.8847 | 0.1826 | 0.0636 | 1.0000 |
