| configuration | task | accuracy (%) | macro-F1 | macro-F1 range |
|---|---|---|---|---|
| thesis | all7 | 47.7 +- 6.7 | 0.488 +- 0.087 | 0.21-0.69 |
| thesis_original | all7 | 65.8 +- 3.5 | 0.307 +- 0.089 | 0.12-0.47 |
| thesis_balanced | all7 | 44.5 +- 6.3 | 0.477 +- 0.084 | 0.21-0.65 |
| thesis_weights | all7 | 56.8 +- 6.3 | 0.496 +- 0.091 | 0.25-0.68 |
| hudgins | all7 | 55.9 +- 6.1 | 0.497 +- 0.093 | 0.21-0.66 |
| ls4 | all7 | 56.0 +- 5.9 | 0.491 +- 0.092 | 0.22-0.66 |
| ls9 | all7 | 55.7 +- 5.7 | 0.486 +- 0.091 | 0.22-0.66 |
| thesis_calib | all7 | 63.4 +- 6.8 | 0.533 +- 0.100 | 0.22-0.67 |
| ls9_calib | all7 | 62.7 +- 5.9 | 0.541 +- 0.103 | 0.22-0.69 |
| ls9_calib_w250 | all7 | 62.7 +- 6.0 | 0.541 +- 0.108 | 0.23-0.71 |
| ls9_calib_et | all7 | 67.6 +- 3.9 | 0.351 +- 0.103 | 0.13-0.53 |
| ls9_calib_lda | all7 | 65.0 +- 5.2 | 0.513 +- 0.108 | 0.19-0.71 |
| ls9_calib_lgbm | all7 | 65.6 +- 5.0 | 0.529 +- 0.110 | 0.22-0.73 |
| ls9_calib_xgb | all7 | 65.9 +- 4.8 | 0.533 +- 0.113 | 0.21-0.73 |
| cnn | all7 | 58.0 +- 7.3 | 0.503 +- 0.104 | 0.20-0.66 |
| cnn_calib | all7 | 62.1 +- 7.4 | 0.530 +- 0.106 | 0.19-0.68 |
| ls9_calib_smooth3 | all7 | 64.5 +- 6.1 | 0.554 +- 0.106 | 0.22-0.70 |
| ls9_calib_smooth5 | all7 | 65.3 +- 6.1 | 0.556 +- 0.111 | 0.19-0.70 |
| idle_thesis_weights | idle6 | 68.4 +- 5.3 | 0.551 +- 0.111 | 0.22-0.78 |
| idle_ls9_calib | idle6 | 71.5 +- 5.4 | 0.611 +- 0.111 | 0.28-0.77 |
| idle_ls9_calib_lda | idle6 | 72.4 +- 4.9 | 0.574 +- 0.122 | 0.22-0.79 |
| idle_ls9_calib_smooth5 | idle6 | 73.9 +- 4.8 | 0.633 +- 0.122 | 0.25-0.79 |
| g6_thesis_weights | gestures6 | 82.6 +- 12.9 | 0.816 +- 0.138 | 0.37-1.00 |
| g6_hudgins | gestures6 | 83.0 +- 13.2 | 0.819 +- 0.142 | 0.37-0.99 |
| g6_ls4 | gestures6 | 82.6 +- 13.2 | 0.814 +- 0.143 | 0.35-0.99 |
| g6_ls9 | gestures6 | 81.9 +- 13.0 | 0.808 +- 0.142 | 0.38-0.99 |
| g6_thesis_calib | gestures6 | 87.7 +- 12.8 | 0.873 +- 0.131 | 0.38-0.99 |
| g6_ls9_calib | gestures6 | 87.4 +- 12.5 | 0.870 +- 0.129 | 0.39-1.00 |
| g6_ls9_calib_et | gestures6 | 88.1 +- 12.9 | 0.877 +- 0.133 | 0.37-1.00 |
| g6_ls9_calib_lda | gestures6 | 86.4 +- 12.4 | 0.859 +- 0.128 | 0.40-0.99 |
| g6_ls9_calib_lgbm | gestures6 | 87.8 +- 13.1 | 0.875 +- 0.135 | 0.37-1.00 |
| g6_ls9_calib_xgb | gestures6 | 87.5 +- 13.3 | 0.872 +- 0.136 | 0.34-1.00 |
| g6_cnn | gestures6 | 80.7 +- 14.6 | 0.794 +- 0.161 | 0.26-0.97 |
| g6_cnn_calib | gestures6 | 84.4 +- 13.8 | 0.835 +- 0.144 | 0.35-0.99 |

Paired Wilcoxon signed-rank tests on per-subject macro-F1 (36 subjects):

| A | B | mean F1(B) - F1(A) | subjects B > A | p |
|---|---|---|---|---|
| thesis | thesis_weights | +0.008 | 19/36 | 0.13 |
| thesis_weights | hudgins | +0.000 | 19/36 | 0.93 |
| thesis_weights | ls9 | -0.010 | 7/36 | 0.00055 |
| thesis_weights | thesis_calib | +0.036 | 24/36 | 0.0011 |
| ls9 | ls9_calib | +0.054 | 31/36 | 4.9e-06 |
| thesis_calib | ls9_calib | +0.008 | 25/36 | 0.015 |
| ls9_calib | ls9_calib_w250 | +0.000 | 21/36 | 0.79 |
| ls9_calib | ls9_calib_lda | -0.028 | 11/36 | 0.0018 |
| ls9_calib | ls9_calib_lgbm | -0.011 | 12/36 | 0.026 |
| ls9_calib | ls9_calib_xgb | -0.008 | 13/36 | 0.14 |
| cnn | cnn_calib | +0.028 | 27/36 | 0.00033 |
| ls9_calib | cnn_calib | -0.010 | 15/36 | 0.15 |
| ls9_calib | ls9_calib_smooth5 | +0.016 | 27/36 | 0.00099 |
| idle_thesis_weights | idle_ls9_calib | +0.060 | 30/36 | 8.1e-06 |
| g6_thesis_weights | g6_thesis_calib | +0.058 | 26/36 | 0.00014 |
| g6_ls9 | g6_ls9_calib | +0.062 | 26/36 | 3.3e-05 |
| g6_thesis_calib | g6_ls9_calib | -0.004 | 14/36 | 0.19 |
| g6_ls9_calib | g6_ls9_calib_et | +0.007 | 26/36 | 0.0088 |
| g6_cnn | g6_cnn_calib | +0.042 | 26/36 | 0.002 |
| g6_ls9_calib | g6_cnn_calib | -0.034 | 7/36 | 7.2e-05 |
