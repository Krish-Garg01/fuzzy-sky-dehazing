# sots_outdoor Ablation Study Summary

## All Images (N = 500)

| Ablation | Method | Description | PSNR (dB) | dPSNR vs F0 | dPSNR vs F1 | SSIM | Sky PSNR | Sky dE00 | NonSky PSNR | SkySat% | Seam L1 | Seam Grad |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1_B0_plain_dcp | `B0` | Plain DCP (no sky correction) | 17.05 | -5.47 | -6.54 | 0.8574 | 15.30 | 14.83 | 20.68 | 4.12% | 0.1459 | 0.1140 |
| A2a_B1_hard_thresh | `B1` | Hard threshold on (f1, f2), tuned tau=0.65 | 23.45 | +0.94 | -0.13 | 0.9285 | 29.74 | 2.82 | 22.88 | 0.47% | 0.0440 | 0.0541 |
| A2b_B2_hard_fcm | `B2` | Hard FCM mask M = 1[mu_raw > 0.5] | 22.32 | -0.20 | -1.27 | 0.9172 | 28.55 | 3.35 | 22.02 | 0.77% | 0.0538 | 0.0663 |
| Ref_F0_soft_fuzzy | `F0` | Soft fuzzy transmission blend (hand-set defaults) | 22.52 | +0.00 | -1.07 | 0.9208 | 28.62 | 3.32 | 22.23 | 0.76% | 0.0525 | 0.0602 |
| Ref_F1_fuzzy_ga | `F1` | Soft fuzzy transmission blend + GA-tuned params | 23.59 | +1.07 | +0.00 | 0.9292 | 29.35 | 3.01 | 23.07 | 0.57% | 0.0464 | 0.0609 |
| A3_no_f4 | `A3_no_f4` | Drop spatial vertical prior f4 (use f1..f3 only) | 22.69 | +0.17 | -0.90 | 0.9215 | 28.25 | 3.62 | 22.53 | 0.74% | 0.0535 | 0.0614 |
| A4_no_f3 | `A4_no_f3` | Drop gradient feature f3 (use f1, f2, f4 only) | 22.03 | -0.49 | -1.56 | 0.9157 | 27.77 | 3.94 | 22.21 | 0.75% | 0.0531 | 0.0618 |
| A5a_with_f5 | `A5a_with_f5` | Add local intensity variance f5 (5 features) | 22.46 | -0.06 | -1.13 | 0.9196 | 28.61 | 3.37 | 22.19 | 0.75% | 0.0538 | 0.0602 |
| A5b_with_f6 | `A5b_with_f6` | Add normalized dark channel f6 (5 features) | 22.45 | -0.06 | -1.13 | 0.9211 | 28.35 | 3.39 | 22.14 | 0.81% | 0.0566 | 0.0620 |
| A6a_c2 | `A6a_c2` | FCM cluster count c = 2 | 23.39 | +0.87 | -0.20 | 0.9274 | 29.51 | 2.95 | 22.74 | 0.67% | 0.0468 | 0.0595 |
| A6b_c4 | `A6b_c4` | FCM cluster count c = 4 | 21.42 | -1.10 | -2.17 | 0.9123 | 26.76 | 4.26 | 21.82 | 0.95% | 0.0637 | 0.0657 |
| A7_tolerance | `A7_tolerance` | Additive tolerance lift vs multiplicative blend | 20.76 | -1.75 | -2.82 | 0.9018 | 25.30 | 5.13 | 21.42 | 0.72% | 0.0691 | 0.0862 |
| A8_no_mu_gf | `A8_no_mu_gf` | Skip guided filter on membership mu_raw | 22.45 | -0.06 | -1.13 | 0.9194 | 28.62 | 3.33 | 22.16 | 0.76% | 0.0528 | 0.0631 |
| A9a_m1.5 | `A9a_m1.5` | FCM fuzziness exponent m = 1.5 | 22.73 | +0.21 | -0.86 | 0.9216 | 29.23 | 3.07 | 22.27 | 0.63% | 0.0500 | 0.0606 |
| A9b_m3.0 | `A9b_m3.0` | FCM fuzziness exponent m = 3.0 | 22.29 | -0.23 | -1.30 | 0.9208 | 27.46 | 3.68 | 22.13 | 0.86% | 0.0609 | 0.0619 |
| A10_gf_before_blend | `A10_gf_before_blend` | Guided filter before blend instead of after | 21.97 | -0.55 | -1.62 | 0.9140 | 27.83 | 3.49 | 21.67 | 0.72% | 0.0655 | 0.0749 |

## Sky-Heavy Images (N = 324)

| Ablation | Method | Description | PSNR (dB) | dPSNR vs F0 | dPSNR vs F1 | SSIM | Sky PSNR | Sky dE00 | NonSky PSNR | SkySat% | Seam L1 | Seam Grad |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1_B0_plain_dcp | `B0` | Plain DCP (no sky correction) | 15.97 | -7.34 | -8.60 | 0.8410 | 13.10 | 18.22 | 21.27 | 4.55% | 0.1584 | 0.1124 |
| A2a_B1_hard_thresh | `B1` | Hard threshold on (f1, f2), tuned tau=0.65 | 24.40 | +1.09 | -0.17 | 0.9352 | 29.96 | 2.86 | 23.66 | 0.48% | 0.0438 | 0.0472 |
| A2b_B2_hard_fcm | `B2` | Hard FCM mask M = 1[mu_raw > 0.5] | 23.11 | -0.20 | -1.46 | 0.9225 | 28.74 | 3.46 | 22.78 | 0.59% | 0.0550 | 0.0623 |
| Ref_F0_soft_fuzzy | `F0` | Soft fuzzy transmission blend (hand-set defaults) | 23.31 | +0.00 | -1.26 | 0.9258 | 28.78 | 3.43 | 22.99 | 0.58% | 0.0536 | 0.0561 |
| Ref_F1_fuzzy_ga | `F1` | Soft fuzzy transmission blend + GA-tuned params | 24.57 | +1.26 | +0.00 | 0.9356 | 30.19 | 2.89 | 23.86 | 0.41% | 0.0429 | 0.0537 |
| A3_no_f4 | `A3_no_f4` | Drop spatial vertical prior f4 (use f1..f3 only) | 23.26 | -0.05 | -1.32 | 0.9249 | 28.54 | 3.76 | 23.11 | 0.70% | 0.0529 | 0.0567 |
| A4_no_f3 | `A4_no_f3` | Drop gradient feature f3 (use f1, f2, f4 only) | 22.54 | -0.77 | -2.03 | 0.9182 | 27.16 | 4.46 | 22.95 | 0.73% | 0.0561 | 0.0593 |
| A5a_with_f5 | `A5a_with_f5` | Add local intensity variance f5 (5 features) | 23.16 | -0.15 | -1.41 | 0.9241 | 28.68 | 3.55 | 22.89 | 0.56% | 0.0547 | 0.0568 |
| A5b_with_f6 | `A5b_with_f6` | Add normalized dark channel f6 (5 features) | 23.20 | -0.11 | -1.37 | 0.9263 | 28.45 | 3.50 | 22.84 | 0.68% | 0.0587 | 0.0587 |
| A6a_c2 | `A6a_c2` | FCM cluster count c = 2 | 24.76 | +1.45 | +0.19 | 0.9382 | 30.46 | 2.72 | 23.89 | 0.44% | 0.0409 | 0.0446 |
| A6b_c4 | `A6b_c4` | FCM cluster count c = 4 | 21.77 | -1.54 | -2.80 | 0.9143 | 25.91 | 4.88 | 22.52 | 0.88% | 0.0694 | 0.0659 |
| A7_tolerance | `A7_tolerance` | Additive tolerance lift vs multiplicative blend | 21.00 | -2.31 | -3.57 | 0.9039 | 24.23 | 5.94 | 22.12 | 0.50% | 0.0748 | 0.0838 |
| A8_no_mu_gf | `A8_no_mu_gf` | Skip guided filter on membership mu_raw | 23.23 | -0.08 | -1.34 | 0.9243 | 28.81 | 3.43 | 22.91 | 0.58% | 0.0539 | 0.0595 |
| A9a_m1.5 | `A9a_m1.5` | FCM fuzziness exponent m = 1.5 | 23.62 | +0.31 | -0.95 | 0.9276 | 29.41 | 3.18 | 23.06 | 0.53% | 0.0504 | 0.0552 |
| A9b_m3.0 | `A9b_m3.0` | FCM fuzziness exponent m = 3.0 | 22.94 | -0.37 | -1.63 | 0.9252 | 27.53 | 3.80 | 22.81 | 0.71% | 0.0629 | 0.0583 |
| A10_gf_before_blend | `A10_gf_before_blend` | Guided filter before blend instead of after | 22.61 | -0.70 | -1.97 | 0.9176 | 27.67 | 3.66 | 22.28 | 0.53% | 0.0714 | 0.0754 |
