# Baseline results — 28 September 2026

The reference point for every future experiment. Before keeping any change,
re-run the same command and compare against the numbers here.

- **Code version:** commit `54057a3` (no code changes since).
- **Data:** full downloads of XJTU-SY (15 bearings), PHM 2012 (17 bearings,
  every snapshot), SEU Mechanical-datasets (20 recordings) and CWRU (40 files).
- **Reproducible:** the RUL and fault validation numbers are identical to the
  first full-data run on 25 September (fixed random seed `RUL_RANDOM_STATE`).
- **Re-create everything:** `sh results/baseline_2026-09-28/run_all.sh` from the
  project root (about 15 minutes).
- **Full output per model and per bearing:** the `raw/` folder.

## 1. RUL (remaining useful life), `tools.rul`

RUL is a fraction of life: 1.0 = new, 0.0 = failed. MAE 0.20 = off by 20% of
life on average. Rank = does the prediction fall in the right order as the
bearing ages (1.0 = perfect)? Validation bearings were never used in training.

### XJTU → PHM (train 15 XJTU-SY bearings, validate 17 PHM bearings)

| Model | Cross val MAE / R² | Validation MAE | RMSE | R² | Rank |
| --- | --- | --- | --- | --- | --- |
| Always guess 0.5 | — | 0.250 | 0.289 | 0.00 | — |
| Linear regression | 0.212 / 0.12 | 0.311 | 0.367 | -0.83 | 0.74 |
| **Random forest** (default) | 0.209 / 0.19 | **0.216** | 0.257 | 0.15 | 0.53 |
| Gradient boosting | 0.193 / 0.32 | 0.268 | 0.317 | -0.35 | 0.65 |

Worst bearings for random forest: Bearing2_5 (rank -0.41), Bearing1_5
(rank -0.03, R² -0.66), Bearing1_7 (rank 0.20). Only random forest beats
guessing 0.5, and only by 0.034.

Raw: `raw/rul_xjtu_to_phm.txt`

### PHM Learning_set → Full_Test_Set (train 6, validate 11 PHM bearings)

| Model | Cross val MAE / R² | Validation MAE | RMSE | R² | Rank |
| --- | --- | --- | --- | --- | --- |
| Always guess 0.5 | — | 0.250 | 0.289 | 0.00 | — |
| Linear regression | 0.202 / 0.19 | 0.257 | 0.328 | -0.39 | 0.65 |
| **Random forest** | 0.200 / 0.17 | **0.198** | 0.243 | 0.22 | 0.83 |
| Gradient boosting | 0.182 / 0.32 | 0.215 | 0.259 | 0.10 | **0.90** |

Worst bearings for random forest: Bearing2_7 (R² -0.68, a 38 minute life),
Bearing1_7 (R² -0.36), Bearing2_5 (rank 0.42). Best: Bearing1_3 (R² 0.83).

Raw: `raw/rul_phm_learning_to_full_test.txt`, `raw/rul_constant_guess.txt`

## 2. Fault classification + anomaly detection on test rigs (fault validation)

Train on the earlier part of each recording, test on a later part.

| | SEU drivetrain | CWRU bearings |
| --- | --- | --- |
| Classes | 9 (4 gear, 4 bearing, healthy) | 4 (ball, inner, outer, healthy) |
| Accuracy / macro F1 | **0.996 / 0.996** | **0.980 / 0.978** |
| Anomaly detection rate | 100% for every fault | 100% except ball fault at load 0 (72%) |
| ROC AUC | 0.96 – 1.00 | 0.99 – 1.00 |
| False alarms on healthy | 3% – 11% | 0% at loads 0–2, **22% at load 3** |

Raw: `raw/fault_validation.json`

## 3. Mechanical anomaly tool, `tools.mechanical_anomaly`

One healthy model per rig and condition (8 setups), scored window by window.
Same data as section 2 but a different windowing, so the numbers differ slightly.

| Setup | False alarms | Weakest detection |
| --- | --- | --- |
| CWRU load 0 | 0.0% | ball fault 64.8% (AUC 0.979) |
| CWRU load 1 | 0.0% | ball fault 96.3% |
| CWRU load 2 | 0.0% | all 100% |
| CWRU load 3 | **22.2%** | all 100% |
| SEU bearing 20-0 | 9.7% | all 100% |
| SEU bearing 30-2 | 11.1% | all 100% (ball fault AUC 0.956) |
| SEU gear 20-0 | 2.8% | all 100% |
| SEU gear 30-2 | 9.7% | all 100% |

Raw: `raw/mechanical_anomaly.txt`, `raw/mechanical_anomaly_report.json`

## 4. Partner data pipeline, `tools.partner_data` — SYNTHETIC ONLY

Run on the generated example dataset (12 runs, 2 joints). **These numbers say
nothing about real performance**; they only show the pipeline works end to end.
Replace with real partner data when it arrives.

| Check | Result |
| --- | --- |
| False alarms (unseen normal runs) | 2.0% (joint 1), 0.0% (joint 2) |
| Detection | friction 98.7%, cooling 75.0%, bearing 52.6%, backlash 22.4% |
| Fault classification | accuracy 0.875, macro F1 0.641 |

Raw: `raw/partner_synthetic.txt`

## 5. Simulator anomaly + health score — not re-run here

Only available inside the app (`python main.py`, Anomaly and Maintenance tabs),
so it cannot be scripted yet. Reference numbers are in the main
[README.md](../../README.md), section "Phase 7 → Measured on the dev machine".

## Known weak spots (what to improve)

1. XJTU → PHM transfer: barely beats guessing (MAE 0.216 vs 0.250), rank 0.53.
2. Short-life bearings (Bearing2_7, 38 min) and sudden failures (Bearing1_4).
3. CWRU load 3: 22% false alarms.
4. Smallest ball fault at CWRU load 0: detected 65–72%.
5. Partner pipeline: backlash and bearing detection weak even on synthetic data.

## Experiment log

Add one row per experiment. Change one thing at a time.

| Date | Experiment | Protocol | MAE | R² | Rank | Worst bearing | Keep? | Why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-28 | Baseline (random forest) | XJTU → PHM | 0.216 | 0.15 | 0.53 | Bearing2_5 (rank -0.41) | — | reference |
| 2026-09-28 | Baseline (random forest) | PHM → PHM | 0.198 | 0.22 | 0.83 | Bearing2_7 (R² -0.68) | — | reference |
| 2026-09-28 | Monotone RUL (EMA 900 s + running min) | XJTU → PHM | 0.212 | 0.16 | 0.94 | Bearing2_5 (R² -0.82) | No | rank becomes almost automatic (0.94–1.00 for every model, even with R² < 0); MAE barely moves; hurts PHM → PHM (0.198 → 0.216) |
| 2026-09-28 | Smooth predictions, EMA 300 s | XJTU → PHM | 0.208 | 0.23 | 0.62 | Bearing2_5 (rank -0.58) | No | helps, but 900 s is better on dev |
| 2026-09-28 | **Smooth predictions, EMA 900 s, only when predicting on a different dataset** | XJTU → PHM | **0.203** | **0.27** | **0.66** | Bearing2_5 (rank -0.81) | **Yes** | better on dev (6 Learning_set) AND on the 11 untouched Full_Test bearings; PHM → PHM unchanged |

Details and scripts: `results/experiments/` (`xjtu_to_phm_rank.py`, `compare_*_after.txt`).
| 2026-09-29 | **Fix 1: loudness from the quietest point so far** (5 min EMA, running min) | XJTU → PHM | 0.196 | 0.31 | 0.81 | Bearing2_7 (R² -0.23) | **Yes** (XJTU → PHM only) | fixes run-in bearings (Bearing2_5 rank -0.81 → 0.57, 1_5 -0.08 → 0.88); hurts PHM → PHM (MAE 0.198 → 0.279), so set per protocol |
| 2026-09-29 | Fix 2: trend features (fast − slow EMA) | XJTU → PHM | 0.195 | 0.32 | 0.80 | — | No | dev gain MAE −0.003 = noise; test rank down |
| 2026-09-29 | Fix 2: shape only (drop loudness features) | XJTU → PHM | 0.218 | 0.06 | 0.64 | — | No | clearly worse on dev: loudness is useful once measured correctly |
| 2026-09-29 | Fix 3: train XJTU + PHM Learning (21 bearings) | → PHM Full_Test (11) | 0.180 | 0.37 | 0.65 | Bearing1_5 (rank -0.25) | Protocol only | changes the question; lower rank than XJTU → PHM with Fixes 1 + 4 (0.172 / 0.42 / 0.85 on the same 11) |
| 2026-09-29 | Fix 4: full two-stage (train on degradation only), k = 3 / 5 / 8 | XJTU → PHM | 0.194–0.201 | 0.25–0.34 | 0.78–0.83 | — | No | no clear dev gain over Fix 1 |
| 2026-09-29 | **Fix 4: gate** (floor 0.72 before degradation is detected, k = 5) | XJTU → PHM | **0.181** | **0.39** | **0.82** | Bearing2_7 (R² -0.24) | **Yes** (XJTU → PHM only) | blocks early false drops; better on all dev metrics and on test; neutral on PHM → PHM |

Full explanation of every row: `CODE_GUIDE_MECHANICAL.md`, section 9.
| 2026-10-01 | A1/A2: less smoothing after degradation is detected, or proportional to time lived (4 variants) | XJTU → PHM | 0.191–0.195 | 0.24–0.32 | 0.68–0.77 | fixes Bearing2_7 (MAE 0.285 → 0.153–0.217) | No | noise returns for the other bearings; worse on dev and test |
| 2026-10-01 | B1/B2: stricter degradation detector (RMS only, or RMS and kurtosis) | XJTU → PHM | 0.188–0.198 | 0.22–0.38 | 0.87–0.88 | Bearing2_3 unchanged | No | better on dev, worse on test (gate stays closed too long) |
| 2026-10-01 | B3: kurtosis only counts once RMS has grown 0.2 | XJTU → PHM | 0.178 | 0.40 | 0.86 | fixes Bearing1_1 (rank 0.19 → 0.90); Bearing2_3 unchanged | No (option kept, off) | dev gain comes from one dev bearing; test slightly worse (0.172 → 0.179) |
| 2026-10-01 | **Sturdier degradation detector: minimum spread 5%** (`RUL_FPT_STD_FLOOR`) | XJTU → PHM | **0.175** | **0.43** | **0.84** | Bearing2_7 (R² -0.23) | **Yes** | old detector fired on a bearing that never wears (6 of 6); this one never does; best on dev among the variants that pass that check; test agrees (0.172 → 0.158) |
| 2026-10-01 | Detector alternatives: 8 / 12 std, minimum spread 10%, 30 min normal window | XJTU → PHM | 0.170–0.197 | 0.28–0.45 | 0.83–0.86 | — | No | still fire on noise (8 std, 30 min) or clearly worse on dev (12 std, 10%) |
| 2026-10-01 | **Uncertainty range: prediction ± 0.42** (90% quantile of cross validation errors) | XJTU → PHM | — | — | — | Bearing1_1 (73% coverage) | **Yes** | coverage 91% on dev, 95% on test, promise 90%; width by predicted value rejected (83% on dev) |
| 2026-10-01 | **Anomaly engine: single feature check** (99th percentile, min 6 robust std; fixed in advance) | SEU + CWRU | — | — | — | CWRU load 3 false alarms still 22% | **Yes** | CWRU load 0 ball fault 64.8% → 96.3% detected; false alarms +1.4 points on three SEU setups |

Sections 10 and 11 of `CODE_GUIDE_MECHANICAL.md` explain these four rows. Note: the
frozen files in `raw/` are from 28 September, before these changes.
| 2026-10-03 | **Anomaly alarms need 2 flagged windows in a row** (rule fixed before measuring) | SEU + CWRU | — | — | — | synthetic backlash 23.7% → 5.3% detected | **Yes** | false alarms 8.8% → 1.7% overall, CWRU load 3 22.2% → 0%; strong faults lose only the first window |
| 2026-10-03 | Train without dominant frequency features | SEU + CWRU | — | — | — | CWRU load 0 ball fault 96.3% → 90.7% | No | false alarms not better (8.8% → 10.0%) |
| 2026-10-03 | **32-bearing evaluation, both directions** (`eval32.py`) | new machine + new bearing | — | — | — | PHM → XJTU: MAE 0.310, worse than guessing | method adopted | this week's fixes: XJTU → PHM MAE −0.041 [−0.074, −0.004] (real); PHM → XJTU +0.031 [+0.002, +0.060] (worse); rank better both ways (0.54 → 0.83); with the lab seen they hurt (+0.020), confirming per-protocol settings |
| 2026-10-03 | B3 again: kurtosis counts only once RMS has grown 0.2 | 32 bearings | +0.006 vs current | — | — | XJTU Bearing3_5 rank 1.00 → −0.68 | No | interval [−0.021, +0.032] contains 0 in view A; 10 better / 14 worse |

Section 12 of `CODE_GUIDE_MECHANICAL.md` explains these rows and corrects the "new machine" claim.
| 2026-10-03 | Which fix hurts PHM → XJTU? (each fix alone and in pairs) | 32 bearings, new machine | — | — | — | — | diagnosis | the quietest-point reference: PHM → XJTU MAE 0.280 → 0.300–0.310; smoothing and gate leave it at 0.280. It also gives the ordering (rank 0.70 → 0.83). Per-direction switch not adopted (would be chosen on the scored bearings) |
| 2026-10-03 | Smoothing over 5 / 10 / 20% of the bearing's age | 32 bearings | +0.004 to +0.007 vs current | — | −0.04 to −0.11 | — | No | no gain, worse ordering |
| 2026-10-03 | Asymmetric smoothing (fast down, slow up) | 32 bearings | 0.000 / +0.001 | — | −0.11 / −0.15 | — | No | lets downward noise through |
| 2026-10-03 | Average of model types (3, or RF + GB) | 32 bearings | −0.009 | — | +0.03 | helps 18–20, hurts 9–10 | No (yet) | interval [−0.021, +0.003] / [−0.019, +0.001] touches zero; 13 versions tested at once |
| 2026-10-03 | Three stage answer (early / late / near failure), boundaries fixed in advance | both directions, 32 bearings | — | — | — | never says "near failure" on 19 of 32 bearings | kept as a measure | right stage 55–58% vs 50% for always "early"; near failure caught 34% (XJTU → PHM) and 15% (PHM → XJTU); too early 0–4% |
| 2026-10-03 | Stage cut-offs calibrated on held-out predictions | both directions | — | — | — | — | No | right stage 58% → 51% and 55% → 55%; cut-offs barely move |
| 2026-10-03 | NASA / IMS dataset checked | — | — | — | — | — | not added | 12 bearings but only 4 failed, from 3 tests; one channel per bearing in tests 2–3; not downloaded |

Section 13 of `CODE_GUIDE_MECHANICAL.md` explains these rows. The published model is now
`models/rul_random_forest_xjtu_to_phm.joblib` (trained 3 Oct with the current code).
