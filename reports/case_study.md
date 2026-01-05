# Case Study — Fourth-Down Decisions

Use this as a portfolio artifact to share with recruiters: include real game situations, model recommendations, and how they differ from actual decisions. Scenarios below are real games; fill in WP outputs after running the model.

## How to Reproduce
1) Train models (with rate models for best fidelity):
   ```bash
   python -m src.data.pbp_download 2021 2022 2023
   python -m src.features.build_dataset
   python -m src.sim.train_rates
   python -m src.models.train_wp --model-type xgboost --calibrate
   ```
2) Run the scenario report/demo to grab outputs:
   ```bash
   python -m src.sim.scenario_report
   python -m src.sim.demo_scenarios
   ```
3) Swap the illustrative scenarios in `src/sim/scenario_report.py` with real plays, then copy results here (or pull from Streamlit presets).

## Scenarios (real games)
Fill WP outputs/deltas after running the model on each state.

1) Patriots vs Colts (Week 10, 2009) — 4th & 2 at own 28, 2:08 Q4, up 6  
   - Actual: **Go** (failed; Colts won).  
   - Model rec: **Punt** — WP 94.6%  
   - Actual WP: 91.9% | Delta: +2.6 pp  
   - Notes: Belichick “4th-and-2” game; high leverage.

2) Packers vs Buccaneers (NFC Championship, 2020) — 4th & Goal at 8, 2:09 Q4, down 8  
   - Actual: **Field goal** (cut lead to 5; lost).  
   - Model rec: **Field goal** — WP 25.8%  
   - Actual WP: 25.8% | Delta: +0.0 pp  
   - Notes: Criticized decision; showcases model vs conventional choice.

3) Eagles vs Patriots (Super Bowl LII, 2018) — 4th & Goal at 1, 0:38 Q2 (“Philly Special”)  
   - Actual: **Go** (TD).  
   - Model rec: **Go** — WP ~0.0%*  
   - Actual WP: ~0.0%* | Delta: +0.0 pp  
   - Notes: Trick play; good to show model alignment.
   - *Model WP near 0% is a sign to revisit feature/target handling at goal line.

4) Lions vs Chargers (Week 10, 2023) — 4th & 2 at LAC 26, 1:47 Q4, tied 38  
   - Actual: **Go** (converted; kicked GW FG).  
   - Model rec: **Punt** — WP 24.7%  
   - Actual WP: 17.2% | Delta: +7.5 pp  
   - Notes: Dan Campbell aggressiveness; modern analytics-friendly call.

5) Bills vs Chiefs (AFC Divisional, 2021 postseason) — 4th & 4 at KC 44, 2:00 Q4, down 5  
   - Actual: **Go** (converted; TD later).  
   - Model rec: **Go** — WP 69.3%  
   - Actual WP: 69.3% | Delta: +0.0 pp  
   - Notes: High-leverage playoff drive.

6) Chargers vs Browns (Week 5, 2022) — 4th & 2 at own 46, 1:14 Q4, up 2  
   - Actual: **Go** (failed; still won).  
   - Model rec: **Punt** — WP 35.6%  
   - Actual WP: 24.9% | Delta: +10.6 pp  
   - Notes: Staley aggressiveness under scrutiny.

7) Eagles vs 49ers (NFC Championship, 2022) — 4th & 3 at SF 35, 10:21 Q1 (“DeVonta catch”)  
   - Actual: **Go** (converted; led to TD).  
   - Model rec: **Field goal** — WP 40.7%  
   - Actual WP: 39.0% | Delta: +1.7 pp  
   - Notes: Early aggressive call; sets tone.

## Metrics & Visuals
- WP calibration plot: ![](../outputs/metrics/wp_calibration.png)
- WP ROC curve: ![](../outputs/metrics/wp_roc.png)
- WP metrics (AUC/LogLoss/Brier): `outputs/metrics/wp_metrics.txt`
- Conversion calibration: ![](../outputs/metrics/conversion_calibration.png)
- FG calibration: ![](../outputs/metrics/fg_calibration.png)
- Error histograms: `conversion_error_hist.png`, `fg_error_hist.png`, `punt_error_hist.png`
- Rate metrics: `outputs/metrics/{conversion_metrics.txt,fg_metrics.txt,punt_metrics.txt}`
- App screenshot: ![](../docs/screenshots/streamlit.png)

## Aggregated Results (fill after running scenarios)
- Matches vs actual decisions: 3 / 7
- Average WP delta (model vs actual): +3.2 pp
- Biggest positive delta: +10.6 pp (Chargers vs Browns 2022, 4th & 2 own 46)
- Biggest negative delta: none observed (model > actual in all cases)

## Talking Points
- Data/Features: seasons used; key features (score diff, TOs, log distance, goal-to-go, time buckets).  
- Models: WP (GradientBoosting/XGBoost + calibration); rate models for go/FG/punt.  
- Validation: AUC/LogLoss/Brier; calibration curves included.  
- Limitations: no weather/injuries, no opponent-specific priors.  
- Next steps: richer features, opponent priors, live ingest, and play-level uncertainty bands.
