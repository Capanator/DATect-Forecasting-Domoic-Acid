#!/usr/bin/env python3
"""
Lag Ablation Spike Analysis
===========================

Runs a quick, leak-free retrospective with and without lag features to assess
their effect on overall metrics and especially on large spikes.

Configuration:
- Uses small n_anchors for speed (default 30 per site)
- Model: XGBoost regression with current config params

Outputs:
- Overall R²/MAE/F1@threshold
- Spike-only MAE and bias for actual >=40 and >=60 ppm
"""

import copy
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from forecasting.forecast_engine import ForecastEngine
from backend.api import _compute_summary


def run_eval(use_lags: bool, n_anchors: int = 100):
    # Snapshot config
    saved_use_lags = cfg.USE_LAG_FEATURES
    saved_lags = copy.deepcopy(cfg.LAG_FEATURES)
    try:
        cfg.USE_LAG_FEATURES = use_lags
        cfg.LAG_FEATURES = [1,2,3,7,14] if use_lags else []

        engine = ForecastEngine(validate_on_init=False)
        df = engine.run_retrospective_evaluation(
            task="regression",
            model_type="xgboost",
            n_anchors=n_anchors,
            min_test_date="2008-01-01",
        )
        if df is None or df.empty:
            return None, None

        # Build base_results in canonical format
        results_json = []
        for _, r in df.iterrows():
            results_json.append({
                'date': r['date'].strftime('%Y-%m-%d'),
                'site': r['site'],
                'actual_da': float(r['actual_da']) if pd.notnull(r['actual_da']) else None,
                'predicted_da': float(r['predicted_da']) if pd.notnull(r['predicted_da']) else None,
            })
        summary = _compute_summary(results_json)

        # Spike-only diagnostics
        spikes40 = df[df['actual_da'] >= 40]
        spikes60 = df[df['actual_da'] >= 60]

        def metrics(sub):
            if sub is None or sub.empty:
                return { 'count': 0 }
            ae = np.abs(sub['predicted_da'] - sub['actual_da'])
            bias = (sub['predicted_da'] - sub['actual_da'])
            return {
                'count': int(len(sub)),
                'mae': float(ae.mean()),
                'bias': float(bias.mean()),
                'underpred_rate': float((bias < 0).mean()),
            }

        s40 = metrics(spikes40)
        s60 = metrics(spikes60)

        return summary, { 'spike40': s40, 'spike60': s60 }
    finally:
        cfg.USE_LAG_FEATURES = saved_use_lags
        cfg.LAG_FEATURES = saved_lags


def main():
    print("Lag Ablation Spike Analysis (n_anchors/site=100)\n")

    with_lags_summary, with_lags_spikes = run_eval(use_lags=True, n_anchors=30)
    without_lags_summary, without_lags_spikes = run_eval(use_lags=False, n_anchors=30)

    def fmt_summary(name, s):
        if s is None:
            print(f"{name}: no results")
            return
        print(f"{name} Overall: R2={s.get('r2_score',0):.3f} MAE={s.get('mae',0):.2f} F1@{cfg.SPIKE_THRESHOLD_PPM:g}={s.get('f1_score',0):.3f}")

    def fmt_spikes(name, sp):
        if sp is None:
            return
        s40, s60 = sp['spike40'], sp['spike60']
        print(f"{name} Spikes>=40: N={s40['count']} MAE={s40.get('mae',0):.2f} Bias={s40.get('bias',0):+.2f} Under%={s40.get('underpred_rate',0):.2f}")
        print(f"{name} Spikes>=60: N={s60['count']} MAE={s60.get('mae',0):.2f} Bias={s60.get('bias',0):+.2f} Under%={s60.get('underpred_rate',0):.2f}")

    fmt_summary("With lags", with_lags_summary)
    fmt_spikes("With lags", with_lags_spikes)
    print()
    fmt_summary("Without lags", without_lags_summary)
    fmt_spikes("Without lags", without_lags_spikes)


if __name__ == "__main__":
    main()
