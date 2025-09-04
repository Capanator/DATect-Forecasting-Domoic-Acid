#!/usr/bin/env python3
"""
Spike Weight / Tweedie Sweep
============================

Runs a compact sweep over MASSIVE spike weights/thresholds and Tweedie options,
reporting overall metrics and spike-only MAE/bias for >=40 and >=60 ppm.

Defaults keep runtime modest: n_anchors/site=50.
"""

import sys, os, copy
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from forecasting.forecast_engine import ForecastEngine
from backend.api import _compute_summary


def eval_current(n_anchors=50):
    engine = ForecastEngine(validate_on_init=False)
    df = engine.run_retrospective_evaluation(
        task="regression",
        model_type="xgboost",
        n_anchors=n_anchors,
        min_test_date="2008-01-01",
    )
    if df is None or df.empty:
        return None, None
    results_json = []
    for _, r in df.iterrows():
        results_json.append({
            'date': r['date'].strftime('%Y-%m-%d'),
            'site': r['site'],
            'actual_da': float(r['actual_da']) if pd.notnull(r['actual_da']) else None,
            'predicted_da': float(r['predicted_da']) if pd.notnull(r['predicted_da']) else None,
        })
    summary = _compute_summary(results_json)
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
    return summary, { 'spike40': metrics(spikes40), 'spike60': metrics(spikes60) }


def run_sweep():
    saved = {
        'MASSIVE_SPIKE_THRESHOLD_PPM': getattr(cfg, 'MASSIVE_SPIKE_THRESHOLD_PPM', 60.0),
        'MASSIVE_SPIKE_WEIGHT_MULT': getattr(cfg, 'MASSIVE_SPIKE_WEIGHT_MULT', 2.0),
        'USE_TWEEDIE_REGRESSION': getattr(cfg, 'USE_TWEEDIE_REGRESSION', False),
        'TWEEDIE_VARIANCE_POWER': getattr(cfg, 'TWEEDIE_VARIANCE_POWER', 1.3),
    }
    try:
        configs = []
        for thr in [50.0, 60.0, 70.0]:
            for mult in [3.0, 4.0]:
                configs.append({
                    'name': f"massive_thr={thr:g}_mult={mult:g}",
                    'MASSIVE_SPIKE_THRESHOLD_PPM': thr,
                    'MASSIVE_SPIKE_WEIGHT_MULT': mult,
                    'USE_TWEEDIE_REGRESSION': False,
                })
        # Tweedie tests at a representative massive setting
        for power in [1.2, 1.5]:
            configs.append({
                'name': f"tweedie_power={power}",
                'MASSIVE_SPIKE_THRESHOLD_PPM': 60.0,
                'MASSIVE_SPIKE_WEIGHT_MULT': 4.0,
                'USE_TWEEDIE_REGRESSION': True,
                'TWEEDIE_VARIANCE_POWER': power,
            })

        print("Spike Weight / Tweedie Sweep (n_anchors/site=50)\n")
        for i, c in enumerate(configs, 1):
            # Apply config
            cfg.MASSIVE_SPIKE_THRESHOLD_PPM = c['MASSIVE_SPIKE_THRESHOLD_PPM']
            cfg.MASSIVE_SPIKE_WEIGHT_MULT = c['MASSIVE_SPIKE_WEIGHT_MULT']
            cfg.USE_TWEEDIE_REGRESSION = c.get('USE_TWEEDIE_REGRESSION', False)
            if cfg.USE_TWEEDIE_REGRESSION:
                cfg.TWEEDIE_VARIANCE_POWER = c.get('TWEEDIE_VARIANCE_POWER', 1.3)
            # Eval
            overall, spikes = eval_current(n_anchors=50)
            if overall is None:
                print(f"[{i}/{len(configs)}] {c['name']}: no results")
                continue
            print(f"[{i}/{len(configs)}] {c['name']}")
            print(f"  Overall: R2={overall.get('r2_score',0):.3f} MAE={overall.get('mae',0):.2f} F1@{getattr(cfg,'SPIKE_THRESHOLD_PPM',20.0):.0f}={overall.get('f1_score',0):.3f}")
            s40, s60 = spikes['spike40'], spikes['spike60']
            print(f"  Spikes>=40: N={s40['count']} MAE={s40.get('mae',0):.2f} Bias={s40.get('bias',0):+.2f} Under%={s40.get('underpred_rate',0):.2f}")
            print(f"  Spikes>=60: N={s60['count']} MAE={s60.get('mae',0):.2f} Bias={s60.get('bias',0):+.2f} Under%={s60.get('underpred_rate',0):.2f}")
    finally:
        cfg.MASSIVE_SPIKE_THRESHOLD_PPM = saved['MASSIVE_SPIKE_THRESHOLD_PPM']
        cfg.MASSIVE_SPIKE_WEIGHT_MULT = saved['MASSIVE_SPIKE_WEIGHT_MULT']
        cfg.USE_TWEEDIE_REGRESSION = saved['USE_TWEEDIE_REGRESSION']
        cfg.TWEEDIE_VARIANCE_POWER = saved['TWEEDIE_VARIANCE_POWER']


if __name__ == "__main__":
    run_sweep()

