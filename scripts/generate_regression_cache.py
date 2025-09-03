#!/usr/bin/env python3
"""
Generate regression XGBoost retrospective cache only, using current config and
model factory settings. Writes to cache/retrospective/regression_xgboost.parquet
and .json, similar to precompute_cache.py but limited to one combo.
"""

import json
import sys
import os
import pandas as pd
from pathlib import Path

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import get_forecast_engine, clean_float_for_json, _compute_summary
import config


def main():
    cache_dir = Path("./cache/retrospective")
    cache_dir.mkdir(parents=True, exist_ok=True)

    engine = get_forecast_engine()
    n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 500)

    results_df = engine.run_retrospective_evaluation(
        task="regression",
        model_type="xgboost",
        n_anchors=n_anchors,
        min_test_date="2008-01-01",
    )

    if results_df is None or results_df.empty:
        print("No results generated.")
        return

    # Save parquet
    out_base = cache_dir / "regression_xgboost"
    results_df.to_parquet(f"{out_base}.parquet", index=False)

    # Save JSON (canonical keys)
    results_json = []
    for _, row in results_df.iterrows():
        rec = {
            'date': row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
            'site': row['site'],
            'actual_da': clean_float_for_json(row.get('actual_da')),
            'predicted_da': clean_float_for_json(row.get('predicted_da')),
        }
        if 'actual_category' in row:
            rec['actual_category'] = clean_float_for_json(row.get('actual_category'))
        if 'predicted_category' in row:
            rec['predicted_category'] = clean_float_for_json(row.get('predicted_category'))
        if 'anchor_date' in results_df.columns and pd.notnull(row.get('anchor_date', None)):
            rec['anchor_date'] = row['anchor_date'].strftime('%Y-%m-%d')
        results_json.append(rec)

    with open(f"{out_base}.json", 'w') as f:
        json.dump(results_json, f, indent=2)

    try:
        summary = _compute_summary(results_json)
        if 'r2_score' in summary:
            print(
                f"Saved {len(results_df)} predictions. Metrics: R²={summary['r2_score']:.4f}, MAE={summary.get('mae', 0):.2f}, F1@{getattr(config,'SPIKE_THRESHOLD_PPM',20.0)}={summary.get('f1_score', 0):.4f}"
            )
        else:
            print(f"Saved {len(results_df)} predictions.")
    except Exception:
        print(f"Saved {len(results_df)} predictions.")


if __name__ == "__main__":
    main()
