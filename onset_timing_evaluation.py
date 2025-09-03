#!/usr/bin/env python3
"""
Onset Timing Evaluation (20 ppm)
================================

Evaluates the timing of the initial domoic acid spike (first crossing of 20 ppm)
and compares model forecasts against a naive baseline (actual shifted forward by 1 week),
focusing on on-time detection of the initial spike.

Notes
- Uses cached retrospective predictions at: cache/retrospective/regression_xgboost.parquet
- Naive predictions are computed using ONLY data prior to each anchor_date (no leakage)
- Onset date is defined as the first date in a site-year where DA crosses >= 20 ppm from below
- Metrics emphasize onset timing and on-time detection rather than overall accuracy
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import sys

import config


def load_predictions():
    pred_path = "./cache/retrospective/regression_xgboost.parquet"
    df = pd.read_parquet(pred_path)
    for col in ["date", "anchor_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def compute_naive_baseline(pred_df: pd.DataFrame, hist_df: pd.DataFrame) -> pd.Series:
    """
    For each forecast row (site, anchor_date), produce a naive prediction equal to
    the historical DA value NAIVE_BASELINE_LAG_DAYS before the anchor date (±3 days).
    This uses ONLY data prior to the anchor_date.
    Returns a Series aligned with pred_df index.
    """
    lag_days = int(getattr(config, 'NAIVE_BASELINE_LAG_DAYS', 7))
    out = []
    for idx, row in pred_df.iterrows():
        site = row['site']
        anchor_date = row['anchor_date']
        site_hist = hist_df[(hist_df['site'] == site) & (hist_df['date'] < anchor_date) & (hist_df['da'].notna())]
        if site_hist.empty:
            out.append(np.nan)
            continue
        target = anchor_date - timedelta(days=lag_days)
        cands = site_hist[(site_hist['date'] >= target - timedelta(days=3)) & (site_hist['date'] <= target + timedelta(days=3))]
        if not cands.empty:
            cands = cands.copy()
            cands['diff'] = (cands['date'] - target).abs().dt.days
            out.append(float(cands.sort_values('diff').iloc[0]['da']))
        else:
            out.append(float(site_hist.sort_values('date').iloc[-1]['da']))
    return pd.Series(out, index=pred_df.index, name='naive_predicted_da')


def find_first_spike_events(hist_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Identify the first crossing of threshold per site-year: da >= threshold and previous
    observation < threshold.
    Returns a DataFrame with columns: site, year, first_spike_date
    """
    df = hist_df.sort_values(['site', 'date']).copy()
    df['year'] = df['date'].dt.year
    df['above'] = df['da'] >= threshold
    df['prev_above'] = df.groupby(['site', 'year'])['above'].shift(1).fillna(False)
    df['prev_da'] = df.groupby(['site', 'year'])['da'].shift(1)
    df['cross_up'] = df['above'] & (~df['prev_above']) & (df['prev_da'].fillna(0) < threshold)
    firsts = df[df['cross_up']].groupby(['site', 'year']).first().reset_index()
    return firsts[['site', 'year', 'date']].rename(columns={'date': 'first_spike_date'})


def earliest_crossing_date(series_df: pd.DataFrame, threshold: float) -> pd.Timestamp | None:
    """
    Given a DataFrame with columns ['date', 'value'] sorted by date,
    return the earliest date where value >= threshold, or None if not present.
    """
    hit = series_df[series_df['value'] >= threshold]
    if hit.empty:
        return None
    return pd.to_datetime(hit.iloc[0]['date'])


def evaluate_onset_timing():
    threshold = float(getattr(config, 'SPIKE_THRESHOLD_PPM', 20.0))
    tol_days = int(getattr(config, 'ONSET_TOLERANCE_DAYS', 3))

    # Load cached predictions and historical ground truth
    preds = load_predictions()
    hist = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    hist['date'] = pd.to_datetime(hist['date'])

    # Compute naive baseline aligned to predictions
    preds['naive_predicted_da'] = compute_naive_baseline(preds, hist)

    # Determine first spike events per site-year from historical truth
    events = find_first_spike_events(hist[['site', 'date', 'da']].dropna(), threshold)

    # For each event, gather predictions in a window around the event
    window_pre = int(getattr(config, 'NAIVE_BASELINE_LAG_DAYS', 7))  # allow early hits
    window_post = 14  # allow some lag, but penalize

    records = []
    for _, e in events.iterrows():
        site = e['site']
        year = int(e['year'])
        spike_date = pd.to_datetime(e['first_spike_date'])
        start = spike_date - timedelta(days=window_pre)
        end = spike_date + timedelta(days=window_post)

        site_preds = preds[(preds['site'] == site) & (preds['date'] >= start) & (preds['date'] <= end)].copy()
        if site_preds.empty:
            continue

        # Build time series for model and naive within window
        if 'onset_prob' in site_preds.columns:
            # Combine signals: cross occurs when either onset_prob >= prob_thresh or predicted_da >= threshold
            prob_thresh = float(getattr(config, 'ONSET_PROB_THRESHOLD', 0.5))
            prob_ts = site_preds[['date', 'onset_prob']].rename(columns={'onset_prob': 'value'}).sort_values('date')
            da_ts = site_preds[['date', 'predicted_da']].rename(columns={'predicted_da': 'value'}) if 'predicted_da' in site_preds.columns else None
            cross_prob = earliest_crossing_date(prob_ts, prob_thresh)
            cross_da = earliest_crossing_date(da_ts.sort_values('date'), threshold) if da_ts is not None else None
            if cross_prob is None:
                model_cross = cross_da
            elif cross_da is None:
                model_cross = cross_prob
            else:
                model_cross = min(cross_prob, cross_da)
            # For "hit at event" consider either condition
            def value_at_event_combined():
                vp = None
                vd = None
                cands_p = prob_ts[(prob_ts['date'] >= spike_date - timedelta(days=3)) & (prob_ts['date'] <= spike_date + timedelta(days=3))]
                if not cands_p.empty:
                    cands_p = cands_p.copy(); cands_p['diff'] = (cands_p['date'] - spike_date).abs().dt.days
                    vp = float(cands_p.sort_values('diff').iloc[0]['value'])
                if da_ts is not None:
                    cands_d = da_ts[(da_ts['date'] >= spike_date - timedelta(days=3)) & (da_ts['date'] <= spike_date + timedelta(days=3))]
                    if not cands_d.empty:
                        cands_d = cands_d.copy(); cands_d['diff'] = (cands_d['date'] - spike_date).abs().dt.days
                        vd = float(cands_d.sort_values('diff').iloc[0]['value'])
                return vp, vd
            vp, vd = value_at_event_combined()
            value_series_for_event = None
        else:
            model_ts = site_preds[['date', 'predicted_da']].rename(columns={'predicted_da': 'value'}).sort_values('date')
            model_cross = earliest_crossing_date(model_ts, threshold)
        naive_ts = site_preds[['date', 'naive_predicted_da']].rename(columns={'naive_predicted_da': 'value'}).sort_values('date')
        naive_cross = earliest_crossing_date(naive_ts, threshold)

        # Pred/naive value exactly at the spike date (closest within ±3 days)
        def value_at_event(df_values):
            cands = df_values[(df_values['date'] >= spike_date - timedelta(days=3)) & (df_values['date'] <= spike_date + timedelta(days=3))]
            if cands.empty:
                return np.nan
            cands = cands.copy()
            cands['diff'] = (cands['date'] - spike_date).abs().dt.days
            return float(cands.sort_values('diff').iloc[0]['value'])

        if 'onset_prob' in site_preds.columns:
            prob_thresh = float(getattr(config, 'ONSET_PROB_THRESHOLD', 0.5))
            model_hit_event = ((vp is not None and vp >= prob_thresh) or (vd is not None and vd >= threshold))
            model_at_event = (vd if vd is not None else (vp if vp is not None else np.nan))
        else:
            model_at_event = value_at_event(model_ts)
        naive_at_event = value_at_event(naive_ts)

        rec = {
            'site': site,
            'year': year,
            'first_spike_date': spike_date,
            'model_cross_date': model_cross,
            'naive_cross_date': naive_cross,
            'model_on_time': (model_cross is not None) and (abs((model_cross - spike_date).days) <= tol_days),
            'naive_on_time': (naive_cross is not None) and (abs((naive_cross - spike_date).days) <= tol_days),
            'model_lead_days': None if model_cross is None else int((model_cross - spike_date).days),
            'naive_lead_days': None if naive_cross is None else int((naive_cross - spike_date).days),
            'model_hit_at_event': (not np.isnan(model_at_event)) and (model_at_event >= threshold),
            'naive_hit_at_event': (not np.isnan(naive_at_event)) and (naive_at_event >= threshold),
        }
        records.append(rec)

    if not records:
        print("No eligible spike events found with sufficient prediction coverage.")
        sys.exit(0)

    df_eval = pd.DataFrame(records)
    covered = len(df_eval)

    def rate(col):
        vals = df_eval[col].dropna()
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals))

    def median_abs_err(col):
        vals = df_eval[col].dropna().values
        if len(vals) == 0:
            return None
        return float(np.median(np.abs(vals)))

    print("\nOnset Timing Evaluation (first crossing of {:.0f} ppm)".format(threshold))
    print("=" * 60)
    print(f"Events covered: {covered}")
    print(f"On-time rate (±{tol_days}d):  Model {rate('model_on_time'):.3f}  |  Naive {rate('naive_on_time'):.3f}")
    print(f"Median |lead| days:           Model {median_abs_err('model_lead_days')}  |  Naive {median_abs_err('naive_lead_days')}")
    print(f"Hit at event date:            Model {rate('model_hit_at_event'):.3f}  |  Naive {rate('naive_hit_at_event'):.3f}")

    # Optional: aggregate by site
    by_site = df_eval.groupby('site').agg(
        covered=('site', 'count'),
        model_on_time=('model_on_time', 'mean'),
        naive_on_time=('naive_on_time', 'mean'),
        model_med_abs_leadDays=('model_lead_days', lambda s: np.median(np.abs([v for v in s.dropna()] or [np.nan]))),
        naive_med_abs_leadDays=('naive_lead_days', lambda s: np.median(np.abs([v for v in s.dropna()] or [np.nan]))),
    ).reset_index()
    print("\nPer-site onset on-time rate (fraction on-time):")
    for _, r in by_site.iterrows():
        print(f"  {r['site']}: N={int(r['covered'])}  Model={r['model_on_time']:.3f}  Naive={r['naive_on_time']:.3f}")


if __name__ == "__main__":
    evaluate_onset_timing()
