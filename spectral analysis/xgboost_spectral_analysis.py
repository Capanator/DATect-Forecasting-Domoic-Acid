#!/usr/bin/env python3
"""
XGBoost Spectral Analysis of DA Forecasting
===========================================

Comprehensive spectral analysis of XGBoost predictions including:
- Power Spectral Density (PSD) for all sites
- Coherence between predicted and actual values
- Phase relationships and time-varying analysis
- Site-by-site and aggregate analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import our forecasting system
from forecasting.core.forecast_engine import ForecastEngine
import config

# For wavelet analysis
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("PyWavelets not installed. Install with: pip install PyWavelets")


def run_xgboost_retrospective():
    """Run XGBoost retrospective evaluation to get predictions."""
    print("Running XGBoost retrospective evaluation...")
    
    # Set to XGBoost
    original_model = config.FORECAST_MODEL
    config.FORECAST_MODEL = "xgboost"
    
    # Run evaluation with moderate anchors for spectral analysis
    engine = ForecastEngine()
    results_df = engine.run_retrospective_evaluation(
        task="regression",
        model_type="xgboost",
        n_anchors=30  # Reduced for faster execution
    )
    
    # Restore original model
    config.FORECAST_MODEL = original_model
    
    return results_df


def compute_power_spectral_density(signal_data, fs=1.0, method='welch'):
    """
    Compute Power Spectral Density of a signal.
    
    Args:
        signal_data: Time series data
        fs: Sampling frequency (1.0 for weekly data)
        method: 'welch', 'periodogram', or 'multitaper'
    """
    if len(signal_data) < 10:  # Need minimum data
        return None, None
        
    if method == 'welch':
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)//4))
    elif method == 'periodogram':
        freqs, psd = signal.periodogram(signal_data, fs=fs)
    else:  # multitaper
        freqs, psd = signal.periodogram(signal_data, fs=fs)
    
    return freqs, psd


def compute_coherence_phase(actual, predicted, fs=1.0):
    """
    Compute coherence and phase between actual and predicted signals.
    
    Returns:
        freqs: Frequency values
        coherence: Coherence values (0-1)
        phase: Phase difference in radians
    """
    if len(actual) < 10 or len(predicted) < 10:
        return None, None, None
        
    # Compute cross-spectral density
    freqs, Pxy = signal.csd(actual, predicted, fs=fs, nperseg=min(256, len(actual)//4))
    
    # Compute auto-spectral densities
    _, Pxx = signal.welch(actual, fs=fs, nperseg=min(256, len(actual)//4))
    _, Pyy = signal.welch(predicted, fs=fs, nperseg=min(256, len(predicted)//4))
    
    # Compute coherence
    coherence = np.abs(Pxy)**2 / (Pxx * Pyy)
    
    # Compute phase
    phase = np.angle(Pxy)
    
    return freqs, coherence, phase


def analyze_xgboost_spectral_patterns(results_df):
    """
    Analyze spectral patterns in XGBoost predictions vs actual for all sites.
    """
    # Group by site for site-specific analysis
    sites = results_df['site'].unique()
    
    all_results = {}
    aggregate_actual = []
    aggregate_predicted = []
    
    print(f"\n{'='*80}")
    print("XGBOOST SPECTRAL ANALYSIS FOR ALL SITES")
    print('='*80)
    
    for site in sites:
        print(f"\n{'='*60}")
        print(f"SPECTRAL ANALYSIS FOR {site.upper()}")
        print('='*60)
        
        site_data = results_df[results_df['site'] == site].copy()
        site_data = site_data.sort_values('date')
        
        # Get actual and predicted values
        actual = site_data['da'].values
        predicted = site_data['Predicted_da'].values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[valid_idx]
        predicted = predicted[valid_idx]
        
        if len(actual) < 20:
            print(f"Not enough data for {site} ({len(actual)} points)")
            continue
        
        # Add to aggregate analysis
        aggregate_actual.extend(actual)
        aggregate_predicted.extend(predicted)
        
        site_results = analyze_single_site_spectral(site, actual, predicted)
        all_results[site] = site_results
    
    # Aggregate analysis across all sites
    if aggregate_actual:
        print(f"\n{'='*60}")
        print("AGGREGATE SPECTRAL ANALYSIS (ALL SITES)")
        print('='*60)
        
        aggregate_actual = np.array(aggregate_actual)
        aggregate_predicted = np.array(aggregate_predicted)
        
        aggregate_results = analyze_single_site_spectral("All_Sites", aggregate_actual, aggregate_predicted)
        all_results["All_Sites"] = aggregate_results
    
    return all_results


def analyze_single_site_spectral(site_name, actual, predicted):
    """Analyze spectral properties for a single site."""
    site_results = {}
    
    # 1. POWER SPECTRAL DENSITY
    print("\n1. Power Spectral Density Analysis")
    print("-" * 40)
    
    # Compute PSD for actual and predicted
    freqs, psd_actual = compute_power_spectral_density(actual)
    _, psd_predicted = compute_power_spectral_density(predicted)
    
    if freqs is None:
        print("Insufficient data for spectral analysis")
        return {}
    
    # Convert frequency to period (weeks)
    periods = 1 / freqs[1:]  # Skip DC component
    
    # Find dominant frequencies
    dominant_idx = np.argsort(psd_actual[1:])[-3:][::-1]  # Top 3 frequencies
    
    print("Dominant periods in actual DA:")
    for idx in dominant_idx:
        print(f"  {periods[idx]:.1f} weeks ({periods[idx]/52:.2f} years)")
    
    # Compare spectral energy
    total_power_actual = np.sum(psd_actual)
    total_power_predicted = np.sum(psd_predicted)
    print(f"\nTotal spectral power:")
    print(f"  Actual: {total_power_actual:.2f}")
    print(f"  Predicted: {total_power_predicted:.2f}")
    print(f"  Ratio (Pred/Actual): {total_power_predicted/total_power_actual:.3f}")
    
    site_results['psd'] = {
        'freqs': freqs,
        'psd_actual': psd_actual,
        'psd_predicted': psd_predicted,
        'dominant_periods': periods[dominant_idx],
        'power_ratio': total_power_predicted/total_power_actual
    }
    
    # 2. COHERENCE AND PHASE
    print("\n2. Coherence and Phase Analysis")
    print("-" * 40)
    
    freqs_coh, coherence, phase = compute_coherence_phase(actual, predicted)
    
    if freqs_coh is None:
        print("Insufficient data for coherence analysis")
        return site_results
    
    # Average coherence in different frequency bands
    # Low frequency (> 26 weeks, semi-annual and longer)
    low_freq_mask = freqs_coh < 1/26
    low_freq_coherence = np.mean(coherence[low_freq_mask]) if np.any(low_freq_mask) else 0
    
    # Mid frequency (4-26 weeks, monthly to semi-annual)
    mid_freq_mask = (freqs_coh >= 1/26) & (freqs_coh < 1/4)
    mid_freq_coherence = np.mean(coherence[mid_freq_mask]) if np.any(mid_freq_mask) else 0
    
    # High frequency (< 4 weeks, sub-monthly)
    high_freq_mask = freqs_coh >= 1/4
    high_freq_coherence = np.mean(coherence[high_freq_mask]) if np.any(high_freq_mask) else 0
    
    print("Coherence by frequency band:")
    print(f"  Low freq (>26 weeks): {low_freq_coherence:.3f}")
    print(f"  Mid freq (4-26 weeks): {mid_freq_coherence:.3f}")
    print(f"  High freq (<4 weeks): {high_freq_coherence:.3f}")
    
    # Phase analysis
    phase_deg = np.rad2deg(phase)
    significant_coherence_mask = coherence > 0.5
    if np.any(significant_coherence_mask):
        avg_phase_lag = np.mean(phase_deg[significant_coherence_mask])
        print(f"\nAverage phase lag (where coherence > 0.5): {avg_phase_lag:.1f}°")
        
        # Determine lead/lag relationship
        if avg_phase_lag > 0:
            print("  → XGBoost predictions LAG behind actual values")
        else:
            print("  → XGBoost predictions LEAD actual values")
    else:
        avg_phase_lag = np.nan
        print("\nNo significant coherence found (coherence > 0.5)")
    
    site_results['coherence'] = {
        'freqs': freqs_coh,
        'coherence': coherence,
        'phase': phase,
        'avg_phase_lag': avg_phase_lag,
        'band_coherence': {
            'low': low_freq_coherence,
            'mid': mid_freq_coherence,
            'high': high_freq_coherence
        }
    }
    
    # 3. TIME-VARYING SPECTRAL ANALYSIS
    print("\n3. Time-Varying Spectral Properties")
    print("-" * 40)
    
    # Compute spectrogram
    if len(actual) >= 32:
        window_size = min(32, len(actual)//4)
        f, t, Sxx_actual = signal.spectrogram(actual, fs=1.0, nperseg=window_size, noverlap=window_size//2)
        _, _, Sxx_pred = signal.spectrogram(predicted, fs=1.0, nperseg=window_size, noverlap=window_size//2)
        
        # Compare spectral evolution
        spectral_correlation = []
        for i in range(Sxx_actual.shape[1]):
            if np.any(Sxx_actual[:, i]) and np.any(Sxx_pred[:, i]):
                corr, _ = pearsonr(Sxx_actual[:, i], Sxx_pred[:, i])
                spectral_correlation.append(corr)
        
        if spectral_correlation:
            avg_spectral_corr = np.mean(spectral_correlation)
            print(f"Average spectral correlation over time: {avg_spectral_corr:.3f}")
            
            # Check if spectral properties change over time
            spectral_std = np.std(spectral_correlation)
            if spectral_std > 0.2:
                print("  → High variability: XGBoost performance varies with time")
            else:
                print("  → Low variability: Consistent XGBoost performance over time")
                
            site_results['spectrogram'] = {
                'freqs': f,
                'times': t,
                'spectral_correlation': spectral_correlation,
                'avg_spectral_corr': avg_spectral_corr
            }
    
    # 4. Overall Performance Metrics
    print("\n4. Overall XGBoost Performance")
    print("-" * 40)
    
    # R-squared
    r2 = np.corrcoef(actual, predicted)[0, 1]**2
    mae = np.mean(np.abs(actual - predicted))
    
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean Actual: {np.mean(actual):.2f}")
    print(f"Mean Predicted: {np.mean(predicted):.2f}")
    
    site_results['performance'] = {
        'r2': r2,
        'mae': mae,
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted)
    }
    
    return site_results


def create_comprehensive_spectral_plots(results, output_prefix="xgboost_spectral"):
    """Create comprehensive visualization plots for all sites."""
    
    sites = [site for site in results.keys() if site != "All_Sites"]
    n_sites = len(sites)
    
    if n_sites == 0:
        return
    
    # Create individual site plots
    for site in sites:
        create_single_site_plot(results, site, f"{output_prefix}_{site}")
    
    # Create aggregate plot
    if "All_Sites" in results:
        create_single_site_plot(results, "All_Sites", f"{output_prefix}_aggregate")
    
    # Create comparison plot across all sites
    create_comparison_plot(results, sites, f"{output_prefix}_comparison")


def create_single_site_plot(results, site, filename):
    """Create visualization plots for a single site."""
    if site not in results or not results[site]:
        return
    
    site_results = results[site]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'XGBoost Spectral Analysis - {site}', fontsize=16, fontweight='bold')
    
    # 1. Power Spectral Density
    ax = axes[0, 0]
    if 'psd' in site_results:
        psd_data = site_results['psd']
        ax.loglog(psd_data['freqs'][1:], psd_data['psd_actual'][1:], 'b-', label='Actual', alpha=0.8, linewidth=2)
        ax.loglog(psd_data['freqs'][1:], psd_data['psd_predicted'][1:], 'r-', label='XGBoost', alpha=0.8, linewidth=2)
        ax.set_xlabel('Frequency (1/weeks)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Coherence
    ax = axes[0, 1]
    if 'coherence' in site_results:
        coh_data = site_results['coherence']
        periods = 1/coh_data['freqs'][1:]
        ax.plot(periods, coh_data['coherence'][1:], 'g-', linewidth=2)
        ax.set_xlabel('Period (weeks)')
        ax.set_ylabel('Coherence')
        ax.set_title('Coherence between Actual and XGBoost')
        ax.set_xscale('log')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Significance threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # 3. Phase
    ax = axes[0, 2]
    if 'coherence' in site_results:
        coh_data = site_results['coherence']
        periods = 1/coh_data['freqs'][1:]
        ax.plot(periods, np.rad2deg(coh_data['phase'][1:]), 'purple', linewidth=2)
        ax.set_xlabel('Period (weeks)')
        ax.set_ylabel('Phase (degrees)')
        ax.set_title('Phase Difference')
        ax.set_xscale('log')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    # 4. Coherence by frequency band
    ax = axes[1, 0]
    if 'coherence' in site_results:
        bands = ['Low\n(>26 weeks)', 'Mid\n(4-26 weeks)', 'High\n(<4 weeks)']
        coherence_values = [
            site_results['coherence']['band_coherence']['low'],
            site_results['coherence']['band_coherence']['mid'],
            site_results['coherence']['band_coherence']['high']
        ]
        colors = ['blue', 'green', 'red']
        bars = ax.bar(bands, coherence_values, color=colors, alpha=0.7)
        ax.set_ylabel('Average Coherence')
        ax.set_title('Coherence by Frequency Band')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, coherence_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # 5. Spectral correlation over time
    ax = axes[1, 1]
    if 'spectrogram' in site_results:
        spec_data = site_results['spectrogram']
        ax.plot(spec_data['times'], spec_data['spectral_correlation'], 'orange', linewidth=2)
        ax.fill_between(spec_data['times'], spec_data['spectral_correlation'], 
                        alpha=0.3, color='orange')
        ax.set_xlabel('Time (weeks)')
        ax.set_ylabel('Spectral Correlation')
        ax.set_title('Time-Varying Spectral Correlation')
        ax.axhline(y=spec_data['avg_spectral_corr'], 
                  color='r', linestyle='--', alpha=0.5, 
                  label=f'Mean: {spec_data["avg_spectral_corr"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"SUMMARY FOR {site.upper()}\n\n"
    
    if 'psd' in site_results:
        summary_text += "Dominant Periods:\n"
        for period in site_results['psd']['dominant_periods'][:3]:
            summary_text += f"  • {period:.1f} weeks\n"
        summary_text += f"  • Power ratio: {site_results['psd']['power_ratio']:.3f}\n"
    
    if 'coherence' in site_results:
        coherence_values = [
            site_results['coherence']['band_coherence']['low'],
            site_results['coherence']['band_coherence']['mid'],
            site_results['coherence']['band_coherence']['high']
        ]
        best_band = ['Low', 'Mid', 'High'][np.argmax(coherence_values)]
        
        summary_text += f"\nCoherence Analysis:\n"
        summary_text += f"  • Best at: {best_band} frequencies\n"
        summary_text += f"  • Overall avg: {np.mean(site_results['coherence']['coherence']):.3f}\n"
        
        if not np.isnan(site_results['coherence']['avg_phase_lag']):
            summary_text += f"\nPhase Relationship:\n"
            summary_text += f"  • Avg lag: {site_results['coherence']['avg_phase_lag']:.1f}°\n"
            if site_results['coherence']['avg_phase_lag'] > 0:
                summary_text += "  • XGBoost lags actual\n"
            else:
                summary_text += "  • XGBoost leads actual\n"
    
    if 'performance' in site_results:
        perf = site_results['performance']
        summary_text += f"\nXGBoost Performance:\n"
        summary_text += f"  • R²: {perf['r2']:.4f}\n"
        summary_text += f"  • MAE: {perf['mae']:.4f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{filename}.png'")
    plt.show()


def create_comparison_plot(results, sites, filename):
    """Create comparison plot across all sites."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XGBoost Spectral Analysis - Site Comparison', fontsize=16, fontweight='bold')
    
    # Collect data for comparison
    site_names = []
    r2_scores = []
    coherence_low = []
    coherence_mid = []
    coherence_high = []
    power_ratios = []
    
    for site in sites:
        if site in results and results[site]:
            site_names.append(site)
            if 'performance' in results[site]:
                r2_scores.append(results[site]['performance']['r2'])
            else:
                r2_scores.append(0)
                
            if 'coherence' in results[site]:
                coherence_low.append(results[site]['coherence']['band_coherence']['low'])
                coherence_mid.append(results[site]['coherence']['band_coherence']['mid'])
                coherence_high.append(results[site]['coherence']['band_coherence']['high'])
            else:
                coherence_low.append(0)
                coherence_mid.append(0)
                coherence_high.append(0)
                
            if 'psd' in results[site]:
                power_ratios.append(results[site]['psd']['power_ratio'])
            else:
                power_ratios.append(0)
    
    # 1. R² scores by site
    ax = axes[0, 0]
    bars = ax.bar(range(len(site_names)), r2_scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('Site')
    ax.set_ylabel('R² Score')
    ax.set_title('XGBoost Performance by Site')
    ax.set_xticks(range(len(site_names)))
    ax.set_xticklabels(site_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2_scores[i]:.3f}', ha='center', va='bottom')
    
    # 2. Coherence by frequency band
    ax = axes[0, 1]
    x = np.arange(len(site_names))
    width = 0.25
    
    ax.bar(x - width, coherence_low, width, label='Low freq', alpha=0.7, color='blue')
    ax.bar(x, coherence_mid, width, label='Mid freq', alpha=0.7, color='green')
    ax.bar(x + width, coherence_high, width, label='High freq', alpha=0.7, color='red')
    
    ax.set_xlabel('Site')
    ax.set_ylabel('Average Coherence')
    ax.set_title('Coherence by Frequency Band')
    ax.set_xticks(x)
    ax.set_xticklabels(site_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Power ratios
    ax = axes[1, 0]
    bars = ax.bar(range(len(site_names)), power_ratios, color='orange', alpha=0.7)
    ax.set_xlabel('Site')
    ax.set_ylabel('Power Ratio (Pred/Actual)')
    ax.set_title('Spectral Power Ratio by Site')
    ax.set_xticks(range(len(site_names)))
    ax.set_xticklabels(site_names, rotation=45, ha='right')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect match')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{power_ratios[i]:.3f}', ha='center', va='bottom')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    if site_names:
        summary_text = f"XGBOOST SPECTRAL SUMMARY\n\n"
        summary_text += f"Sites Analyzed: {len(site_names)}\n\n"
        summary_text += f"Average R²: {np.mean(r2_scores):.4f}\n"
        summary_text += f"Best R² Site: {site_names[np.argmax(r2_scores)]}\n"
        summary_text += f"Best R² Score: {max(r2_scores):.4f}\n\n"
        
        summary_text += f"Average Coherence:\n"
        summary_text += f"  • Low freq: {np.mean(coherence_low):.3f}\n"
        summary_text += f"  • Mid freq: {np.mean(coherence_mid):.3f}\n"
        summary_text += f"  • High freq: {np.mean(coherence_high):.3f}\n\n"
        
        best_freq_band = ['Low', 'Mid', 'High'][np.argmax([np.mean(coherence_low), np.mean(coherence_mid), np.mean(coherence_high)])]
        summary_text += f"Best Frequency Band: {best_freq_band}\n\n"
        
        summary_text += f"Average Power Ratio: {np.mean(power_ratios):.3f}\n"
        if np.mean(power_ratios) < 1:
            summary_text += "XGBoost underestimates variability\n"
        else:
            summary_text += "XGBoost captures full variability\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as '{filename}.png'")
    plt.show()


def main():
    """Run complete XGBoost spectral analysis."""
    print("=" * 80)
    print("XGBOOST SPECTRAL ANALYSIS OF DA FORECASTING")
    print("=" * 80)
    
    # Get XGBoost predictions
    results_df = run_xgboost_retrospective()
    
    if results_df is None or results_df.empty:
        print("Failed to generate XGBoost predictions")
        return
    
    print(f"\nGenerated {len(results_df)} XGBoost predictions for analysis")
    print(f"Sites: {', '.join(results_df['site'].unique())}")
    
    # Perform spectral analysis
    spectral_results = analyze_xgboost_spectral_patterns(results_df)
    
    # Create comprehensive plots
    if spectral_results:
        print(f"\nCreating comprehensive visualizations...")
        create_comprehensive_spectral_plots(spectral_results, "xgboost_spectral")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL XGBOOST SPECTRAL INSIGHTS")
    print("=" * 80)
    
    if spectral_results:
        sites = [site for site in spectral_results.keys() if site != "All_Sites"]
        
        # Aggregate coherence analysis
        avg_coherences = {'low': [], 'mid': [], 'high': []}
        avg_r2 = []
        avg_power_ratios = []
        
        for site in sites:
            if 'coherence' in spectral_results[site]:
                for band in ['low', 'mid', 'high']:
                    avg_coherences[band].append(spectral_results[site]['coherence']['band_coherence'][band])
            
            if 'performance' in spectral_results[site]:
                avg_r2.append(spectral_results[site]['performance']['r2'])
                
            if 'psd' in spectral_results[site]:
                avg_power_ratios.append(spectral_results[site]['psd']['power_ratio'])
        
        print(f"\nAnalyzed {len(sites)} sites with XGBoost")
        print(f"Average XGBoost R²: {np.mean(avg_r2):.4f}")
        
        print("\nAverage coherence across all sites:")
        print(f"  Low frequency: {np.mean(avg_coherences['low']):.3f}")
        print(f"  Mid frequency: {np.mean(avg_coherences['mid']):.3f}")
        print(f"  High frequency: {np.mean(avg_coherences['high']):.3f}")
        
        best_band = max(avg_coherences, key=lambda x: np.mean(avg_coherences[x]))
        print(f"\n→ XGBoost performs best at {best_band} frequencies")
        
        if best_band == 'low':
            print("  XGBoost captures long-term trends and seasonal patterns well")
        elif best_band == 'mid':
            print("  XGBoost captures monthly to seasonal variations well")
        else:
            print("  XGBoost captures short-term fluctuations well")
        
        print(f"\nAverage spectral power ratio: {np.mean(avg_power_ratios):.3f}")
        if np.mean(avg_power_ratios) < 1:
            print("  → XGBoost tends to underestimate variability")
        else:
            print("  → XGBoost captures or overestimates variability")


if __name__ == "__main__":
    main()