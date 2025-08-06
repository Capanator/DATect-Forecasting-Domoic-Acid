#!/usr/bin/env python3
"""
Spectral Analysis of Random Forest DA Forecasting
=================================================

Performs spectral analysis including:
- Power Spectral Density (PSD)
- Coherence between predicted and actual
- Phase relationships
- Wavelet analysis
- Cross-spectral analysis
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


def run_rf_retrospective():
    """Run RF retrospective evaluation to get predictions."""
    print("Running Random Forest retrospective evaluation...")
    
    # Set to RF
    original_model = config.FORECAST_MODEL
    config.FORECAST_MODEL = "rf"
    
    # Run evaluation with more anchors for better spectral analysis
    engine = ForecastEngine()
    results_df = engine.run_retrospective_evaluation(
        task="regression",
        model_type="rf",
        n_anchors=50  # More data for spectral analysis
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
    if method == 'welch':
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)//4))
    elif method == 'periodogram':
        freqs, psd = signal.periodogram(signal_data, fs=fs)
    else:  # multitaper
        from scipy.signal.windows import dpss
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


def wavelet_analysis(signal_data, wavelet='db4', levels=5):
    """
    Perform wavelet decomposition.
    
    Args:
        signal_data: Time series data
        wavelet: Wavelet type
        levels: Decomposition levels
    """
    if not HAS_PYWT:
        return None
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
    
    # Reconstruct at each level
    reconstructed = []
    for i in range(len(coeffs)):
        coeff_list = [np.zeros_like(c) if j != i else c for j, c in enumerate(coeffs)]
        reconstructed.append(pywt.waverec(coeff_list, wavelet))
    
    return coeffs, reconstructed


def analyze_temporal_patterns(results_df):
    """
    Analyze temporal patterns in predictions vs actual.
    """
    # Group by site for site-specific analysis
    sites = results_df['site'].unique()
    
    all_results = {}
    
    for site in sites[:3]:  # Analyze first 3 sites for brevity
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
        
        site_results = {}
        
        # 1. POWER SPECTRAL DENSITY
        print("\n1. Power Spectral Density Analysis")
        print("-" * 40)
        
        # Compute PSD for actual and predicted
        freqs, psd_actual = compute_power_spectral_density(actual)
        _, psd_predicted = compute_power_spectral_density(predicted)
        
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
            'dominant_periods': periods[dominant_idx]
        }
        
        # 2. COHERENCE AND PHASE
        print("\n2. Coherence and Phase Analysis")
        print("-" * 40)
        
        freqs_coh, coherence, phase = compute_coherence_phase(actual, predicted)
        
        # Average coherence in different frequency bands
        # Low frequency (> 26 weeks, semi-annual and longer)
        low_freq_mask = freqs_coh < 1/26
        low_freq_coherence = np.mean(coherence[low_freq_mask])
        
        # Mid frequency (4-26 weeks, monthly to semi-annual)
        mid_freq_mask = (freqs_coh >= 1/26) & (freqs_coh < 1/4)
        mid_freq_coherence = np.mean(coherence[mid_freq_mask])
        
        # High frequency (< 4 weeks, sub-monthly)
        high_freq_mask = freqs_coh >= 1/4
        high_freq_coherence = np.mean(coherence[high_freq_mask])
        
        print("Coherence by frequency band:")
        print(f"  Low freq (>26 weeks): {low_freq_coherence:.3f}")
        print(f"  Mid freq (4-26 weeks): {mid_freq_coherence:.3f}")
        print(f"  High freq (<4 weeks): {high_freq_coherence:.3f}")
        
        # Phase analysis
        phase_deg = np.rad2deg(phase)
        avg_phase_lag = np.mean(phase_deg[coherence > 0.5])  # Only where coherence is significant
        
        print(f"\nAverage phase lag (where coherence > 0.5): {avg_phase_lag:.1f}°")
        
        # Determine lead/lag relationship
        if avg_phase_lag > 0:
            print("  → Predictions LAG behind actual values")
        else:
            print("  → Predictions LEAD actual values")
        
        site_results['coherence'] = {
            'freqs': freqs_coh,
            'coherence': coherence,
            'phase': phase,
            'band_coherence': {
                'low': low_freq_coherence,
                'mid': mid_freq_coherence,
                'high': high_freq_coherence
            }
        }
        
        # 3. WAVELET ANALYSIS
        if HAS_PYWT:
            print("\n3. Wavelet Decomposition Analysis")
            print("-" * 40)
            
            # Perform wavelet decomposition
            coeffs_actual, _ = wavelet_analysis(actual, levels=4)
            coeffs_pred, _ = wavelet_analysis(predicted, levels=4)
            
            # Compare energy at each scale
            print("Energy distribution by scale:")
            for i, (ca, cp) in enumerate(zip(coeffs_actual, coeffs_pred)):
                if ca is not None and cp is not None:
                    energy_actual = np.sum(ca**2)
                    energy_pred = np.sum(cp**2)
                    if i == 0:
                        scale_name = "Approximation (lowest freq)"
                    else:
                        scale_name = f"Detail level {i} (~{2**i} week scale)"
                    
                    print(f"  {scale_name}:")
                    print(f"    Actual energy: {energy_actual:.2f}")
                    print(f"    Predicted energy: {energy_pred:.2f}")
                    print(f"    Ratio: {energy_pred/energy_actual:.3f}")
        
        # 4. TIME-VARYING SPECTRAL ANALYSIS
        print("\n4. Time-Varying Spectral Properties")
        print("-" * 40)
        
        # Compute spectrogram
        window_size = min(32, len(actual)//4)
        f, t, Sxx_actual = signal.spectrogram(actual, fs=1.0, nperseg=window_size)
        _, _, Sxx_pred = signal.spectrogram(predicted, fs=1.0, nperseg=window_size)
        
        # Compare spectral evolution
        spectral_correlation = []
        for i in range(Sxx_actual.shape[1]):
            corr, _ = pearsonr(Sxx_actual[:, i], Sxx_pred[:, i])
            spectral_correlation.append(corr)
        
        avg_spectral_corr = np.mean(spectral_correlation)
        print(f"Average spectral correlation over time: {avg_spectral_corr:.3f}")
        
        # Check if spectral properties change over time
        spectral_std = np.std(spectral_correlation)
        if spectral_std > 0.2:
            print("  → High variability: Model performance varies with time")
        else:
            print("  → Low variability: Consistent model performance over time")
        
        site_results['spectrogram'] = {
            'freqs': f,
            'times': t,
            'spectral_correlation': spectral_correlation
        }
        
        all_results[site] = site_results
    
    return all_results


def create_spectral_plots(results, site):
    """Create visualization plots for spectral analysis."""
    if site not in results:
        return
    
    site_results = results[site]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Spectral Analysis - {site}', fontsize=16)
    
    # 1. Power Spectral Density
    ax = axes[0, 0]
    psd_data = site_results['psd']
    ax.loglog(psd_data['freqs'][1:], psd_data['psd_actual'][1:], 'b-', label='Actual', alpha=0.7)
    ax.loglog(psd_data['freqs'][1:], psd_data['psd_predicted'][1:], 'r-', label='Predicted', alpha=0.7)
    ax.set_xlabel('Frequency (1/weeks)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Coherence
    ax = axes[0, 1]
    coh_data = site_results['coherence']
    ax.plot(1/coh_data['freqs'][1:], coh_data['coherence'][1:], 'g-')
    ax.set_xlabel('Period (weeks)')
    ax.set_ylabel('Coherence')
    ax.set_title('Coherence between Actual and Predicted')
    ax.set_xscale('log')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Significance threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Phase
    ax = axes[0, 2]
    ax.plot(1/coh_data['freqs'][1:], np.rad2deg(coh_data['phase'][1:]), 'purple')
    ax.set_xlabel('Period (weeks)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Phase Difference')
    ax.set_xscale('log')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 4. Coherence by frequency band
    ax = axes[1, 0]
    bands = ['Low\n(>26 weeks)', 'Mid\n(4-26 weeks)', 'High\n(<4 weeks)']
    coherence_values = [
        coh_data['band_coherence']['low'],
        coh_data['band_coherence']['mid'],
        coh_data['band_coherence']['high']
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
        ax.axhline(y=np.mean(spec_data['spectral_correlation']), 
                  color='r', linestyle='--', alpha=0.5, 
                  label=f'Mean: {np.mean(spec_data["spectral_correlation"]):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    SUMMARY FOR {site.upper()}
    
    Dominant Periods:
    """
    for period in psd_data['dominant_periods'][:3]:
        summary_text += f"\n  • {period:.1f} weeks"
    
    summary_text += f"""
    
    Coherence Analysis:
    • Best at: {'Low' if coh_data['band_coherence']['low'] == max(coherence_values) else 
                'Mid' if coh_data['band_coherence']['mid'] == max(coherence_values) else 'High'} frequencies
    • Overall avg: {np.mean(coh_data['coherence']):.3f}
    
    Phase Relationship:
    • Avg lag: {np.mean(np.rad2deg(coh_data['phase'][coh_data['coherence'] > 0.5])):.1f}°
    • {'Predictions lag' if np.mean(coh_data['phase'][coh_data['coherence'] > 0.5]) > 0 else 'Predictions lead'}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'spectral_analysis_{site}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'spectral_analysis_{site}.png'")
    plt.show()


def main():
    """Run complete spectral analysis."""
    print("="*60)
    print("SPECTRAL ANALYSIS OF RANDOM FOREST DA FORECASTING")
    print("="*60)
    
    # Get RF predictions
    results_df = run_rf_retrospective()
    
    if results_df is None or results_df.empty:
        print("Failed to generate predictions")
        return
    
    print(f"\nGenerated {len(results_df)} predictions for analysis")
    
    # Perform spectral analysis
    spectral_results = analyze_temporal_patterns(results_df)
    
    # Create plots for first site
    if spectral_results:
        first_site = list(spectral_results.keys())[0]
        print(f"\nCreating visualization for {first_site}...")
        create_spectral_plots(spectral_results, first_site)
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SPECTRAL INSIGHTS")
    print("="*60)
    
    if spectral_results:
        avg_coherences = {'low': [], 'mid': [], 'high': []}
        for site, results in spectral_results.items():
            for band in ['low', 'mid', 'high']:
                avg_coherences[band].append(results['coherence']['band_coherence'][band])
        
        print("\nAverage coherence across all sites:")
        print(f"  Low frequency: {np.mean(avg_coherences['low']):.3f}")
        print(f"  Mid frequency: {np.mean(avg_coherences['mid']):.3f}")
        print(f"  High frequency: {np.mean(avg_coherences['high']):.3f}")
        
        best_band = max(avg_coherences, key=lambda x: np.mean(avg_coherences[x]))
        print(f"\n→ Random Forest performs best at {best_band} frequencies")
        
        if best_band == 'low':
            print("  Model captures long-term trends and seasonal patterns well")
        elif best_band == 'mid':
            print("  Model captures monthly to seasonal variations well")
        else:
            print("  Model captures short-term fluctuations well")


if __name__ == "__main__":
    main()