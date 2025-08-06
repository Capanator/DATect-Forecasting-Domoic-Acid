#!/usr/bin/env python3
"""
Advanced ACF/PACF Analysis for Lag Selection Justification
==========================================================

This module provides robust ACF/PACF analysis to scientifically justify
the choice of lags [1,2,3] in the DATect forecasting model.

Uses multiple approaches to handle statsmodels compatibility issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try different statsmodels approaches
try:
    from statsmodels.tsa.stattools import acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class AdvancedACFPACF:
    """
    Advanced ACF/PACF analysis with multiple fallback methods.
    """
    
    def __init__(self, save_plots=True, plot_dir="./acf_pacf_plots/"):
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        import os
        if self.save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    def manual_acf(self, ts, max_lags=20):
        """
        Manual ACF calculation as fallback.
        """
        n = len(ts)
        ts_centered = ts - np.mean(ts)
        
        acf_vals = np.zeros(max_lags + 1)
        acf_vals[0] = 1.0  # ACF at lag 0 is always 1
        
        # Calculate autocorrelations
        for lag in range(1, max_lags + 1):
            if lag < n:
                c_0 = np.sum(ts_centered ** 2) / n
                c_lag = np.sum(ts_centered[:-lag] * ts_centered[lag:]) / n
                acf_vals[lag] = c_lag / c_0 if c_0 != 0 else 0
            else:
                acf_vals[lag] = 0
                
        return acf_vals
    
    def manual_pacf(self, ts, max_lags=20):
        """
        Manual PACF calculation using Yule-Walker equations.
        """
        acf_vals = self.manual_acf(ts, max_lags)
        pacf_vals = np.zeros(max_lags + 1)
        pacf_vals[0] = 1.0
        
        if max_lags >= 1:
            pacf_vals[1] = acf_vals[1]
        
        # Calculate PACF using recursive formula
        for k in range(2, max_lags + 1):
            if k < len(acf_vals):
                # Build Toeplitz matrix
                R = np.array([[acf_vals[abs(i-j)] for j in range(k)] for i in range(k)])
                r = np.array([acf_vals[i] for i in range(1, k+1)])
                
                try:
                    phi = np.linalg.solve(R, r)
                    pacf_vals[k] = phi[-1]
                except np.linalg.LinAlgError:
                    pacf_vals[k] = 0
            else:
                pacf_vals[k] = 0
                
        return pacf_vals
    
    def robust_acf_analysis(self, data, target_col='da', site_col='site', max_lags=20):
        """
        Robust ACF/PACF analysis with multiple fallback methods.
        """
        print("🔍 Advanced ACF/PACF Analysis for Lag Selection Justification")
        print("=" * 70)
        
        results = {}
        sites = data[site_col].unique()
        
        # Create plots if requested
        if self.save_plots:
            n_sites = min(len(sites), 4)  # Limit to 4 sites for readability
            fig, axes = plt.subplots(n_sites, 2, figsize=(15, 4*n_sites))
            if n_sites == 1:
                axes = axes.reshape(1, -1)
        
        for i, site in enumerate(sites[:4] if self.save_plots else sites):
            site_data = data[data[site_col] == site].copy()
            site_data = site_data.sort_values('date').dropna(subset=[target_col])
            
            if len(site_data) < 20:
                print(f"[WARNING] Site {site}: Insufficient data ({len(site_data)} points)")
                continue
                
            ts = site_data[target_col].values
            n = len(ts)
            
            print(f"\n📊 Site: {site} (n={n} observations)")
            
            # Try statsmodels first, fall back to manual calculation
            try:
                if STATSMODELS_AVAILABLE:
                    # Method 1: Try with fft=True
                    try:
                        acf_vals = acf(ts, nlags=min(max_lags, n//4), fft=True)
                        pacf_vals = pacf(ts, nlags=min(max_lags, n//4), method='ols')
                        method_used = "statsmodels_fft"
                    except:
                        # Method 2: Try without fft
                        acf_vals = acf(ts, nlags=min(max_lags, n//4), fft=False)
                        pacf_vals = pacf(ts, nlags=min(max_lags, n//4), method='ols')
                        method_used = "statsmodels_standard"
                else:
                    raise ImportError("Statsmodels not available")
                    
            except Exception as e:
                print(f"   Statsmodels failed ({e}), using manual calculation")
                acf_vals = self.manual_acf(ts, min(max_lags, n//4))
                pacf_vals = self.manual_pacf(ts, min(max_lags, n//4))
                method_used = "manual_calculation"
            
            # Calculate confidence intervals
            ci_95 = 1.96 / np.sqrt(n)
            ci_99 = 2.58 / np.sqrt(n)
            
            # Find significant lags
            significant_lags_95 = []
            significant_lags_99 = []
            
            for lag in range(1, len(acf_vals)):
                if abs(acf_vals[lag]) > ci_95:
                    significant_lags_95.append(lag)
                if abs(acf_vals[lag]) > ci_99:
                    significant_lags_99.append(lag)
            
            # Check PACF significance
            pacf_significant_95 = []
            pacf_significant_99 = []
            
            for lag in range(1, len(pacf_vals)):
                if abs(pacf_vals[lag]) > ci_95:
                    pacf_significant_95.append(lag)
                if abs(pacf_vals[lag]) > ci_99:
                    pacf_significant_99.append(lag)
            
            # Store results
            results[site] = {
                'n_observations': n,
                'method_used': method_used,
                'acf': acf_vals.tolist() if hasattr(acf_vals, 'tolist') else list(acf_vals),
                'pacf': pacf_vals.tolist() if hasattr(pacf_vals, 'tolist') else list(pacf_vals),
                'ci_95': ci_95,
                'ci_99': ci_99,
                'significant_acf_lags_95': significant_lags_95[:10],
                'significant_acf_lags_99': significant_lags_99[:10],
                'significant_pacf_lags_95': pacf_significant_95[:10],
                'significant_pacf_lags_99': pacf_significant_99[:10]
            }
            
            # Analyze model lag justification [1,2,3]
            model_lags = [1, 2, 3]
            justified_acf_95 = [lag for lag in model_lags if lag in significant_lags_95]
            justified_acf_99 = [lag for lag in model_lags if lag in significant_lags_99]
            justified_pacf_95 = [lag for lag in model_lags if lag in pacf_significant_95]
            justified_pacf_99 = [lag for lag in model_lags if lag in pacf_significant_99]
            
            results[site]['model_lag_justification'] = {
                'acf_95_justified': justified_acf_95,
                'acf_99_justified': justified_acf_99,
                'pacf_95_justified': justified_pacf_95,
                'pacf_99_justified': justified_pacf_99
            }
            
            # Print analysis
            print(f"   Method: {method_used}")
            print(f"   Significant ACF lags (95%): {significant_lags_95[:10]}")
            print(f"   Significant PACF lags (95%): {pacf_significant_95[:10]}")
            print(f"   Model lags [1,2,3] justified by ACF (95%): {justified_acf_95}")
            print(f"   Model lags [1,2,3] justified by PACF (95%): {justified_pacf_95}")
            
            if justified_acf_95 or justified_pacf_95:
                justification_strength = len(set(justified_acf_95 + justified_pacf_95))
                print(f"   ✅ Lag justification strength: {justification_strength}/3 lags supported")
            else:
                print(f"   ⚠️  Model lags [1,2,3] not strongly supported at 95% confidence")
            
            # Plot if requested
            if self.save_plots and i < len(axes):
                try:
                    # ACF plot
                    lags_range = range(len(acf_vals))
                    axes[i, 0].plot(lags_range, acf_vals, 'b-', linewidth=2)
                    axes[i, 0].axhline(y=ci_95, color='red', linestyle='--', alpha=0.7, label='95% CI')
                    axes[i, 0].axhline(y=-ci_95, color='red', linestyle='--', alpha=0.7)
                    axes[i, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Highlight model lags
                    for lag in [1, 2, 3]:
                        if lag < len(acf_vals):
                            axes[i, 0].plot(lag, acf_vals[lag], 'ro', markersize=8, alpha=0.8)
                    
                    axes[i, 0].set_title(f'{site} - ACF')
                    axes[i, 0].set_xlabel('Lag')
                    axes[i, 0].set_ylabel('Autocorrelation')
                    axes[i, 0].grid(True, alpha=0.3)
                    axes[i, 0].legend()
                    
                    # PACF plot
                    axes[i, 1].plot(lags_range, pacf_vals, 'g-', linewidth=2)
                    axes[i, 1].axhline(y=ci_95, color='red', linestyle='--', alpha=0.7, label='95% CI')
                    axes[i, 1].axhline(y=-ci_95, color='red', linestyle='--', alpha=0.7)
                    axes[i, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Highlight model lags
                    for lag in [1, 2, 3]:
                        if lag < len(pacf_vals):
                            axes[i, 1].plot(lag, pacf_vals[lag], 'ro', markersize=8, alpha=0.8)
                    
                    axes[i, 1].set_title(f'{site} - PACF')
                    axes[i, 1].set_xlabel('Lag')
                    axes[i, 1].set_ylabel('Partial Autocorrelation')
                    axes[i, 1].grid(True, alpha=0.3)
                    axes[i, 1].legend()
                    
                except Exception as e:
                    print(f"   Plotting failed for {site}: {e}")
        
        # Save plots
        if self.save_plots:
            try:
                plt.tight_layout()
                plot_path = f"{self.plot_dir}/advanced_acf_pacf_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"\n📊 Plots saved: {plot_path}")
            except Exception as e:
                print(f"Plot saving failed: {e}")
                plt.close()
        
        return results
    
    def summarize_lag_justification(self, results):
        """
        Provide scientific summary of lag selection justification.
        """
        print(f"\n🎯 LAG SELECTION JUSTIFICATION SUMMARY")
        print("=" * 50)
        
        if not results:
            print("No results available for analysis")
            return
        
        total_sites = len(results)
        model_lags = [1, 2, 3]
        
        justification_summary = {lag: {'acf': 0, 'pacf': 0, 'either': 0} for lag in model_lags}
        
        for site, site_results in results.items():
            justification = site_results.get('model_lag_justification', {})
            
            for lag in model_lags:
                if lag in justification.get('acf_95_justified', []):
                    justification_summary[lag]['acf'] += 1
                    justification_summary[lag]['either'] += 1
                elif lag in justification.get('pacf_95_justified', []):
                    justification_summary[lag]['pacf'] += 1
                    justification_summary[lag]['either'] += 1
        
        print(f"Analysis across {total_sites} monitoring sites:")
        print()
        
        for lag in model_lags:
            acf_count = justification_summary[lag]['acf']
            pacf_count = justification_summary[lag]['pacf']
            either_count = justification_summary[lag]['either']
            
            acf_pct = (acf_count / total_sites) * 100
            pacf_pct = (pacf_count / total_sites) * 100
            either_pct = (either_count / total_sites) * 100
            
            print(f"Lag {lag}:")
            print(f"  - Justified by ACF:  {acf_count}/{total_sites} sites ({acf_pct:.1f}%)")
            print(f"  - Justified by PACF: {pacf_count}/{total_sites} sites ({pacf_pct:.1f}%)")
            print(f"  - Overall support:   {either_count}/{total_sites} sites ({either_pct:.1f}%)")
            
            if either_pct >= 50:
                print(f"  ✅ STRONG statistical justification")
            elif either_pct >= 25:
                print(f"  ⚠️  MODERATE statistical justification")
            else:
                print(f"  ❌ WEAK statistical justification")
            print()
        
        # Overall assessment
        total_justifications = sum(justification_summary[lag]['either'] for lag in model_lags)
        max_possible = total_sites * 3
        overall_pct = (total_justifications / max_possible) * 100
        
        print(f"Overall Assessment:")
        print(f"- Total lag justifications: {total_justifications}/{max_possible} ({overall_pct:.1f}%)")
        
        if overall_pct >= 60:
            print(f"✅ STRONG overall justification for lag selection [1,2,3]")
        elif overall_pct >= 40:
            print(f"⚠️  MODERATE overall justification for lag selection [1,2,3]")
        else:
            print(f"❌ WEAK overall justification - consider alternative lag selection")


def main():
    """Run advanced ACF/PACF analysis."""
    try:
        # Load data
        print("📊 Loading DATect environmental data...")
        data = pd.read_parquet('data/processed/final_output.parquet')
        print(f"Loaded {len(data)} records from {data['site'].nunique()} sites")
        
        # Initialize analyzer
        analyzer = AdvancedACFPACF(save_plots=True)
        
        # Run analysis on subset for speed
        sample_data = data.groupby('site').apply(lambda x: x.head(500)).reset_index(drop=True)
        print(f"Analyzing subset: {len(sample_data)} records")
        
        # Perform ACF/PACF analysis
        results = analyzer.robust_acf_analysis(sample_data, max_lags=15)
        
        # Summarize justification
        analyzer.summarize_lag_justification(results)
        
        print(f"\n🎉 Advanced ACF/PACF analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()