#!/usr/bin/env python3
"""
Spike Detection Model Championship
==================================

Comprehensive testing of all spike detection models to find the best approach
for initial domoic acid spike timing prediction.

This script tests multiple model architectures and provides detailed comparison
against naive baselines with spike-focused metrics.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import warnings
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from forecasting.forecast_engine import ForecastEngine
from forecasting.spike_timing_optimizer import SpikeTimingOptimizer
from forecasting.data_processor import DataProcessor
from forecasting.logging_config import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class SpikeModelChampionship:
    """
    Comprehensive testing framework for spike detection models.
    """
    
    def __init__(self, spike_threshold=20.0, temporal_window=7):
        self.spike_threshold = spike_threshold
        self.temporal_window = temporal_window
        self.optimizer = SpikeTimingOptimizer(spike_threshold=spike_threshold)
        self.engine = ForecastEngine(validate_on_init=False)
        self.results = {}
        
        # Set spike optimization config
        config.SPIKE_THRESHOLD = spike_threshold
        config.SPIKE_WEIGHT_MULTIPLIER = 5.0
        
        logger.info(f"Championship initialized: spike_threshold={spike_threshold}, window={temporal_window}")
        
    def test_model(self, model_type, n_anchors=50, min_test_date="2010-01-01", quick_test=False):
        """
        Test a single model type.
        
        Args:
            model_type: Model type to test
            n_anchors: Number of anchor points for evaluation
            min_test_date: Earliest test date
            quick_test: If True, use reduced parameters for faster testing
        """
        print(f"\n🔬 Testing {model_type.upper()} model...")
        
        if quick_test:
            n_anchors = min(n_anchors, 20)  # Limit for quick testing
            min_test_date = "2015-01-01"  # More recent data only
        
        try:
            # Generate predictions
            results_df = self.engine.run_retrospective_evaluation(
                task="regression",
                model_type=model_type,
                n_anchors=n_anchors,
                min_test_date=min_test_date
            )
            
            if results_df is None or results_df.empty:
                print(f"   ❌ {model_type} failed to generate predictions")
                return None
            
            print(f"   ✅ Generated {len(results_df)} predictions")
            
            # Evaluate spike timing performance
            performance = self.optimizer.evaluate_spike_timing_performance(results_df)
            
            # Store results
            self.results[model_type] = {
                'predictions': results_df,
                'performance': performance,
                'n_predictions': len(results_df),
                'test_params': {
                    'n_anchors': n_anchors,
                    'min_test_date': min_test_date,
                    'quick_test': quick_test
                }
            }
            
            # Print quick summary
            if performance['spike_detection']:
                det = performance['spike_detection']
                print(f"   📊 F1: {det['f1_score']:.3f}, Precision: {det['precision']:.3f}, Recall: {det['recall']:.3f}")
                print(f"   🎯 Spikes: {performance['n_actual_spikes']} actual, {performance['n_predicted_spikes']} predicted")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error testing {model_type}: {e}")
            print(f"   ❌ {model_type} failed with error: {e}")
            return None
    
    def run_championship(self, models_to_test=None, n_anchors=50, quick_test=False, compare_baselines=True):
        """
        Run the complete championship across multiple models.
        
        Args:
            models_to_test: List of model types to test (None = test all)
            n_anchors: Number of anchor points per model
            quick_test: If True, use reduced parameters for faster testing
            compare_baselines: If True, include baseline comparisons
        """
        
        print("🏆 SPIKE DETECTION MODEL CHAMPIONSHIP")
        print("=" * 50)
        
        if models_to_test is None:
            models_to_test = [
                'xgboost',           # Original baseline
                'spike_xgboost',     # Weighted XGBoost
                'ensemble',          # Ensemble approach
                'rate_of_change',    # Rate-based detector
                'multi_horizon',     # Multi-step forecasting
                'anomaly',           # Anomaly detection
                'gradient',          # Gradient-based
                'linear'             # Simple baseline
            ]
        
        print(f"Testing {len(models_to_test)} models: {', '.join(models_to_test)}")
        print(f"Parameters: n_anchors={n_anchors}, quick_test={quick_test}")
        
        # Test each model
        successful_tests = 0
        for i, model_type in enumerate(models_to_test, 1):
            print(f"\n[{i}/{len(models_to_test)}] " + "=" * 40)
            
            result = self.test_model(model_type, n_anchors=n_anchors, quick_test=quick_test)
            if result is not None:
                successful_tests += 1
        
        print(f"\n✅ Successfully tested {successful_tests}/{len(models_to_test)} models")
        
        # Compare with baselines if requested
        if compare_baselines and successful_tests > 0:
            print("\n🔄 Generating baseline comparisons...")
            self._compare_with_baselines()
        
        # Generate championship report
        self._generate_championship_report()
        
        return self.results
    
    def _compare_with_baselines(self):
        """Compare all models with naive baselines."""
        
        # Load or generate baselines
        data_processor = DataProcessor()
        full_data = data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        
        print("   Generating naive baselines...")
        
        # Generate baselines
        naive_lag_baseline = self.optimizer.create_naive_lag_baseline(full_data, lag_days=7)
        persistence_baseline = self.optimizer.create_persistence_baseline(full_data)
        
        # Evaluate baselines
        naive_performance = self.optimizer.evaluate_spike_timing_performance(naive_lag_baseline)
        persistence_performance = self.optimizer.evaluate_spike_timing_performance(persistence_baseline)
        
        # Store baseline results
        self.results['naive_7d_lag'] = {
            'predictions': naive_lag_baseline,
            'performance': naive_performance,
            'n_predictions': len(naive_lag_baseline),
            'is_baseline': True
        }
        
        self.results['persistence'] = {
            'predictions': persistence_baseline,
            'performance': persistence_performance,
            'n_predictions': len(persistence_baseline),
            'is_baseline': True
        }
        
        print("   ✅ Baseline comparisons complete")
    
    def _generate_championship_report(self):
        """Generate comprehensive championship report."""
        
        if not self.results:
            print("❌ No results to report")
            return
        
        print("\n🏆 CHAMPIONSHIP RESULTS")
        print("=" * 60)
        
        # Create results table
        table_data = []
        
        for model_name, result in self.results.items():
            perf = result['performance']
            
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Type': 'Baseline' if result.get('is_baseline', False) else 'ML Model',
                'N_Predictions': result['n_predictions'],
                'Actual_Spikes': perf['n_actual_spikes'],
                'Predicted_Spikes': perf['n_predicted_spikes']
            }
            
            # Spike detection metrics
            if perf['spike_detection']:
                det = perf['spike_detection']
                row.update({
                    'Precision': det['precision'],
                    'Recall': det['recall'],
                    'F1_Score': det['f1_score'],
                    'True_Positives': det['true_positives'],
                    'False_Positives': det['false_positives'],
                    'False_Negatives': det['false_negatives']
                })
            
            # Spike magnitude metrics
            if perf['spike_magnitude']:
                mag = perf['spike_magnitude']
                row.update({
                    'Spike_MAE': mag.get('spike_mae', np.nan),
                    'Spike_RMSE': mag.get('spike_rmse', np.nan),
                    'Spike_Bias': mag.get('spike_bias', np.nan),
                    'Spike_R2': mag.get('spike_r2', np.nan)
                })
            
            # Overall performance
            if perf['overall_performance']:
                overall = perf['overall_performance']
                row.update({
                    'Overall_MAE': overall['overall_mae'],
                    'Overall_RMSE': overall['overall_rmse'],
                    'Overall_R2': overall['overall_r2']
                })
            
            table_data.append(row)
        
        # Convert to DataFrame and sort by F1 score
        df_results = pd.DataFrame(table_data)
        if 'F1_Score' in df_results.columns:
            df_results = df_results.sort_values('F1_Score', ascending=False)
        
        # Display top performers
        print("\n🥇 TOP SPIKE DETECTION PERFORMERS:")
        print("-" * 60)
        
        if 'F1_Score' in df_results.columns:
            top_5 = df_results.head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1]
                print(f"{medal} {row['Model']}: F1={row['F1_Score']:.3f}, "
                     f"Precision={row['Precision']:.3f}, Recall={row['Recall']:.3f}")
                
                if row['Type'] == 'Baseline':
                    print("   ⚠️  This is a NAIVE BASELINE - ML models should beat this!")
        
        # Find the best ML model
        ml_models = df_results[df_results['Type'] == 'ML Model']
        if not ml_models.empty and 'F1_Score' in ml_models.columns:
            best_ml = ml_models.iloc[0]
            print(f"\n🤖 BEST ML MODEL: {best_ml['Model']}")
            print(f"   F1: {best_ml['F1_Score']:.3f}")
            print(f"   Spike MAE: {best_ml.get('Spike_MAE', 'N/A'):.2f} ppm" if pd.notna(best_ml.get('Spike_MAE')) else "   Spike MAE: N/A")
            print(f"   Overall R²: {best_ml['Overall_R2']:.3f}")
        
        # Compare best ML vs best baseline
        baselines = df_results[df_results['Type'] == 'Baseline']
        if not baselines.empty and not ml_models.empty and 'F1_Score' in df_results.columns:
            best_baseline = baselines.iloc[0]
            best_ml = ml_models.iloc[0]
            
            improvement = best_ml['F1_Score'] - best_baseline['F1_Score']
            
            print(f"\n⚡ ML vs BASELINE COMPARISON:")
            print(f"   Best ML ({best_ml['Model']}): F1={best_ml['F1_Score']:.3f}")
            print(f"   Best Baseline ({best_baseline['Model']}): F1={best_baseline['F1_Score']:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
            
            if improvement > 0.05:
                print("   ✅ ML model shows significant improvement!")
            elif improvement > 0.02:
                print("   ⚠️  ML model shows modest improvement")
            else:
                print("   ❌ ML model does not outperform baseline significantly")
        
        # Save detailed results
        self._save_championship_results(df_results)
        
        print(f"\n📊 Full results saved to championship_results/")
    
    def _save_championship_results(self, df_results):
        """Save championship results to files."""
        
        # Create results directory
        results_dir = Path("championship_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV summary
        csv_path = results_dir / f"championship_summary_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)
        
        # Save detailed JSON results
        json_path = results_dir / f"championship_detailed_{timestamp}.json"
        
        # Convert results for JSON serialization
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = {
                'performance': result['performance'],
                'n_predictions': result['n_predictions'],
                'test_params': result.get('test_params', {}),
                'is_baseline': result.get('is_baseline', False)
            }
        
        # Add metadata
        json_results['_metadata'] = {
            'championship_date': datetime.now().isoformat(),
            'spike_threshold': self.spike_threshold,
            'temporal_window': self.temporal_window,
            'total_models_tested': len([k for k in self.results.keys() if not k.startswith('_')])
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(results_dir, timestamp, df_results)
        
        print(f"   💾 Results saved:")
        print(f"      CSV: {csv_path}")
        print(f"      JSON: {json_path}")
        print(f"      Report: championship_results/championship_report_{timestamp}.md")
    
    def _generate_markdown_report(self, results_dir, timestamp, df_results):
        """Generate a comprehensive markdown report."""
        
        report_path = results_dir / f"championship_report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"""# 🏆 Spike Detection Model Championship Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Spike Threshold:** {self.spike_threshold} ppm  
**Temporal Window:** {self.temporal_window} days  

## 🎯 Championship Overview

This report presents the results of a comprehensive comparison between multiple machine learning models
and naive baselines for domoic acid spike timing prediction.

### 🔬 Models Tested

""")
            
            # List all models tested
            ml_models = [k for k, v in self.results.items() if not v.get('is_baseline', False)]
            baselines = [k for k, v in self.results.items() if v.get('is_baseline', False)]
            
            f.write(f"**ML Models ({len(ml_models)}):** {', '.join(ml_models)}  \n")
            f.write(f"**Baselines ({len(baselines)}):** {', '.join(baselines)}  \n\n")
            
            # Performance table
            f.write("## 📊 Performance Summary\n\n")
            f.write("| Model | Type | F1 Score | Precision | Recall | Spike MAE | Overall R² |\n")
            f.write("|-------|------|----------|-----------|--------|-----------|------------|\n")
            
            for _, row in df_results.iterrows():
                f1 = row.get('F1_Score', 0)
                prec = row.get('Precision', 0) 
                recall = row.get('Recall', 0)
                spike_mae = row.get('Spike_MAE', np.nan)
                r2 = row.get('Overall_R2', 0)
                
                f.write(f"| {row['Model']} | {row['Type']} | {f1:.3f} | {prec:.3f} | {recall:.3f} | ")
                if pd.notna(spike_mae):
                    f.write(f"{spike_mae:.2f} | ")
                else:
                    f.write("N/A | ")
                f.write(f"{r2:.3f} |\n")
            
            # Analysis section
            f.write("\n## 🔍 Key Findings\n\n")
            
            if not df_results.empty and 'F1_Score' in df_results.columns:
                best_overall = df_results.iloc[0]
                f.write(f"### 🥇 Best Overall Performer: {best_overall['Model']}\n")
                f.write(f"- **F1 Score:** {best_overall['F1_Score']:.3f}\n")
                f.write(f"- **Precision:** {best_overall['Precision']:.3f}\n") 
                f.write(f"- **Recall:** {best_overall['Recall']:.3f}\n\n")
                
                # Compare ML vs baselines
                ml_models_df = df_results[df_results['Type'] == 'ML Model']
                baselines_df = df_results[df_results['Type'] == 'Baseline']
                
                if not ml_models_df.empty and not baselines_df.empty:
                    best_ml = ml_models_df.iloc[0]
                    best_baseline = baselines_df.iloc[0]
                    improvement = best_ml['F1_Score'] - best_baseline['F1_Score']
                    
                    f.write(f"### ⚡ ML vs Baseline Comparison\n")
                    f.write(f"- **Best ML Model:** {best_ml['Model']} (F1: {best_ml['F1_Score']:.3f})\n")
                    f.write(f"- **Best Baseline:** {best_baseline['Model']} (F1: {best_baseline['F1_Score']:.3f})\n")
                    f.write(f"- **Improvement:** {improvement:+.3f}\n\n")
                    
                    if improvement > 0.05:
                        f.write("✅ **Conclusion:** ML models show significant improvement over baselines.\n\n")
                    elif improvement > 0.02:
                        f.write("⚠️ **Conclusion:** ML models show modest improvement over baselines.\n\n")  
                    else:
                        f.write("❌ **Conclusion:** ML models do not significantly outperform baselines.\n\n")
            
            # Recommendations
            f.write("## 🎯 Recommendations\n\n")
            f.write("Based on the championship results:\n\n")
            f.write("1. **Production Model:** Consider the best-performing model for deployment\n")
            f.write("2. **Baseline Validation:** Always validate against naive baselines\n") 
            f.write("3. **Spike Focus:** Continue emphasizing spike timing over overall accuracy\n")
            f.write("4. **Further Development:** Investigate hybrid approaches combining top models\n\n")
            
            f.write("---\n")
            f.write("*Report generated by DATect Spike Model Championship*\n")


def main():
    """Main championship runner."""
    
    parser = argparse.ArgumentParser(description="Run spike detection model championship")
    parser.add_argument("--models", nargs='+', 
                       help="Specific models to test (default: all)")
    parser.add_argument("--anchors", type=int, default=50,
                       help="Number of anchor points per model")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (fewer anchors, recent data only)")
    parser.add_argument("--no-baselines", action="store_true", 
                       help="Skip baseline comparisons")
    parser.add_argument("--spike-threshold", type=float, default=20.0,
                       help="DA threshold for spike events (ppm)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    print("🏆 SPIKE DETECTION MODEL CHAMPIONSHIP")
    print(f"   Spike threshold: {args.spike_threshold} ppm")
    print(f"   Anchor points per model: {args.anchors}")
    print(f"   Quick test mode: {'Yes' if args.quick else 'No'}")
    print(f"   Include baselines: {'No' if args.no_baselines else 'Yes'}")
    
    if args.quick:
        print("\n⚡ Quick test mode: Using reduced parameters for faster testing")
        print("   - Limited to 20 anchor points maximum")
        print("   - Using only recent data (2015+)")
    
    # Initialize championship
    championship = SpikeModelChampionship(spike_threshold=args.spike_threshold)
    
    # Run championship
    try:
        results = championship.run_championship(
            models_to_test=args.models,
            n_anchors=args.anchors,
            quick_test=args.quick,
            compare_baselines=not args.no_baselines
        )
        
        print("\n🎉 Championship completed successfully!")
        print(f"   Tested {len(results)} model configurations")
        print("   Check championship_results/ for detailed analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️  Championship interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Championship failed: {e}")
        print(f"\n❌ Championship failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())