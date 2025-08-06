"""
Automated Data Quality Reporter
===============================

Generates comprehensive, automated data quality reports with visualizations,
trend analysis, and actionable recommendations for the DATect system.

This module provides:
- Automated quality report generation
- Quality trend analysis and monitoring
- Visual quality dashboards and charts
- Quality alert system with thresholds
- Exportable reports in multiple formats
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from .logging_config import get_logger
from .data_quality import DataValidator, QualityReport, QualityLevel
from .data_freshness import DataFreshnessValidator, validate_pipeline_freshness
from .retry_mechanisms import get_retry_statistics
import config

logger = get_logger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend('Agg')


@dataclass
class QualityTrend:
    """Quality trend analysis result."""
    metric_name: str
    current_value: float
    previous_value: float
    change_percentage: float
    trend_direction: str  # 'improving', 'declining', 'stable'
    significance: str     # 'major', 'minor', 'negligible'


@dataclass
class QualityAlert:
    """Quality alert information."""
    alert_type: str      # 'critical', 'warning', 'info'
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: str
    recommendations: List[str]


class QualityReporter:
    """
    Automated data quality reporter with comprehensive analysis and visualization.
    """
    
    def __init__(self, report_dir: str = "./outputs/quality_reports/"):
        """
        Initialize quality reporter.
        
        Args:
            report_dir: Directory to store quality reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = DataValidator()
        self.freshness_validator = DataFreshnessValidator()
        
        # Quality thresholds for alerts
        self.alert_thresholds = {
            'overall_score': {'critical': 50, 'warning': 70},
            'freshness_score': {'critical': 40, 'warning': 60},
            'missing_data_pct': {'critical': 50, 'warning': 25},
            'duplicate_pct': {'critical': 10, 'warning': 5}
        }
        
        logger.info(f"Initialized QualityReporter with report directory: {report_dir}")
    
    def generate_comprehensive_report(self, datasets: Dict[str, pd.DataFrame], 
                                    report_name: str = "pipeline_quality") -> Dict[str, Any]:
        """
        Generate comprehensive quality report for multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            report_name: Name for the report
            
        Returns:
            Comprehensive quality report dictionary
        """
        logger.info(f"Generating comprehensive quality report: {report_name}")
        
        report_timestamp = datetime.now().isoformat()
        
        # Generate individual quality reports
        quality_reports = {}
        for name, df in datasets.items():
            try:
                quality_reports[name] = self.validator.validate_dataset(df, name)
                logger.debug(f"Generated quality report for {name}")
            except Exception as e:
                logger.error(f"Error generating quality report for {name}: {e}")
        
        # Generate freshness report
        freshness_report = validate_pipeline_freshness(datasets)
        
        # Generate retry statistics
        retry_stats = get_retry_statistics()
        
        # Analyze trends (if historical data available)
        trends = self._analyze_quality_trends(quality_reports)
        
        # Generate alerts
        alerts = self._generate_quality_alerts(quality_reports, freshness_report)
        
        # Create summary metrics
        summary = self._create_summary_metrics(quality_reports, freshness_report)
        
        # Generate recommendations
        recommendations = self._generate_consolidated_recommendations(
            quality_reports, freshness_report, alerts
        )
        
        # Create visualizations
        charts = self._create_quality_visualizations(quality_reports, freshness_report)
        
        # Compile comprehensive report
        comprehensive_report = {
            'report_metadata': {
                'report_name': report_name,
                'timestamp': report_timestamp,
                'datasets_analyzed': len(datasets),
                'total_records': sum(len(df) for df in datasets.values()),
                'report_version': '1.0'
            },
            'executive_summary': summary,
            'quality_scores': {
                name: {
                    'overall_score': report.overall_score,
                    'quality_level': report.quality_level.value,
                    'total_records': report.total_records
                }
                for name, report in quality_reports.items()
            },
            'freshness_analysis': freshness_report,
            'detailed_quality_reports': {
                name: report.to_dict() for name, report in quality_reports.items()
            },
            'trend_analysis': trends,
            'alerts': [asdict(alert) for alert in alerts],
            'retry_statistics': retry_stats,
            'recommendations': recommendations,
            'visualizations': charts
        }
        
        # Save report
        report_path = self._save_report(comprehensive_report, report_name)
        comprehensive_report['report_file_path'] = str(report_path)
        
        logger.info(f"Comprehensive quality report generated and saved to: {report_path}")
        
        return comprehensive_report
    
    def _analyze_quality_trends(self, quality_reports: Dict[str, QualityReport]) -> List[QualityTrend]:
        """Analyze quality trends by comparing with historical reports."""
        trends = []
        
        try:
            # Look for historical reports to compare
            for dataset_name, current_report in quality_reports.items():
                historical_report = self._load_most_recent_historical_report(dataset_name)
                
                if historical_report:
                    # Compare overall scores
                    current_score = current_report.overall_score
                    historical_score = historical_report.overall_score
                    
                    change_pct = ((current_score - historical_score) / historical_score * 100) if historical_score > 0 else 0
                    
                    # Determine trend direction and significance
                    if abs(change_pct) < 2:
                        direction = 'stable'
                        significance = 'negligible'
                    elif change_pct > 0:
                        direction = 'improving'
                        significance = 'major' if change_pct > 10 else 'minor'
                    else:
                        direction = 'declining'
                        significance = 'major' if change_pct < -10 else 'minor'
                    
                    trends.append(QualityTrend(
                        metric_name=f"{dataset_name}_overall_score",
                        current_value=current_score,
                        previous_value=historical_score,
                        change_percentage=change_pct,
                        trend_direction=direction,
                        significance=significance
                    ))
                    
                    logger.debug(f"Analyzed trend for {dataset_name}: {direction} ({change_pct:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error analyzing quality trends: {e}")
        
        return trends
    
    def _load_most_recent_historical_report(self, dataset_name: str) -> Optional[QualityReport]:
        """Load the most recent historical report for comparison."""
        try:
            # Look for recent report files (within last 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)
            
            report_files = list(self.report_dir.glob(f"*{dataset_name}*.json"))
            
            if not report_files:
                return None
            
            # Sort by modification time and get most recent
            report_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Skip the most recent file (likely current report) and get previous
            for report_file in report_files[1:]:
                try:
                    file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_time >= cutoff_date:
                        with open(report_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract quality report data
                        if 'detailed_quality_reports' in data and dataset_name in data['detailed_quality_reports']:
                            report_data = data['detailed_quality_reports'][dataset_name]
                            
                            # Reconstruct QualityReport (simplified)
                            return type('HistoricalReport', (), {
                                'overall_score': report_data.get('overall_score', 0),
                                'quality_level': report_data.get('quality_level', 'unknown')
                            })()
                            
                except Exception as e:
                    logger.debug(f"Error loading historical report from {report_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error loading historical report for {dataset_name}: {e}")
            return None
    
    def _generate_quality_alerts(self, quality_reports: Dict[str, QualityReport],
                               freshness_report: Dict[str, Any]) -> List[QualityAlert]:
        """Generate quality alerts based on thresholds."""
        alerts = []
        
        try:
            # Check overall quality scores
            for name, report in quality_reports.items():
                score = report.overall_score
                
                if score < self.alert_thresholds['overall_score']['critical']:
                    alerts.append(QualityAlert(
                        alert_type='critical',
                        metric=f'{name}_overall_quality',
                        current_value=score,
                        threshold=self.alert_thresholds['overall_score']['critical'],
                        message=f"Critical quality issue in {name}: {score:.1f}% quality score",
                        timestamp=datetime.now().isoformat(),
                        recommendations=[
                            f"Immediate review of {name} data required",
                            "Check data sources and processing pipeline",
                            "Consider data cleaning and validation steps"
                        ]
                    ))
                elif score < self.alert_thresholds['overall_score']['warning']:
                    alerts.append(QualityAlert(
                        alert_type='warning',
                        metric=f'{name}_overall_quality',
                        current_value=score,
                        threshold=self.alert_thresholds['overall_score']['warning'],
                        message=f"Quality warning for {name}: {score:.1f}% quality score",
                        timestamp=datetime.now().isoformat(),
                        recommendations=[
                            f"Monitor {name} data quality trends",
                            "Consider additional validation steps"
                        ]
                    ))
            
            # Check freshness scores
            if 'overall_score' in freshness_report:
                freshness_score = freshness_report['overall_score']
                
                if freshness_score < self.alert_thresholds['freshness_score']['critical']:
                    alerts.append(QualityAlert(
                        alert_type='critical',
                        metric='data_freshness',
                        current_value=freshness_score,
                        threshold=self.alert_thresholds['freshness_score']['critical'],
                        message=f"Critical freshness issue: {freshness_score:.1f}% freshness score",
                        timestamp=datetime.now().isoformat(),
                        recommendations=[
                            "Update data sources immediately",
                            "Check data pipeline scheduling",
                            "Verify external data source availability"
                        ]
                    ))
                elif freshness_score < self.alert_thresholds['freshness_score']['warning']:
                    alerts.append(QualityAlert(
                        alert_type='warning',
                        metric='data_freshness',
                        current_value=freshness_score,
                        threshold=self.alert_thresholds['freshness_score']['warning'],
                        message=f"Freshness warning: {freshness_score:.1f}% freshness score",
                        timestamp=datetime.now().isoformat(),
                        recommendations=[
                            "Schedule data refresh soon",
                            "Monitor freshness trends"
                        ]
                    ))
            
            logger.info(f"Generated {len(alerts)} quality alerts")
            
        except Exception as e:
            logger.error(f"Error generating quality alerts: {e}")
        
        return alerts
    
    def _create_summary_metrics(self, quality_reports: Dict[str, QualityReport],
                              freshness_report: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary metrics."""
        if not quality_reports:
            return {
                'status': 'No data to analyze',
                'overall_health': 'Unknown'
            }
        
        # Calculate aggregate metrics
        avg_quality_score = np.mean([report.overall_score for report in quality_reports.values()])
        
        quality_distribution = {
            'excellent': sum(1 for r in quality_reports.values() if r.quality_level == QualityLevel.EXCELLENT),
            'good': sum(1 for r in quality_reports.values() if r.quality_level == QualityLevel.GOOD),
            'acceptable': sum(1 for r in quality_reports.values() if r.quality_level == QualityLevel.ACCEPTABLE),
            'poor': sum(1 for r in quality_reports.values() if r.quality_level == QualityLevel.POOR),
            'critical': sum(1 for r in quality_reports.values() if r.quality_level == QualityLevel.CRITICAL)
        }
        
        # Determine overall health
        if avg_quality_score >= 85:
            overall_health = 'Excellent'
        elif avg_quality_score >= 70:
            overall_health = 'Good'
        elif avg_quality_score >= 50:
            overall_health = 'Acceptable'
        else:
            overall_health = 'Poor'
        
        # Count issues
        critical_issues = sum(1 for r in quality_reports.values() if r.quality_level == QualityLevel.CRITICAL)
        total_datasets = len(quality_reports)
        
        return {
            'overall_health': overall_health,
            'average_quality_score': round(avg_quality_score, 1),
            'freshness_score': freshness_report.get('overall_score', 0),
            'total_datasets_analyzed': total_datasets,
            'datasets_with_issues': critical_issues + quality_distribution['poor'],
            'quality_distribution': quality_distribution,
            'critical_issues_count': critical_issues,
            'key_findings': [
                f"Average quality score: {avg_quality_score:.1f}%",
                f"Data freshness: {freshness_report.get('overall_score', 0):.1f}%",
                f"Datasets needing attention: {critical_issues + quality_distribution['poor']}"
            ]
        }
    
    def _generate_consolidated_recommendations(self, quality_reports: Dict[str, QualityReport],
                                             freshness_report: Dict[str, Any],
                                             alerts: List[QualityAlert]) -> List[str]:
        """Generate consolidated recommendations across all reports."""
        recommendations = set()
        
        # Priority recommendations from alerts
        for alert in alerts:
            if alert.alert_type == 'critical':
                recommendations.update(alert.recommendations)
        
        # Add general recommendations
        avg_quality = np.mean([report.overall_score for report in quality_reports.values()])
        
        if avg_quality < 70:
            recommendations.add("Implement comprehensive data cleaning procedures")
            recommendations.add("Review and validate data sources")
        
        freshness_score = freshness_report.get('overall_score', 100)
        if freshness_score < 80:
            recommendations.add("Increase data update frequency")
            recommendations.add("Monitor data source availability")
        
        # Dataset-specific recommendations
        for name, report in quality_reports.items():
            if report.quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL]:
                recommendations.add(f"Urgent: Address data quality issues in {name}")
        
        # Add monitoring recommendations
        recommendations.add("Set up automated quality monitoring")
        recommendations.add("Establish quality trend alerts")
        recommendations.add("Regular quality report review")
        
        return sorted(list(recommendations))
    
    def _create_quality_visualizations(self, quality_reports: Dict[str, QualityReport],
                                     freshness_report: Dict[str, Any]) -> Dict[str, str]:
        """Create quality visualization charts as base64 encoded images."""
        charts = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Chart 1: Quality Score Overview
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Quality scores bar chart
            names = list(quality_reports.keys())
            scores = [report.overall_score for report in quality_reports.values()]
            
            bars = ax1.bar(names, scores, color=['red' if s < 50 else 'orange' if s < 70 else 'yellow' if s < 85 else 'green' for s in scores])
            ax1.set_title('Data Quality Scores by Dataset')
            ax1.set_ylabel('Quality Score (%)')
            ax1.set_ylim(0, 100)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.1f}%', ha='center', va='bottom')
            
            # Quality level distribution pie chart
            quality_levels = [report.quality_level.value for report in quality_reports.values()]
            level_counts = pd.Series(quality_levels).value_counts()
            
            colors = {'excellent': 'green', 'good': 'lightgreen', 'acceptable': 'yellow', 
                     'poor': 'orange', 'critical': 'red'}
            chart_colors = [colors.get(level, 'gray') for level in level_counts.index]
            
            ax2.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
                   colors=chart_colors, startangle=90)
            ax2.set_title('Quality Level Distribution')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            charts['quality_overview'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # Chart 2: Freshness Analysis
            if 'source_details' in freshness_report:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                source_details = freshness_report['source_details']
                sources = list(source_details.keys())
                freshness_scores = [details['score'] for details in source_details.values()]
                
                bars = ax.bar(sources, freshness_scores, 
                            color=['red' if s < 40 else 'orange' if s < 60 else 'yellow' if s < 80 else 'green' for s in freshness_scores])
                ax.set_title('Data Freshness Scores by Source')
                ax.set_ylabel('Freshness Score (%)')
                ax.set_ylim(0, 100)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add score labels
                for bar, score in zip(bars, freshness_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{score:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                charts['freshness_analysis'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            logger.debug(f"Created {len(charts)} quality visualizations")
            
        except Exception as e:
            logger.error(f"Error creating quality visualizations: {e}")
            charts['error'] = f"Visualization error: {str(e)}"
        
        return charts
    
    def _save_report(self, report: Dict[str, Any], report_name: str) -> Path:
        """Save comprehensive report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_report_{report_name}_{timestamp}.json"
        filepath = self.report_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Quality report saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise
    
    def generate_daily_report(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Generate daily automated quality report."""
        report_name = f"daily_{datetime.now().strftime('%Y%m%d')}"
        
        try:
            comprehensive_report = self.generate_comprehensive_report(datasets, report_name)
            
            # Check for critical alerts
            alerts = comprehensive_report.get('alerts', [])
            critical_alerts = [a for a in alerts if a.get('alert_type') == 'critical']
            
            if critical_alerts:
                logger.warning(f"Daily report generated with {len(critical_alerts)} critical alerts")
            else:
                logger.info("Daily quality report generated successfully")
            
            return comprehensive_report['report_file_path']
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            raise
    
    def export_report_html(self, report: Dict[str, Any]) -> str:
        """Export report as HTML for easy viewing."""
        try:
            html_content = self._generate_html_report(report)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"quality_report_{report['report_metadata']['report_name']}_{timestamp}.html"
            html_filepath = self.report_dir / html_filename
            
            with open(html_filepath, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report exported to: {html_filepath}")
            return str(html_filepath)
            
        except Exception as e:
            logger.error(f"Error exporting HTML report: {e}")
            raise
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for the report."""
        # Basic HTML template - would be expanded for production use
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DATect Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; }}
                .alert {{ background-color: #ffe6e6; border-left: 5px solid #ff0000; padding: 10px; margin: 10px 0; }}
                .warning {{ background-color: #fff8e1; border-left: 5px solid #ff9800; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DATect Data Quality Report</h1>
                <p>Generated: {report['report_metadata']['timestamp']}</p>
                <p>Report: {report['report_metadata']['report_name']}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Overall Health:</strong> {report['executive_summary']['overall_health']}</p>
                <p><strong>Average Quality Score:</strong> {report['executive_summary']['average_quality_score']}%</p>
                <p><strong>Datasets Analyzed:</strong> {report['executive_summary']['total_datasets_analyzed']}</p>
            </div>
            
            <h2>Quality Scores</h2>
            <table>
                <tr><th>Dataset</th><th>Quality Score</th><th>Quality Level</th><th>Records</th></tr>
                {''.join([
                    f"<tr><td>{name}</td><td>{data['overall_score']:.1f}%</td><td>{data['quality_level']}</td><td>{data['total_records']}</td></tr>"
                    for name, data in report['quality_scores'].items()
                ])}
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                {''.join([f"<li>{rec}</li>" for rec in report['recommendations']])}
            </ul>
            
        </body>
        </html>
        """
        
        return html_template