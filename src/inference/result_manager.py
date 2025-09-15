"""
Prediction result management system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, date, timedelta
import json
import sqlite3
from contextlib import contextmanager

from utils.logger import get_logger
from utils.config import get_config


class PredictionResultManager:
    """Manages storage, retrieval and analysis of prediction results"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize result manager"""
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'results.{key}', value)
        
        self.logger = get_logger("result_manager")
        
        # Storage paths
        self.results_dir = self.config.get_data_dir('predictions')
        self.db_path = self.results_dir / "predictions.db"
        self.csv_dir = self.results_dir / "daily_csv"
        self.reports_dir = self.results_dir / "reports"
        
        # Create directories
        for directory in [self.results_dir, self.csv_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        self._init_database()
        
        # Configuration
        self.max_csv_retention_days = self.config.get('results.csv_retention_days', 365)
        self.enable_csv_backup = self.config.get('results.enable_csv_backup', True)
        
    def _init_database(self) -> None:
        """Initialize SQLite database for storing results"""
        with self._get_db_connection() as conn:
            # Predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date DATE NOT NULL,
                    stock_code TEXT NOT NULL,
                    prediction_probability REAL NOT NULL,
                    prediction_binary INTEGER NOT NULL,
                    current_price REAL,
                    current_volume INTEGER,
                    model_version TEXT,
                    extraction_rank INTEGER,
                    threshold_used REAL,
                    confidence_category TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(prediction_date, stock_code)
                )
            """)
            
            # Model predictions (individual models)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    model_name TEXT NOT NULL,
                    probability REAL NOT NULL,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
                )
            """)
            
            # Extraction sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date DATE NOT NULL,
                    total_stocks_analyzed INTEGER,
                    stocks_extracted INTEGER,
                    effective_threshold REAL,
                    threshold_adjustment REAL,
                    market_volatility REAL,
                    session_metadata TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_date)
                )
            """)
            
            # Performance tracking (for backtesting)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date DATE NOT NULL,
                    stock_code TEXT NOT NULL,
                    prediction_probability REAL NOT NULL,
                    actual_next_day_return REAL,
                    actual_high_return REAL,
                    target_achieved INTEGER,
                    days_to_target INTEGER,
                    max_return_achieved REAL,
                    evaluation_date DATE,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(prediction_date, stock_code) 
                        REFERENCES predictions(prediction_date, stock_code)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_code ON predictions(stock_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON prediction_performance(prediction_date)")
            
            conn.commit()
        
        self.logger.info("Database initialized successfully")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def save_predictions(self, 
                        predictions: pd.DataFrame,
                        extraction_info: Dict[str, Any],
                        individual_predictions: Optional[pd.DataFrame] = None) -> int:
        """
        Save predictions to database and CSV
        
        Args:
            predictions: Main predictions DataFrame
            extraction_info: Extraction session information
            individual_predictions: Individual model predictions
            
        Returns:
            Session ID
        """
        prediction_date = predictions['Date'].iloc[0] if not predictions.empty else date.today()
        if isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date).date()
        elif isinstance(prediction_date, pd.Timestamp):
            prediction_date = prediction_date.date()
        
        self.logger.info("Saving predictions to database",
                        prediction_date=prediction_date,
                        predictions_count=len(predictions))
        
        session_id = None
        
        with self._get_db_connection() as conn:
            # Save extraction session
            try:
                session_metadata = json.dumps(extraction_info)
                
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO extraction_sessions 
                    (session_date, total_stocks_analyzed, stocks_extracted, 
                     effective_threshold, threshold_adjustment, session_metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    prediction_date,
                    extraction_info.get('total_stocks_analyzed', 0),
                    extraction_info.get('final_selections', 0),
                    extraction_info.get('effective_threshold', 0.85),
                    extraction_info.get('threshold_adjustment', 0.0),
                    session_metadata
                ))
                
                session_id = cursor.lastrowid
                
            except Exception as e:
                self.logger.error(f"Failed to save extraction session: {e}")
                raise
            
            # Save main predictions
            for _, row in predictions.iterrows():
                try:
                    cursor = conn.execute("""
                        INSERT OR REPLACE INTO predictions
                        (prediction_date, stock_code, prediction_probability, prediction_binary,
                         current_price, current_volume, extraction_rank, threshold_used, confidence_category)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prediction_date,
                        row['Code'],
                        row['prediction_probability'],
                        row.get('prediction_binary', 0),
                        row.get('current_close'),
                        row.get('current_volume'),
                        row.get('selection_rank'),
                        row.get('extraction_threshold'),
                        row.get('confidence_category')
                    ))
                    
                    prediction_id = cursor.lastrowid
                    
                    # Save individual model predictions if available
                    if individual_predictions is not None:
                        stock_individual = individual_predictions[
                            individual_predictions.index == row.name
                        ]
                        
                        for model_name in stock_individual.columns:
                            model_prob = stock_individual[model_name].iloc[0]
                            conn.execute("""
                                INSERT INTO model_predictions 
                                (prediction_id, model_name, probability)
                                VALUES (?, ?, ?)
                            """, (prediction_id, model_name, model_prob))
                    
                except Exception as e:
                    self.logger.error(f"Failed to save prediction for {row['Code']}: {e}")
                    continue
            
            conn.commit()
        
        # Save CSV backup if enabled
        if self.enable_csv_backup:
            self._save_csv_backup(predictions, prediction_date)
        
        self.logger.info("Predictions saved successfully", session_id=session_id)
        
        return session_id
    
    def _save_csv_backup(self, predictions: pd.DataFrame, prediction_date: date) -> None:
        """Save CSV backup of predictions"""
        try:
            date_str = prediction_date.strftime('%Y%m%d')
            csv_file = self.csv_dir / f"predictions_{date_str}.csv"
            
            predictions.to_csv(csv_file, index=False)
            
            # Clean up old CSV files
            self._cleanup_old_csv_files()
            
            self.logger.debug(f"CSV backup saved: {csv_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save CSV backup: {e}")
    
    def _cleanup_old_csv_files(self) -> None:
        """Remove CSV files older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.max_csv_retention_days)
        
        for csv_file in self.csv_dir.glob("predictions_*.csv"):
            try:
                # Extract date from filename
                date_str = csv_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    csv_file.unlink()
                    self.logger.debug(f"Removed old CSV file: {csv_file}")
                    
            except (ValueError, IndexError):
                # Skip files that don't match expected pattern
                continue
    
    def get_predictions(self, 
                       start_date: Optional[Union[str, date]] = None,
                       end_date: Optional[Union[str, date]] = None,
                       stock_codes: Optional[List[str]] = None,
                       min_probability: Optional[float] = None) -> pd.DataFrame:
        """
        Retrieve predictions from database
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering  
            stock_codes: List of stock codes to filter
            min_probability: Minimum probability threshold
            
        Returns:
            Predictions DataFrame
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if start_date is not None:
            query += " AND prediction_date >= ?"
            params.append(start_date)
            
        if end_date is not None:
            query += " AND prediction_date <= ?"
            params.append(end_date)
            
        if stock_codes is not None:
            placeholders = ','.join('?' * len(stock_codes))
            query += f" AND stock_code IN ({placeholders})"
            params.extend(stock_codes)
            
        if min_probability is not None:
            query += " AND prediction_probability >= ?"
            params.append(min_probability)
            
        query += " ORDER BY prediction_date DESC, prediction_probability DESC"
        
        with self._get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        self.logger.info(f"Retrieved {len(df)} predictions from database")
        
        return df
    
    def get_daily_summary(self, target_date: Union[str, date]) -> Dict[str, Any]:
        """Get daily prediction summary"""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        
        with self._get_db_connection() as conn:
            # Get basic prediction stats
            pred_query = """
                SELECT COUNT(*) as total_predictions,
                       AVG(prediction_probability) as mean_probability,
                       MIN(prediction_probability) as min_probability,
                       MAX(prediction_probability) as max_probability,
                       COUNT(CASE WHEN extraction_rank IS NOT NULL THEN 1 END) as extracted_count
                FROM predictions 
                WHERE prediction_date = ?
            """
            
            pred_stats = conn.execute(pred_query, (target_date,)).fetchone()
            
            # Get extraction session info
            session_query = """
                SELECT * FROM extraction_sessions 
                WHERE session_date = ?
            """
            
            session_info = conn.execute(session_query, (target_date,)).fetchone()
            
            # Get top predictions
            top_query = """
                SELECT stock_code, prediction_probability, confidence_category
                FROM predictions 
                WHERE prediction_date = ?
                ORDER BY prediction_probability DESC
                LIMIT 10
            """
            
            top_predictions = conn.execute(top_query, (target_date,)).fetchall()
        
        summary = {
            'date': target_date,
            'total_predictions': pred_stats['total_predictions'] if pred_stats else 0,
            'extracted_count': pred_stats['extracted_count'] if pred_stats else 0,
            'mean_probability': pred_stats['mean_probability'] if pred_stats else 0,
            'probability_range': {
                'min': pred_stats['min_probability'] if pred_stats else 0,
                'max': pred_stats['max_probability'] if pred_stats else 0
            },
            'top_predictions': [dict(row) for row in top_predictions] if top_predictions else [],
            'session_info': dict(session_info) if session_info else {}
        }
        
        return summary
    
    def generate_performance_report(self, 
                                  start_date: Union[str, date],
                                  end_date: Union[str, date]) -> Dict[str, Any]:
        """Generate performance report for a date range"""
        predictions_df = self.get_predictions(start_date, end_date)
        
        if predictions_df.empty:
            return {'error': 'No predictions found for the specified period'}
        
        # Basic statistics
        total_predictions = len(predictions_df)
        total_extracted = predictions_df['extraction_rank'].notna().sum()
        unique_dates = predictions_df['prediction_date'].nunique()
        unique_stocks = predictions_df['stock_code'].nunique()
        
        # Probability distribution
        prob_stats = {
            'mean': predictions_df['prediction_probability'].mean(),
            'median': predictions_df['prediction_probability'].median(),
            'std': predictions_df['prediction_probability'].std(),
            'min': predictions_df['prediction_probability'].min(),
            'max': predictions_df['prediction_probability'].max()
        }
        
        # Daily extraction rates
        daily_extraction = predictions_df.groupby('prediction_date').agg({
            'stock_code': 'count',
            'extraction_rank': lambda x: x.notna().sum()
        }).rename(columns={'stock_code': 'total_analyzed', 'extraction_rank': 'extracted'})
        
        daily_extraction['extraction_rate'] = daily_extraction['extracted'] / daily_extraction['total_analyzed']
        
        # Top performing stocks (by frequency of selection)
        stock_selection_freq = predictions_df[predictions_df['extraction_rank'].notna()]['stock_code'].value_counts()
        
        # Confidence category breakdown
        confidence_breakdown = predictions_df['confidence_category'].value_counts().to_dict()
        
        report = {
            'period': {'start_date': start_date, 'end_date': end_date},
            'summary_stats': {
                'total_predictions': total_predictions,
                'total_extracted': total_extracted,
                'unique_dates': unique_dates,
                'unique_stocks': unique_stocks,
                'extraction_rate': total_extracted / total_predictions if total_predictions > 0 else 0
            },
            'probability_statistics': prob_stats,
            'daily_extraction_stats': {
                'mean_daily_analyzed': daily_extraction['total_analyzed'].mean(),
                'mean_daily_extracted': daily_extraction['extracted'].mean(),
                'mean_extraction_rate': daily_extraction['extraction_rate'].mean(),
                'extraction_rate_std': daily_extraction['extraction_rate'].std()
            },
            'top_selected_stocks': stock_selection_freq.head(10).to_dict(),
            'confidence_breakdown': confidence_breakdown
        }
        
        return report
    
    def export_detailed_report(self, 
                             start_date: Union[str, date],
                             end_date: Union[str, date],
                             output_path: Optional[str] = None) -> str:
        """Export detailed performance report to file"""
        report = self.generate_performance_report(start_date, end_date)
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.reports_dir / f"performance_report_{timestamp}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("PREDICTION PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Period
            f.write(f"Period: {report['period']['start_date']} to {report['period']['end_date']}\n\n")
            
            # Summary
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            summary = report['summary_stats']
            f.write(f"Total Predictions: {summary['total_predictions']:,}\n")
            f.write(f"Total Extracted: {summary['total_extracted']:,}\n")
            f.write(f"Overall Extraction Rate: {summary['extraction_rate']:.2%}\n")
            f.write(f"Trading Days: {summary['unique_dates']}\n")
            f.write(f"Unique Stocks Analyzed: {summary['unique_stocks']}\n\n")
            
            # Probability statistics
            f.write("PROBABILITY DISTRIBUTION\n")
            f.write("-" * 25 + "\n")
            prob_stats = report['probability_statistics']
            f.write(f"Mean Probability: {prob_stats['mean']:.3f}\n")
            f.write(f"Median Probability: {prob_stats['median']:.3f}\n")
            f.write(f"Std Deviation: {prob_stats['std']:.3f}\n")
            f.write(f"Range: {prob_stats['min']:.3f} - {prob_stats['max']:.3f}\n\n")
            
            # Daily statistics
            f.write("DAILY EXTRACTION STATISTICS\n")
            f.write("-" * 30 + "\n")
            daily_stats = report['daily_extraction_stats']
            f.write(f"Avg Daily Stocks Analyzed: {daily_stats['mean_daily_analyzed']:.1f}\n")
            f.write(f"Avg Daily Stocks Extracted: {daily_stats['mean_daily_extracted']:.1f}\n")
            f.write(f"Mean Daily Extraction Rate: {daily_stats['mean_extraction_rate']:.2%}\n\n")
            
            # Top selected stocks
            f.write("MOST FREQUENTLY SELECTED STOCKS\n")
            f.write("-" * 35 + "\n")
            for stock, count in report['top_selected_stocks'].items():
                f.write(f"{stock}: {count} times\n")
            f.write("\n")
            
            # Confidence breakdown
            if 'confidence_breakdown' in report and report['confidence_breakdown']:
                f.write("CONFIDENCE CATEGORY BREAKDOWN\n")
                f.write("-" * 30 + "\n")
                for category, count in report['confidence_breakdown'].items():
                    f.write(f"{category}: {count}\n")
        
        self.logger.info(f"Detailed report exported to {output_path}")
        
        return str(output_path)
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """Clean up old prediction data"""
        cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
        
        deleted_counts = {}
        
        with self._get_db_connection() as conn:
            # Delete old model predictions first (foreign key constraint)
            model_pred_result = conn.execute("""
                DELETE FROM model_predictions 
                WHERE prediction_id IN (
                    SELECT id FROM predictions WHERE prediction_date < ?
                )
            """, (cutoff_date,))
            deleted_counts['model_predictions'] = model_pred_result.rowcount
            
            # Delete old predictions
            pred_result = conn.execute("""
                DELETE FROM predictions WHERE prediction_date < ?
            """, (cutoff_date,))
            deleted_counts['predictions'] = pred_result.rowcount
            
            # Delete old sessions
            session_result = conn.execute("""
                DELETE FROM extraction_sessions WHERE session_date < ?
            """, (cutoff_date,))
            deleted_counts['sessions'] = session_result.rowcount
            
            # Delete old performance data
            perf_result = conn.execute("""
                DELETE FROM prediction_performance WHERE prediction_date < ?
            """, (cutoff_date,))
            deleted_counts['performance'] = perf_result.rowcount
            
            conn.commit()
        
        self.logger.info(f"Cleaned up old data before {cutoff_date}",
                        deleted_counts=deleted_counts)
        
        return deleted_counts