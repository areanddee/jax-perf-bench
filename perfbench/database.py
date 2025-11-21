"""
Performance database with JSON storage

Manage benchmark results with query and export capabilities.
"""

import json
from pathlib import Path
from dataclasses import asdict
from typing import Any, List, Optional
from datetime import datetime

from .schema import (
    BenchmarkResult,
    HardwareConfig,
    NumericsConfig,
    ProblemConfig,
    BenchmarkConfig,
    PerformanceMetrics
)


class PerformanceDatabase:
    """Performance database with JSON storage"""
    
    def __init__(self, db_path: str = "performance_results.json"):
        self.db_path = Path(db_path)
        self.results: List[BenchmarkResult] = []
        
        # Load existing database if it exists
        if self.db_path.exists():
            self.load()
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the database"""
        self.results.append(result)
    
    def save(self, backup: bool = True):
        """Save database to JSON file"""
        if backup and self.db_path.exists():
            backup_path = self.db_path.with_suffix('.json.bak')
            self.db_path.rename(backup_path)
        
        # Convert to JSON-serializable format
        data = {
            'schema_version': '1.0.0',
            'last_updated': datetime.now().isoformat(),
            'n_results': len(self.results),
            'results': [self._result_to_dict(r) for r in self.results]
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load database from JSON file"""
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        self.results = [self._dict_to_result(r) for r in data['results']]
    
    def query(
        self,
        test_case: Optional[str] = None,
        method: Optional[str] = None,
        device_type: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Query database with filters"""
        filtered = self.results
        
        if test_case:
            filtered = [r for r in filtered if r.problem.test_case == test_case]
        
        if method:
            filtered = [r for r in filtered if r.numerics.method == method]
        
        if device_type:
            filtered = [r for r in filtered if r.hardware.device_type == device_type]
        
        if min_date:
            filtered = [r for r in filtered if r.timestamp >= min_date]
        
        if max_date:
            filtered = [r for r in filtered if r.timestamp <= max_date]
        
        # Additional filters from kwargs
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested attributes like 'numerics.tt_rank'
                parts = key.split('.')
                filtered = [r for r in filtered 
                           if self._get_nested_attr(r, parts) == value]
        
        return filtered
    
    def export_csv(self, output_path: str, results: Optional[List[BenchmarkResult]] = None):
        """Export results to CSV for analysis"""
        import csv
        
        results = results or self.results
        if not results:
            return
        
        # Flatten results for CSV
        rows = []
        for r in results:
            row = {
                'benchmark_id': r.benchmark_id,
                'timestamp': r.timestamp,
                'test_case': r.problem.test_case,
                'method': r.numerics.method,
                'device_type': r.hardware.device_type,
                'device_name': r.hardware.device_name,
                'n_horizontal': r.problem.n_horizontal,
                'n_vertical': r.problem.n_vertical,
                'batch_size': r.problem.batch_size,
                'tt_rank': r.numerics.tt_rank,
                'precision': r.numerics.precision,
                'mean_time': r.metrics.mean_time,
                'median_time': r.metrics.median_time,
                'std_time': r.metrics.std_time,
                'p50': r.metrics.percentiles.get('p50'),
                'p90': r.metrics.percentiles.get('p90'),
                'p95': r.metrics.percentiles.get('p95'),
                'throughput': r.metrics.throughput,
                'memory_allocated': r.metrics.memory_allocated,
            }
            rows.append(row)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    def _result_to_dict(self, result: BenchmarkResult) -> dict:
        """Convert BenchmarkResult to JSON-serializable dict"""
        return asdict(result)
    
    def _dict_to_result(self, data: dict) -> BenchmarkResult:
        """Convert dict to BenchmarkResult"""
        # Reconstruct nested dataclasses
        return BenchmarkResult(
            benchmark_id=data['benchmark_id'],
            timestamp=data['timestamp'],
            hardware=HardwareConfig(**data['hardware']),
            numerics=NumericsConfig(**data['numerics']),
            problem=ProblemConfig(**data['problem']),
            benchmark_config=BenchmarkConfig(**data['benchmark_config']),
            metrics=PerformanceMetrics(**data['metrics']),
            git_commit=data.get('git_commit'),
            jax_version=data.get('jax_version'),
            notes=data.get('notes', ''),
            schema_version=data.get('schema_version', '1.0.0'),
            extra=data.get('extra', {})
        )
    
    def _get_nested_attr(self, obj: Any, parts: List[str]) -> Any:
        """Get nested attribute by path like ['numerics', 'tt_rank']"""
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
