"""
Analysis tools for benchmark results

Query, compare, and analyze performance data.
"""

from typing import Any, Dict, List
from .schema import BenchmarkResult
from .database import PerformanceDatabase


class BenchmarkAnalyzer:
    """Tools for analyzing benchmark results"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
    
    def compare_methods(
        self,
        test_case: str,
        methods: List[str],
        metric: str = 'median_time'
    ) -> Dict[str, float]:
        """Compare different methods on the same test case"""
        results = {}
        for method in methods:
            method_results = self.db.query(test_case=test_case, method=method)
            if method_results:
                # Get most recent result
                latest = sorted(method_results, key=lambda r: r.timestamp)[-1]
                results[method] = getattr(latest.metrics, metric)
        
        return results
    
    def scaling_analysis(
        self,
        test_case: str,
        method: str,
        vary_param: str,  # e.g., 'batch_size' or 'numerics.tt_rank'
        metric: str = 'median_time'
    ) -> Dict[Any, float]:
        """Analyze how performance scales with a parameter"""
        all_results = self.db.query(test_case=test_case, method=method)
        
        scaling = {}
        for result in all_results:
            if '.' in vary_param:
                parts = vary_param.split('.')
                param_value = self.db._get_nested_attr(result, parts)
            else:
                param_value = getattr(result.problem, vary_param, None)
            
            if param_value is not None:
                metric_value = getattr(result.metrics, metric)
                scaling[param_value] = metric_value
        
        return dict(sorted(scaling.items()))
    
    def summary_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a summary report of benchmark results"""
        if not results:
            return "No results to summarize"
        
        report = []
        report.append(f"Summary of {len(results)} benchmark results")
        report.append("=" * 60)
        
        # Group by test case
        by_case = {}
        for r in results:
            case = r.problem.test_case
            if case not in by_case:
                by_case[case] = []
            by_case[case].append(r)
        
        for case, case_results in by_case.items():
            report.append(f"\nTest Case: {case}")
            report.append("-" * 60)
            
            for r in sorted(case_results, key=lambda x: x.metrics.median_time):
                report.append(
                    f"  {r.numerics.method:10s} | "
                    f"rank={str(r.numerics.tt_rank) if r.numerics.tt_rank else 'N/A':3s} | "
                    f"batch={str(r.problem.batch_size) if r.problem.batch_size else 'N/A':4s} | "
                    f"time={r.metrics.median_time*1000:6.2f}ms | "
                    f"device={r.hardware.device_type}"
                )
        
        return "\n".join(report)
