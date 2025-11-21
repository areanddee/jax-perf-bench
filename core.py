"""
Core benchmarking functionality

Timer and BenchmarkRunner for JAX-aware performance measurement.
"""

import time
from typing import Callable, Optional, Dict
from datetime import datetime
import numpy as np
import jax

from .schema import (
    BenchmarkConfig,
    BenchmarkResult,
    PerformanceMetrics,
    HardwareConfig,
    NumericsConfig,
    ProblemConfig
)


class Timer:
    """Unified timer with JAX integration"""
    
    def __init__(self, use_jax_block: bool = True, device_sync: bool = True):
        self.use_jax_block = use_jax_block
        self.device_sync = device_sync
        
    def time_function(self, func: Callable, *args, **kwargs) -> float:
        """Time a single function call"""
        # Warmup to ensure compilation
        if self.use_jax_block:
            result = func(*args, **kwargs)
            if isinstance(result, jax.Array):
                jax.block_until_ready(result)
        
        # Actual timing
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.use_jax_block and isinstance(result, jax.Array):
            jax.block_until_ready(result)
        
        if self.device_sync:
            # Additional synchronization for multi-device
            jax.effects_barrier()
        
        end = time.perf_counter()
        return end - start
    
    def benchmark(self, func: Callable, *args, 
                  n_warmup: int = 5, 
                  n_iterations: int = 20,
                  **kwargs) -> np.ndarray:
        """Run full benchmark with warmup and multiple iterations"""
        
        # Warmup phase
        for _ in range(n_warmup):
            result = func(*args, **kwargs)
            if self.use_jax_block and isinstance(result, jax.Array):
                jax.block_until_ready(result)
        
        # Timing phase
        times = []
        for _ in range(n_iterations):
            elapsed = self.time_function(func, *args, **kwargs)
            times.append(elapsed)
        
        return np.array(times)


class BenchmarkRunner:
    """Main benchmark runner with statistical analysis"""
    
    def __init__(self, timer: Optional[Timer] = None):
        self.timer = timer or Timer()
    
    def run_benchmark(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        benchmark_config: Optional[BenchmarkConfig] = None
    ) -> np.ndarray:
        """Run benchmark and return raw timing data"""
        kwargs = kwargs or {}
        config = benchmark_config or BenchmarkConfig()
        
        times = self.timer.benchmark(
            func, *args,
            n_warmup=config.n_warmup,
            n_iterations=config.n_iterations,
            **kwargs
        )
        
        return times
    
    def compute_statistics(
        self,
        times: np.ndarray,
        percentiles: list = None
    ) -> PerformanceMetrics:
        """Compute statistical metrics from timing data"""
        percentiles = percentiles or [50, 90, 95, 99]
        
        percentile_dict = {
            f'p{int(p)}': float(np.percentile(times, p))
            for p in percentiles
        }
        
        return PerformanceMetrics(
            mean_time=float(np.mean(times)),
            median_time=float(np.median(times)),
            std_time=float(np.std(times)),
            min_time=float(np.min(times)),
            max_time=float(np.max(times)),
            percentiles=percentile_dict
        )
    
    def run_full_benchmark(
        self,
        func: Callable,
        hardware: HardwareConfig,
        numerics: NumericsConfig,
        problem: ProblemConfig,
        benchmark_config: Optional[BenchmarkConfig] = None,
        args: tuple = (),
        kwargs: dict = None,
        accuracy_func: Optional[Callable] = None,
        notes: str = ""
    ) -> BenchmarkResult:
        """
        Run complete benchmark with all metadata collection
        
        Args:
            func: Function to benchmark
            hardware: Hardware configuration
            numerics: Numerics configuration
            problem: Problem configuration
            benchmark_config: Benchmark methodology config
            args: Positional arguments for func
            kwargs: Keyword arguments for func
            accuracy_func: Optional function to compute accuracy metrics
                          Should return Dict[str, float]
            notes: Additional notes about this benchmark
        
        Returns:
            Complete BenchmarkResult with all metadata
        """
        config = benchmark_config or BenchmarkConfig()
        kwargs = kwargs or {}
        
        # Run timing benchmark
        times = self.run_benchmark(func, args, kwargs, config)
        
        # Compute statistics
        metrics = self.compute_statistics(times, config.percentiles)
        
        # Compute throughput if applicable
        if problem.n_columns and problem.n_steps:
            total_columns = problem.n_columns * problem.n_steps
            metrics.throughput = total_columns / metrics.median_time
        
        # Compute accuracy metrics if provided
        if accuracy_func is not None:
            try:
                metrics.accuracy_metrics = accuracy_func()
            except Exception as e:
                metrics.accuracy_metrics = {'error': str(e)}
        
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            benchmark_config=config,
            metrics=metrics,
            jax_version=jax.__version__,
            notes=notes
        )
        
        return result
    
    def _generate_id(self) -> str:
        """Generate unique benchmark ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
