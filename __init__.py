"""
JAX Performance Benchmarking Framework

A comprehensive framework for benchmarking, storing, and analyzing performance
data for JAX-based scientific computing projects.

Example usage:
    >>> from perfbench import (
    ...     PerformanceDatabase, BenchmarkRunner,
    ...     HardwareConfig, NumericsConfig, ProblemConfig
    ... )
    >>> 
    >>> db = PerformanceDatabase("results.json")
    >>> runner = BenchmarkRunner()
    >>> 
    >>> hardware = HardwareConfig(device_type='gpu', device_name='A100')
    >>> numerics = NumericsConfig(method='TT', tt_rank=4)
    >>> problem = ProblemConfig(test_case='kessler', n_horizontal=10000, n_vertical=100)
    >>> 
    >>> result = runner.run_full_benchmark(my_func, hardware, numerics, problem)
    >>> db.add_result(result)
    >>> db.save()
"""

__version__ = "0.1.0"

# Schema
from .schema import (
    HardwareConfig,
    NumericsConfig,
    ProblemConfig,
    BenchmarkConfig,
    PerformanceMetrics,
    BenchmarkResult,
)

# Core functionality
from .core import (
    Timer,
    BenchmarkRunner,
)

# Database
from .database import (
    PerformanceDatabase,
)

# Analysis
from .analysis import (
    BenchmarkAnalyzer,
)

# Plotting (optional, may fail if matplotlib not installed)
try:
    from .plotting import (
        PlotConfig,
        plot_sweep,
        plot_comparison,
        plot_heatmap,
        plot_scaling,
        plot_timeline,
        plot_multi_sweep,
    )
    __all_plotting__ = [
        'PlotConfig',
        'plot_sweep',
        'plot_comparison',
        'plot_heatmap',
        'plot_scaling',
        'plot_timeline',
        'plot_multi_sweep',
    ]
except ImportError:
    __all_plotting__ = []

__all__ = [
    # Version
    '__version__',
    
    # Schema
    'HardwareConfig',
    'NumericsConfig',
    'ProblemConfig',
    'BenchmarkConfig',
    'PerformanceMetrics',
    'BenchmarkResult',
    
    # Core
    'Timer',
    'BenchmarkRunner',
    
    # Database
    'PerformanceDatabase',
    
    # Analysis
    'BenchmarkAnalyzer',
] + __all_plotting__
