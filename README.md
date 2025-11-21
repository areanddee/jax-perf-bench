# JAX Performance Benchmarking Framework

A comprehensive, extensible framework for benchmarking, storing, and analyzing performance data for JAX-based scientific computing projects. Designed specfically for atmospheric modeling kernels and other compute-intensive applications.

## Features

✨ **JAX-Aware Timing**: Proper synchronization and warmup for accurate benchmarks  
📊 **Statistical Analysis**: Mean, median, std, percentiles (p50, p90, p95, p99)  
🗄️ **JSON Database**: Human-readable storage with query capabilities  
🔍 **Rich Metadata**: Track hardware, numerics, problem config, and more  
📈 **Built-in Plotting**: Publication-quality visualizations (optional)  
🔧 **Extensible Schema**: Easy to add custom fields via `extra` dicts  
📦 **Pip Installable**: Simple installation and usage

## Installation

```bash
# Minimal install (no plotting)
pip install jax-perf-bench

# With plotting support
pip install jax-perf-bench[plotting]

# Development install
git clone https://github.com/yourusername/jax-perf-bench
cd jax-perf-bench
pip install -e .[plotting,dev]
```

## Quick Start

```python
from perfbench import (
    PerformanceDatabase, BenchmarkRunner,
    HardwareConfig, NumericsConfig, ProblemConfig
)
import jax.numpy as jnp

# Initialize
db = PerformanceDatabase("my_results.json")
runner = BenchmarkRunner()

# Configure
hardware = HardwareConfig(device_type='gpu', device_name='A100')
numerics = NumericsConfig(method='TT', tt_rank=4, precision='float32')
problem = ProblemConfig(
    test_case='kessler_microphysics',
    n_horizontal=10000,
    n_vertical=100,
    batch_size=128
)

# Your function to benchmark
def my_physics_function():
    x = jnp.ones((128, 100))
    return jnp.sum(jnp.sin(x) * jnp.cos(x))

# Benchmark
result = runner.run_full_benchmark(
    func=my_physics_function,
    hardware=hardware,
    numerics=numerics,
    problem=problem,
    notes="Initial baseline test"
)

# Save
db.add_result(result)
db.save()

print(f"Median time: {result.metrics.median_time*1000:.2f} ms")
```

## Core Components

### Configuration Classes

All configuration uses Python dataclasses with `extra` fields for extensibility:

```python
# Hardware configuration
hardware = HardwareConfig(
    device_type='gpu',  # 'cpu', 'gpu', 'tpu'
    device_name='NVIDIA A100',
    memory_gb=40,
    extra={'pci_bandwidth': '64 GB/s'}  # Custom fields
)

# Numerical method configuration
numerics = NumericsConfig(
    method='TT',  # 'TT', 'FV', 'PLR', 'Spectral', etc.
    tt_rank=4,
    precision='float32',
    extra={'adaptive_rank': True}  # Custom fields
)

# Problem configuration
problem = ProblemConfig(
    test_case='kessler_microphysics',
    n_horizontal=10000,
    n_vertical=100,
    batch_size=128,
    n_substeps=10,
    physics_params={'dt': 10.0},
    extra={'vertical_coordinate': 'hybrid_sigma'}  # Custom fields
)
```

### Database Operations

```python
db = PerformanceDatabase("results.json")

# Add results
db.add_result(result)
db.save()  # Auto-creates .bak backup

# Query
tt_results = db.query(method='TT', test_case='kessler')
gpu_results = db.query(device_type='gpu')
recent = db.query(min_date='2024-01-01')

# Advanced queries with nested attributes
rank4 = db.query(**{'numerics.tt_rank': 4})

# Export to CSV
db.export_csv("results.csv")
```

### Analysis Tools

```python
from perfbench import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer(db)

# Compare methods
comparison = analyzer.compare_methods(
    test_case='kessler',
    methods=['TT', 'FV', 'PLR']
)

# Scaling analysis
scaling = analyzer.scaling_analysis(
    test_case='kessler',
    method='TT',
    vary_param='numerics.tt_rank',
    metric='median_time'
)

# Summary report
print(analyzer.summary_report(db.results))
```

## Plotting (Optional)

Install with `pip install jax-perf-bench[plotting]`

### Parameter Sweeps

```python
from perfbench import plot_sweep

# Simple sweep with error bars
fig, ax = plot_sweep(
    db,
    test_case='kessler',
    method='TT',
    x='numerics.tt_rank',
    error_bars='std'
)
plt.savefig('tt_rank_sweep.pdf')
```

### Method Comparison

```python
from perfbench import plot_comparison

fig, ax = plot_comparison(
    db,
    test_case='kessler',
    methods=['TT', 'FV', 'PLR'],
    error_bars='p90-p10'
)
```

### 2D Parameter Sweeps (Heatmaps)

```python
from perfbench import plot_heatmap

fig, ax = plot_heatmap(
    db,
    test_case='kessler',
    method='TT',
    x='problem.batch_size',
    y='numerics.tt_rank',
    z='median_time',
    contours=True
)
```

### Scaling Analysis

```python
from perfbench import plot_scaling

# Log-log plot with power law fit
fig, ax = plot_scaling(
    db,
    test_case='acoustic_solver',
    method='TT',
    x='problem.n_vertical',
    scale='loglog',
    fit_power_law=True  # Shows O(n^α)
)
```

### Performance Regression

```python
from perfbench import plot_timeline

fig, ax = plot_timeline(
    db,
    test_case='kessler',
    method='TT',
    smooth=True,  # Moving average
    window=5
)
```

## Common Workflows

### TT Rank Optimization

```python
for rank in [2, 4, 8, 16, 32]:
    numerics = NumericsConfig(method='TT', tt_rank=rank)
    result = runner.run_full_benchmark(func, hardware, numerics, problem)
    db.add_result(result)

db.save()

# Analyze
from perfbench import plot_sweep
fig, ax = plot_sweep(db, 'kessler', 'TT', 'numerics.tt_rank')
```

### Batch Size Sweep

```python
for batch_size in [16, 32, 64, 128, 256]:
    problem.batch_size = batch_size
    result = runner.run_full_benchmark(func, hardware, numerics, problem)
    db.add_result(result)

db.save()
```

### Cross-Platform Comparison

```python
for device_type in ['cpu', 'gpu', 'tpu']:
    hardware.device_type = device_type
    result = runner.run_full_benchmark(func, hardware, numerics, problem)
    db.add_result(result)

db.save()
```

## Extension Examples

### Custom Metrics

```python
def compute_accuracy():
    # Your accuracy computation
    return {
        'l2_error': 1e-5,
        'max_error': 1e-4,
        'conservation_error': 1e-7
    }

result = runner.run_full_benchmark(
    func, hardware, numerics, problem,
    accuracy_func=compute_accuracy
)
```

### Custom Fields

```python
# Add custom fields anywhere
numerics.extra['adaptive_rank'] = True
problem.extra['vertical_coordinate'] = 'hybrid_sigma'
result.metrics.extra['peak_memory_gb'] = 12.5
```

## API Reference

See `examples/` directory for complete working examples.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{jax_perf_bench,
  title = {JAX Performance Benchmarking Framework},
  author = {Richard Loft},
  year = {2025},
  url = {https://github.com/areanddee/jax-perf-bench}
}
```

## Acknowledgments

Designed for benchmarking test kernels in JAX.
