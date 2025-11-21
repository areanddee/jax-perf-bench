"""
Example Usage of Performance Database Framework

This demonstrates practical workflows for benchmarking atmospheric model components
"""

import functools
import jax
import jax.numpy as jnp
from perfbench import (
    PerformanceDatabase,
    BenchmarkRunner,
    BenchmarkAnalyzer,
    HardwareConfig,
    NumericsConfig,
    ProblemConfig,
    BenchmarkConfig
)


# =============================================================================
# Example 1: Simple Benchmark Workflow
# =============================================================================

def example_1_basic_workflow():
    """Basic workflow: benchmark a single function"""
    print("\n" + "="*70)
    print("Example 1: Basic Benchmark Workflow")
    print("="*70)
    
    # Initialize
    db = PerformanceDatabase("examples.json")
    runner = BenchmarkRunner()
    
    # Setup hardware config (auto-detect would be better in practice)
    hardware = HardwareConfig(
        device_type=jax.devices()[0].platform,
        device_name=str(jax.devices()[0].device_kind),
        device_count=jax.device_count()
    )
    
    # Simple test function - create arrays outside JIT for proper benchmarking
    n = 1000
    # Note: JAX tridiagonal_solve requires all diagonals to have same length
    dl = jnp.concatenate([jnp.zeros(1), jnp.ones(n-1)])  # dl[0] = 0
    d = jnp.ones(n) * 2.0
    du = jnp.concatenate([jnp.ones(n-1), jnp.zeros(1)])  # du[-1] = 0
    b = jnp.ones((n, 1))  # RHS as column vector
    
    @jax.jit
    def tridiagonal_solve_test(dl, d, du, b):
        """Simulate a tridiagonal solve"""
        x = jax.lax.linalg.tridiagonal_solve(dl, d, du, b)
        return x
    
    # Configure
    numerics = NumericsConfig(method='Thomas', precision='float32')
    problem = ProblemConfig(
        test_case='tridiagonal_solve',
        n_horizontal=1,
        n_vertical=n,
        n_steps=1
    )
    
    # Benchmark
    result = runner.run_full_benchmark(
        func=lambda: tridiagonal_solve_test(dl, d, du, b),
        hardware=hardware,
        numerics=numerics,
        problem=problem,
        notes="Baseline tridiagonal solver test"
    )
    
    db.add_result(result)
    db.save()
    
    print(f"Median time: {result.metrics.median_time*1e6:.1f} µs")
    print(f"p90 time: {result.metrics.percentiles['p90']*1e6:.1f} µs")
    print(f"Saved to: {db.db_path}")


# =============================================================================
# Example 2: Comparing TT Ranks
# =============================================================================

def example_2_tt_rank_sweep():
    """Benchmark different TT ranks to find optimal compression"""
    print("\n" + "="*70)
    print("Example 2: TT Rank Sweep")
    print("="*70)
    
    db = PerformanceDatabase("tt_rank_sweep.json")
    runner = BenchmarkRunner()
    
    hardware = HardwareConfig(
        device_type=jax.devices()[0].platform,
        device_name=str(jax.devices()[0].device_kind)
    )
    
    # Simulate TT decompression + computation
    @functools.partial(jax.jit, static_argnums=(0, 1, 2))
    def tt_physics_kernel(rank, batch_size, n_levels):
        """Simulated TT-compressed column physics"""
        # Simulate TT factors: G(i,k,r)
        G = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_levels, rank))
        
        # Simulate decompression (contract over rank dimension)
        # In reality: state = sum_r G(i,k,r) * G(j,k,r)
        state = jnp.sum(G, axis=-1)  # Simplified
        
        # Simulate physics computation
        tendency = jnp.sin(state) * jnp.cos(state)
        
        return tendency
    
    # Sweep over ranks
    ranks = [2, 4, 8, 16, 32]
    batch_size = 128
    n_levels = 100
    
    for rank in ranks:
        numerics = NumericsConfig(
            method='TT',
            tt_rank=rank,
            precision='float32'
        )
        
        problem = ProblemConfig(
            test_case='kessler_microphysics',
            n_horizontal=10000,
            n_vertical=n_levels,
            n_columns=10000,
            batch_size=batch_size,
            n_steps=10
        )
        
        # Custom benchmark config for faster iteration
        bench_config = BenchmarkConfig(
            n_warmup=3,
            n_iterations=10
        )
        
        result = runner.run_full_benchmark(
            func=lambda: tt_physics_kernel(rank, batch_size, n_levels),
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            benchmark_config=bench_config,
            notes=f"TT rank={rank} sweep"
        )
        
        db.add_result(result)
        print(f"Rank {rank:2d}: {result.metrics.median_time*1000:6.2f} ms")
    
    db.save()
    
    # Analyze scaling
    analyzer = BenchmarkAnalyzer(db)
    scaling = analyzer.scaling_analysis(
        test_case='kessler_microphysics',
        method='TT',
        vary_param='numerics.tt_rank',
        metric='median_time'
    )
    
    print("\nScaling Analysis:")
    print("Rank | Time (ms)")
    print("-" * 20)
    for rank, time in scaling.items():
        print(f"{rank:4d} | {time*1000:8.2f}")


# =============================================================================
# Example 3: Batch Size Optimization
# =============================================================================

def example_3_batch_size_sweep():
    """Find optimal batch size for cache efficiency"""
    print("\n" + "="*70)
    print("Example 3: Batch Size Optimization")
    print("="*70)
    
    db = PerformanceDatabase("batch_size_sweep.json")
    runner = BenchmarkRunner()
    
    hardware = HardwareConfig(
        device_type=jax.devices()[0].platform,
        device_name=str(jax.devices()[0].device_kind)
    )
    
    @functools.partial(jax.jit, static_argnums=(0, 1))
    def column_physics(batch_size, n_levels):
        """Simulate column physics with temporaries"""
        # State variables
        T = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_levels))
        q = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_levels))
        
        # Compute derived quantities (temporaries)
        e_sat = 611.2 * jnp.exp(17.67 * T / (T + 243.5))  # Saturation vapor pressure
        q_sat = 0.622 * e_sat / (1e5 - e_sat)             # Saturation mixing ratio
        
        # Physics tendency
        condensation = jnp.maximum(0.0, q - q_sat) / 300.0  # Simple relaxation
        
        return condensation
    
    # Sweep batch sizes
    batch_sizes = [16, 32, 64, 128, 256, 512]
    n_levels = 100
    
    for batch_size in batch_sizes:
        numerics = NumericsConfig(method='FV', fv_order=2, precision='float32')
        problem = ProblemConfig(
            test_case='simple_condensation',
            n_horizontal=10000,
            n_vertical=n_levels,
            batch_size=batch_size,
            n_steps=10
        )
        
        result = runner.run_full_benchmark(
            func=lambda: column_physics(batch_size, n_levels),
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            notes=f"Batch size={batch_size} optimization"
        )
        
        db.add_result(result)
        
        # Compute efficiency metric: columns processed per second
        columns_per_sec = batch_size / result.metrics.median_time
        print(f"Batch {batch_size:3d}: {result.metrics.median_time*1000:6.2f} ms "
              f"({columns_per_sec:8.0f} col/s)")
    
    db.save()


# =============================================================================
# Example 4: Acoustic Substep Overhead
# =============================================================================

def example_4_acoustic_substeps():
    """Test overhead of repeated calls (acoustic substeps)"""
    print("\n" + "="*70)
    print("Example 4: Acoustic Substep Overhead")
    print("="*70)
    
    db = PerformanceDatabase("acoustic_substeps.json")
    runner = BenchmarkRunner()
    
    hardware = HardwareConfig(
        device_type=jax.devices()[0].platform,
        device_name=str(jax.devices()[0].device_kind)
    )
    
    # Pre-create tridiagonal matrices to avoid shape tracing issues
    batch_size = 128
    n_levels = 100
    
    # Tridiagonal matrices for acoustic solve - all must have same length
    dl = jnp.concatenate([jnp.zeros(1), jnp.ones(n_levels-1) * -0.5])
    d = jnp.ones(n_levels) * 2.0
    du = jnp.concatenate([jnp.ones(n_levels-1) * -0.5, jnp.zeros(1)])
    
    @jax.jit
    def acoustic_solve_step(state, dl, d, du):
        """Single acoustic solve step"""
        # Solve for each column in batch
        def solve_column(col):
            return jax.lax.linalg.tridiagonal_solve(dl, d, du, col[:, None]).squeeze()
        
        # Vectorize over batch
        result = jax.vmap(solve_column)(state)
        return result
    
    # Test with different numbers of substeps
    substep_counts = [1, 5, 10, 20]
    
    for n_substeps in substep_counts:
        def acoustic_solver():
            """Full acoustic solver with substeps"""
            state = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_levels))
            
            # Multiple substeps
            for _ in range(n_substeps):
                state = acoustic_solve_step(state, dl, d, du)
            
            return state
        
        numerics = NumericsConfig(method='HEVI', precision='float32')
        problem = ProblemConfig(
            test_case='acoustic_solver',
            n_horizontal=10000,
            n_vertical=n_levels,
            batch_size=batch_size,
            n_substeps=n_substeps,
            n_steps=1
        )
        
        result = runner.run_full_benchmark(
            func=acoustic_solver,
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            notes=f"Acoustic solver with {n_substeps} substeps"
        )
        
        db.add_result(result)
        
        time_per_substep = result.metrics.median_time / n_substeps
        print(f"Substeps {n_substeps:2d}: "
              f"total={result.metrics.median_time*1000:6.2f} ms, "
              f"per substep={time_per_substep*1000:6.2f} ms")
    
    db.save()


# =============================================================================
# Example 5: Method Comparison
# =============================================================================

def example_5_method_comparison():
    """Compare different numerical methods"""
    print("\n" + "="*70)
    print("Example 5: Method Comparison")
    print("="*70)
    
    db = PerformanceDatabase("method_comparison.json")
    runner = BenchmarkRunner()
    
    hardware = HardwareConfig(
        device_type=jax.devices()[0].platform,
        device_name=str(jax.devices()[0].device_kind)
    )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def physics_kernel(shape):
        x = jax.random.normal(jax.random.PRNGKey(0), shape)
        return jnp.sin(x) * jnp.cos(x)
    
    methods = [
        ('TT', {'tt_rank': 4}),
        ('TT', {'tt_rank': 8}),
        ('FV', {'fv_order': 2}),
        ('FV', {'fv_order': 3}),
    ]
    
    batch_size = 128
    n_levels = 100
    
    for method, params in methods:
        numerics = NumericsConfig(method=method, precision='float32', **params)
        problem = ProblemConfig(
            test_case='standard_physics',
            n_horizontal=10000,
            n_vertical=n_levels,
            batch_size=batch_size
        )
        
        result = runner.run_full_benchmark(
            func=lambda: physics_kernel((batch_size, n_levels)),
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            notes=f"Method comparison: {method} with {params}"
        )
        
        db.add_result(result)
    
    db.save()
    
    # Compare methods
    analyzer = BenchmarkAnalyzer(db)
    print("\n" + analyzer.summary_report(db.results))
    
    # Export to CSV for further analysis
    db.export_csv("method_comparison.csv")
    print("\nResults exported to method_comparison.csv")


# =============================================================================
# Example 6: Query and Analysis
# =============================================================================

def example_6_query_analysis():
    """Demonstrate query and analysis capabilities"""
    print("\n" + "="*70)
    print("Example 6: Query and Analysis")
    print("="*70)
    
    # Assuming we have existing database
    db = PerformanceDatabase("examples.json")
    
    if not db.results:
        print("No results in database. Run other examples first.")
        return
    
    analyzer = BenchmarkAnalyzer(db)
    
    # Query specific test case
    kessler_results = db.query(test_case='kessler_microphysics')
    print(f"\nFound {len(kessler_results)} Kessler microphysics results")
    
    # Query by method and device
    gpu_tt_results = db.query(method='TT', device_type='gpu')
    print(f"Found {len(gpu_tt_results)} GPU TT results")
    
    # Compare methods on same test case
    if kessler_results:
        comparison = analyzer.compare_methods(
            test_case='kessler_microphysics',
            methods=['TT', 'FV']
        )
        print("\nMethod comparison:")
        for method, time in comparison.items():
            print(f"  {method}: {time*1000:.2f} ms")
    
    # Scaling analysis
    if gpu_tt_results:
        scaling = analyzer.scaling_analysis(
            test_case='kessler_microphysics',
            method='TT',
            vary_param='numerics.tt_rank'
        )
        print("\nTT rank scaling:")
        for rank, time in scaling.items():
            print(f"  rank {rank}: {time*1000:.2f} ms")
    
    # Summary report
    print("\n" + analyzer.summary_report(db.results[:5]))


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Performance Database Framework - Examples")
    print("="*70)
    
    # Run examples
    example_1_basic_workflow()
    example_2_tt_rank_sweep()
    example_3_batch_size_sweep()
    example_4_acoustic_substeps()
    example_5_method_comparison()
    example_6_query_analysis()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
