"""
Improved plotting demo showcasing clean API without matplotlib code

This demonstrates the improved API where users pass title/subtitle/labels
directly to plotting functions instead of needing matplotlib code.
"""

import sys
from pathlib import Path
import os
import functools
import jax
import jax.numpy as jnp

from perfbench import (
    PerformanceDatabase, BenchmarkRunner,
    HardwareConfig, NumericsConfig, ProblemConfig,
    plot_sweep, plot_comparison, plot_heatmap, plot_scaling
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib not available - skipping plotting demo")
    sys.exit(0)


def generate_sample_data():
    """Generate sample benchmark data"""
    
    # Clear old database to avoid accumulation
    db_file = "demo_plots.json"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"✓ Cleared old database: {db_file}\n")
    
    db = PerformanceDatabase(db_file)
    runner = BenchmarkRunner()
    
    hardware = HardwareConfig(device_type='cpu', device_name='Demo CPU')
    
    # Generate TT rank sweep data
    print("Generating sample data...")
    for rank in [2, 4, 8, 16, 32]:
        numerics = NumericsConfig(method='TT', tt_rank=rank, precision='float32')
        problem = ProblemConfig(
            test_case='demo_physics',
            n_horizontal=10000,
            n_vertical=100,
            batch_size=128
        )
        
        @functools.partial(jax.jit, static_argnums=(0, 1, 2))
        def test_func(rank, batch_size, n_levels):
            G = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_levels, rank))
            state = jnp.sum(G, axis=-1)
            return jnp.sum(jnp.sin(state) * jnp.cos(state))
        
        result = runner.run_full_benchmark(
            func=lambda: test_func(rank, 128, 100),
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            notes=f"TT rank {rank} demo"
        )
        
        db.add_result(result)
        print(f"  Rank {rank}: {result.metrics.median_time*1000:.2f} ms")
    
    # Generate FV method results for comparison
    for order in [2, 3, 4]:
        numerics = NumericsConfig(method='FV', fv_order=order, precision='float32')
        problem = ProblemConfig(
            test_case='demo_physics',
            n_horizontal=10000,
            n_vertical=100,
            batch_size=128
        )
        
        @functools.partial(jax.jit, static_argnums=(0,))
        def test_func_fv(batch_size):
            x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 100))
            return jnp.sum(jnp.sin(x) * jnp.cos(x))
        
        result = runner.run_full_benchmark(
            func=lambda: test_func_fv(128),
            hardware=hardware,
            numerics=numerics,
            problem=problem,
            notes=f"FV order {order} demo"
        )
        
        db.add_result(result)
        print(f"  FV order {order}: {result.metrics.median_time*1000:.2f} ms")
    
    # Generate 2D sweep data for heatmap
    for batch_size in [32, 64, 128, 256]:
        for rank in [2, 4, 8, 16]:
            numerics = NumericsConfig(method='TT', tt_rank=rank)
            problem = ProblemConfig(
                test_case='heatmap_demo',
                n_horizontal=10000,
                n_vertical=100,
                batch_size=batch_size
            )
            
            @functools.partial(jax.jit, static_argnums=(0, 1, 2))
            def test_func_2d(rank, batch_size, n_levels):
                G = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_levels, rank))
                state = jnp.sum(G, axis=-1)
                return jnp.sum(jnp.sin(state))
            
            result = runner.run_full_benchmark(
                func=lambda: test_func_2d(rank, batch_size, 100),
                hardware=hardware,
                numerics=numerics,
                problem=problem,
                notes=f"2D sweep: batch={batch_size}, rank={rank}"
            )
            
            db.add_result(result)
    
    print(f"  Added 2D sweep data for heatmap\n")
    
    db.save()
    return db


def create_plots_improved_api(db, output_dir='.'):
    """
    Create plots using the improved API (no matplotlib code needed!)
    
    Args:
        db: Performance database
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots with improved API...")
    print("=" * 70)
    
    # 1. Parameter sweep - Clean API with automatic error bar labeling
    print("\n1. PARAMETER SWEEP (with automatic error bar labeling)")
    print("   Code:")
    print("   plot_sweep(db, 'demo_physics', 'TT', 'numerics.tt_rank',")
    print("              error_bars='std',")
    print("              title='TT Rank Optimization',")
    print("              subtitle='Error bars: ±1σ (n=20 iterations)')")
    
    fig, ax = plot_sweep(
        db, 
        test_case='demo_physics',
        method='TT',
        x='numerics.tt_rank',
        error_bars='std',
        title='TT Rank Optimization',
        subtitle='Error bars: ±1σ standard deviation (n=20 iterations)'
    )
    outfile = output_path / 'plot_sweep_improved.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {outfile.name}")
    
    # 2. Method comparison - Clean API
    print("\n2. METHOD COMPARISON")
    print("   Code:")
    print("   plot_comparison(db, 'demo_physics', ['TT', 'FV'],")
    print("                   title='Method Performance Comparison',")
    print("                   subtitle='Lower is better')")
    
    fig, ax = plot_comparison(
        db,
        test_case='demo_physics',
        methods=['TT', 'FV'],
        error_bars='std',
        title='Method Performance Comparison',
        subtitle='Lower is better'
    )
    outfile = output_path / 'plot_comparison_improved.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {outfile.name}")
    
    # 3. Scaling analysis - Clean API
    print("\n3. SCALING ANALYSIS (with power law fit)")
    print("   Code:")
    print("   plot_scaling(db, 'demo_physics', 'TT', 'numerics.tt_rank',")
    print("                scale='loglog', fit_power_law=True,")
    print("                title='TT Rank Scaling',")
    print("                subtitle='Expected: O(rank) for compression overhead')")
    
    fig, ax = plot_scaling(
        db,
        test_case='demo_physics',
        method='TT',
        x='numerics.tt_rank',
        scale='loglog',
        fit_power_law=True,
        title='TT Rank Scaling',
        subtitle='Expected: O(rank) for compression overhead'
    )
    outfile = output_path / 'plot_scaling_improved.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {outfile.name}")
    
    # 4. Heatmap - Clean API with interpolation option
    print("\n4. PARAMETER SPACE HEATMAP")
    print("   Code:")
    print("   plot_heatmap(db, 'heatmap_demo', 'TT',")
    print("                x='problem.batch_size', y='numerics.tt_rank',")
    print("                title='2D Parameter Space',")
    print("                subtitle='Exploring batch size vs TT rank')")
    
    fig, ax = plot_heatmap(
        db,
        test_case='heatmap_demo',
        method='TT',
        x='problem.batch_size',
        y='numerics.tt_rank',
        z='median_time',
        contours=True,
        title='2D Parameter Space Exploration',
        subtitle='Discrete grid: Each cell = one (batch_size, rank) measurement'
    )
    outfile = output_path / 'plot_heatmap_improved.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {outfile.name}")
    
    # 5. Combined figure
    print("\n5. COMBINED FIGURE (all plots in one)")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    plot_sweep(db, 'demo_physics', 'TT', 'numerics.tt_rank', 
               error_bars='std', 
               title='Parameter Sweep',
               subtitle='±1σ error bars',
               ax=axes[0, 0])
    
    plot_comparison(db, 'demo_physics', ['TT', 'FV'], 
                   title='Method Comparison',
                   ax=axes[0, 1])
    
    plot_scaling(db, 'demo_physics', 'TT', 'numerics.tt_rank', 
                scale='loglog', fit_power_law=True,
                title='Scaling Analysis',
                ax=axes[1, 0])
    
    plot_heatmap(db, 'heatmap_demo', 'TT', 
                'problem.batch_size', 'numerics.tt_rank',
                contours=True,
                title='Parameter Space',
                ax=axes[1, 1])
    
    plt.tight_layout()
    outfile = output_path / 'plot_combined_improved.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {outfile.name}")
    
    print("\n" + "=" * 70)
    print(f"\n✓ All plots generated successfully!")
    print(f"\nPlots saved to: {output_path.resolve()}/")
    print("  - plot_sweep_improved.png")
    print("  - plot_comparison_improved.png")
    print("  - plot_scaling_improved.png")
    print("  - plot_heatmap_improved.png")
    print("  - plot_combined_improved.png")


if __name__ == "__main__":
    print("=" * 70)
    print("IMPROVED PLOTTING API DEMO")
    print("=" * 70)
    print("\nThis demo shows the improved API where users can pass:")
    print("  • title: Custom plot title")
    print("  • subtitle: Additional context line")
    print("  • xlabel, ylabel: Custom axis labels")
    print("  • Automatic error bar labeling in legend")
    print("\nNO matplotlib code needed in user scripts!")
    print("=" * 70)
    
    # Output directory
    output_dir = Path.cwd() / 'plots_improved'
    
    # Generate data
    db = generate_sample_data()
    
    # Create plots with improved API
    create_plots_improved_api(db, output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print(f"Check {output_dir.resolve()}/ for plots.")
    print("=" * 70)
