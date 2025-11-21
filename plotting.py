"""
Plotting utilities for benchmark visualization

Optional module requiring matplotlib. Install with:
    pip install jax-perf-bench[plotting]
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    Figure = Any  # Type hint placeholder
    Axes = Any


from .database import PerformanceDatabase
from .schema import BenchmarkResult


def _check_plotting():
    """Check if plotting dependencies are available"""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "Plotting requires matplotlib. Install with:\n"
            "  pip install jax-perf-bench[plotting]\n"
            "or:\n"
            "  pip install matplotlib"
        )


class PlotConfig:
    """Configuration for plot styling"""
    
    def __init__(self):
        self.figsize = (10, 6)
        self.dpi = 100
        self.style = 'seaborn-v0_8-whitegrid'  # matplotlib style
        self.colormap = 'viridis'
        self.marker_size = 8
        self.line_width = 2
        self.error_alpha = 0.2  # Transparency for error bands
        self.grid = True
        self.legend_loc = 'best'


def _extract_data(
    db: PerformanceDatabase,
    test_case: str,
    method: str,
    x_param: str,
    y_param: str = 'median_time'
) -> Tuple[np.ndarray, np.ndarray, List[BenchmarkResult]]:
    """
    Extract x, y data from database
    
    Returns:
        x_values, y_values, results
    """
    results = db.query(test_case=test_case, method=method)
    
    if not results:
        raise ValueError(f"No results found for test_case='{test_case}', method='{method}'")
    
    data_points = []
    for result in results:
        # Get x value
        if '.' in x_param:
            parts = x_param.split('.')
            x_val = db._get_nested_attr(result, parts)
        else:
            x_val = getattr(result.problem, x_param, None)
        
        # Get y value
        if '.' in y_param:
            parts = y_param.split('.')
            y_val = db._get_nested_attr(result, parts)
        else:
            y_val = getattr(result.metrics, y_param, None)
        
        if x_val is not None and y_val is not None:
            data_points.append((x_val, y_val, result))
    
    # Sort by x value
    data_points.sort(key=lambda p: p[0])
    
    x_values = np.array([p[0] for p in data_points])
    y_values = np.array([p[1] for p in data_points])
    results = [p[2] for p in data_points]
    
    return x_values, y_values, results


def plot_sweep(
    db: PerformanceDatabase,
    test_case: str,
    method: str,
    x: str,
    y: str = 'median_time',
    error_bars: Optional[str] = 'std',
    scale: str = 'linear',
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None,
    **plot_kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot a parameter sweep
    
    Args:
        db: Performance database
        test_case: Test case name
        method: Numerical method name
        x: Parameter to vary (e.g., 'numerics.tt_rank', 'problem.batch_size')
        y: Metric to plot (default: 'median_time')
        error_bars: Error bar style - 'std', 'p90-p10', 'percentile', 'minmax', None
        scale: 'linear', 'loglog', 'semilogx', 'semilogy'
        title: Custom title (default: auto-generated)
        subtitle: Additional subtitle line
        xlabel: Custom x-axis label (default: auto-generated)
        ylabel: Custom y-axis label (default: auto-generated)
        ax: Matplotlib axes (creates new if None)
        config: PlotConfig for styling
        **plot_kwargs: Additional arguments passed to plot()
    
    Returns:
        fig, ax
    
    Examples:
        >>> plot_sweep(db, 'kessler', 'TT', 'numerics.tt_rank', 
        ...            error_bars='std',
        ...            title='TT Rank Optimization',
        ...            subtitle='Error bars: ±1σ (n=20 iterations)')
    """
    _check_plotting()
    
    config = config or PlotConfig()
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    # Extract data
    x_values, y_values, results = _extract_data(db, test_case, method, x, y)
    
    # Compute error bars if requested
    if error_bars:
        if error_bars == 'std':
            y_err = np.array([r.metrics.std_time for r in results])
        elif error_bars == 'p90-p10':
            y_upper = np.array([r.metrics.percentiles.get('p90', r.metrics.median_time) for r in results])
            y_lower = np.array([r.metrics.percentiles.get('p10', r.metrics.median_time) for r in results])
            y_err = [y_values - y_lower, y_upper - y_values]
        elif error_bars == 'percentile':  # 25-75
            y_upper = np.array([r.metrics.percentiles.get('p75', r.metrics.median_time) for r in results])
            y_lower = np.array([r.metrics.percentiles.get('p25', r.metrics.median_time) for r in results])
            y_err = [y_values - y_lower, y_upper - y_values]
        elif error_bars == 'minmax':
            y_err = [y_values - np.array([r.metrics.min_time for r in results]),
                     np.array([r.metrics.max_time for r in results]) - y_values]
        else:
            y_err = None
    else:
        y_err = None
    
    # Auto-generate label with error bar info
    error_bar_labels = {
        'std': '±1σ',
        'percentile': 'IQR',
        'p90-p10': 'p10-p90',
        'minmax': 'min-max'
    }
    
    if 'label' not in plot_kwargs:
        label = f'{method}'
        if error_bars and error_bars in error_bar_labels:
            label += f' ({error_bar_labels[error_bars]})'
        plot_kwargs['label'] = label
    
    # Plot
    plot_kwargs.setdefault('marker', 'o')
    plot_kwargs.setdefault('markersize', config.marker_size)
    plot_kwargs.setdefault('linewidth', config.line_width)
    plot_kwargs.setdefault('label', f'{method}')
    
    if y_err is not None:
        ax.errorbar(x_values, y_values, yerr=y_err, 
                   capsize=5, capthick=2, **plot_kwargs)
    else:
        ax.plot(x_values, y_values, **plot_kwargs)
    
    # Set scale
    if scale == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif scale == 'semilogx':
        ax.set_xscale('log')
    elif scale == 'semilogy':
        ax.set_yscale('log')
    
    # Labels with defaults
    x_label = xlabel or x.split('.')[-1].replace('_', ' ').title()
    y_label = ylabel or (y.split('.')[-1].replace('_', ' ').title() + ' (s)')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Title with optional subtitle
    if title is None:
        title = f'{test_case}: {method} - {y.split(".")[-1]} vs {x.split(".")[-1]}'
    
    if subtitle:
        full_title = f'{title}\n{subtitle}'
    else:
        full_title = title
    
    ax.set_title(full_title)
    
    if config.grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc=config.legend_loc)
    
    fig.tight_layout()
    return fig, ax


def plot_comparison(
    db: PerformanceDatabase,
    test_case: str,
    methods: List[str],
    metric: str = 'median_time',
    error_bars: str = 'std',
    normalize: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None,
    **plot_kwargs
) -> Tuple[Figure, Axes]:
    """
    Compare multiple methods on same test case
    
    Args:
        db: Performance database
        test_case: Test case name
        methods: List of method names to compare
        metric: Metric to compare (default: 'median_time')
        error_bars: Error bar style - 'std', 'p90-p10', None
        normalize: If True, normalize to fastest method
        title: Custom title (default: auto-generated)
        subtitle: Additional subtitle line
        ylabel: Custom y-axis label (default: auto-generated)
        ax: Matplotlib axes
        config: PlotConfig for styling
    
    Returns:
        fig, ax
    
    Examples:
        >>> plot_comparison(db, 'kessler', ['TT', 'FV', 'PLR'],
        ...                 title='Method Comparison',
        ...                 subtitle='Error bars: ±1σ (n=20 iterations)')
    """
    _check_plotting()
    
    config = config or PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    # Collect data
    method_values = []
    method_errors = []
    
    for method in methods:
        results = db.query(test_case=test_case, method=method)
        if results:
            # Use most recent result
            result = sorted(results, key=lambda r: r.timestamp)[-1]
            value = getattr(result.metrics, metric)
            
            if error_bars == 'std':
                error = result.metrics.std_time
            elif error_bars == 'p90-p10':
                p90 = result.metrics.percentiles.get('p90', value)
                p10 = result.metrics.percentiles.get('p10', value)
                error = (p90 - p10) / 2
            else:
                error = 0
            
            method_values.append(value)
            method_errors.append(error)
        else:
            method_values.append(None)
            method_errors.append(0)
    
    # Remove None values
    valid_indices = [i for i, v in enumerate(method_values) if v is not None]
    methods = [methods[i] for i in valid_indices]
    method_values = [method_values[i] for i in valid_indices]
    method_errors = [method_errors[i] for i in valid_indices]
    
    # Normalize if requested
    if normalize and method_values:
        min_val = min(method_values)
        method_values = [v / min_val for v in method_values]
        method_errors = [e / min_val for e in method_errors]
    
    # Plot
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, method_values, yerr=method_errors if error_bars else None,
           capsize=5, alpha=0.8, **plot_kwargs)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    
    # Y-axis label with default
    y_label = ylabel or (f"{metric.replace('_', ' ').title()}" + 
                         (" (normalized)" if normalize else " (s)"))
    ax.set_ylabel(y_label)
    
    # Title with optional subtitle
    if title is None:
        title = f'{test_case}: Method Comparison'
    
    if subtitle:
        full_title = f'{title}\n{subtitle}'
    else:
        full_title = title
    
    ax.set_title(full_title)
    
    if config.grid:
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return fig, ax


def plot_heatmap(
    db: PerformanceDatabase,
    test_case: str,
    method: str,
    x: str,
    y: str,
    z: str = 'median_time',
    contours: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    interpolation: Optional[str] = None,
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None,
    **plot_kwargs
) -> Tuple[Figure, Axes]:
    """
    2D parameter sweep heatmap
    
    Note: This creates a grid-based heatmap for discrete parameter sweeps.
    Each cell represents a specific (x, y) parameter combination. For smoother
    visualization, run more parameter combinations or use interpolation='bilinear'.
    
    Args:
        db: Performance database
        test_case: Test case name
        method: Numerical method name
        x: First parameter (e.g., 'problem.batch_size')
        y: Second parameter (e.g., 'numerics.tt_rank')
        z: Metric to plot (default: 'median_time')
        contours: If True, overlay contour lines
        title: Custom title (default: auto-generated)
        subtitle: Additional subtitle line
        xlabel: Custom x-axis label (default: auto-generated)
        ylabel: Custom y-axis label (default: auto-generated)
        zlabel: Custom colorbar label (default: auto-generated)
        interpolation: Interpolation method ('bilinear', 'bicubic', None for discrete)
        ax: Matplotlib axes
        config: PlotConfig for styling
    
    Returns:
        fig, ax
    
    Examples:
        >>> # Discrete grid (default)
        >>> plot_heatmap(db, 'test', 'TT', 'problem.batch_size', 'numerics.tt_rank')
        >>> 
        >>> # With smoothing
        >>> plot_heatmap(db, 'test', 'TT', 'problem.batch_size', 'numerics.tt_rank',
        ...              interpolation='bilinear',
        ...              title='Parameter Space Exploration')
    """
    _check_plotting()
    
    config = config or PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    # Extract data
    results = db.query(test_case=test_case, method=method)
    
    # Build 2D grid
    data_dict = {}
    for result in results:
        # Get x value
        if '.' in x:
            x_val = db._get_nested_attr(result, x.split('.'))
        else:
            x_val = getattr(result.problem, x, None)
        
        # Get y value
        if '.' in y:
            y_val = db._get_nested_attr(result, y.split('.'))
        else:
            y_val = getattr(result.problem, y, None)
        
        # Get z value
        if '.' in z:
            z_val = db._get_nested_attr(result, z.split('.'))
        else:
            z_val = getattr(result.metrics, z, None)
        
        if x_val is not None and y_val is not None and z_val is not None:
            data_dict[(x_val, y_val)] = z_val
    
    if not data_dict:
        raise ValueError(f"No data found for {test_case}, {method}")
    
    # Create grid
    x_unique = sorted(set(k[0] for k in data_dict.keys()))
    y_unique = sorted(set(k[1] for k in data_dict.keys()))
    
    Z = np.full((len(y_unique), len(x_unique)), np.nan)
    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            if (x_val, y_val) in data_dict:
                Z[i, j] = data_dict[(x_val, y_val)]
    
    # Plot heatmap
    if interpolation:
        # Use imshow for smooth interpolation
        extent = [min(x_unique), max(x_unique), min(y_unique), max(y_unique)]
        im = ax.imshow(Z, cmap=config.colormap, aspect='auto', 
                      origin='lower', extent=extent, 
                      interpolation=interpolation, **plot_kwargs)
    else:
        # Use pcolormesh for discrete grid
        im = ax.pcolormesh(x_unique, y_unique, Z, 
                          cmap=config.colormap, shading='auto', **plot_kwargs)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar_label = zlabel or (z.replace('_', ' ').title() + ' (s)')
    cbar.set_label(cbar_label)
    
    # Add contours if requested
    if contours:
        # Create meshgrid for contours
        X, Y = np.meshgrid(x_unique, y_unique)
        cs = ax.contour(X, Y, Z, colors='white', alpha=0.5, linewidths=1)
        ax.clabel(cs, inline=True, fontsize=8)
    
    # Labels with defaults
    x_label = xlabel or x.split('.')[-1].replace('_', ' ').title()
    y_label = ylabel or y.split('.')[-1].replace('_', ' ').title()
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Title with optional subtitle
    if title is None:
        title = f'{test_case}: {method} - {z} Heatmap'
    
    if subtitle:
        full_title = f'{title}\n{subtitle}'
    else:
        full_title = title
    
    ax.set_title(full_title)
    
    fig.tight_layout()
    return fig, ax


def plot_scaling(
    db: PerformanceDatabase,
    test_case: str,
    method: str,
    x: str,
    y: str = 'median_time',
    scale: str = 'loglog',
    fit_power_law: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None,
    **plot_kwargs
) -> Tuple[Figure, Axes]:
    """
    Scaling analysis with optional power law fit
    
    Args:
        db: Performance database
        test_case: Test case name
        method: Numerical method name
        x: Parameter to vary (e.g., 'problem.n_vertical')
        y: Metric to plot (default: 'median_time')
        scale: 'loglog', 'semilogx', 'semilogy', 'linear'
        fit_power_law: If True, fit y = A * x^α and show α
        title: Custom title (default: auto-generated)
        subtitle: Additional subtitle line
        xlabel: Custom x-axis label (default: auto-generated)
        ylabel: Custom y-axis label (default: auto-generated)
        ax: Matplotlib axes
        config: PlotConfig for styling
    
    Returns:
        fig, ax
    
    Examples:
        >>> plot_scaling(db, 'test', 'TT', 'problem.n_vertical',
        ...              title='Vertical Scaling',
        ...              subtitle='Expected: O(n) for tridiagonal solve')
    """
    _check_plotting()
    
    config = config or PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    # Extract data
    x_values, y_values, results = _extract_data(db, test_case, method, x, y)
    
    # Plot data
    plot_kwargs.setdefault('marker', 'o')
    plot_kwargs.setdefault('markersize', config.marker_size)
    plot_kwargs.setdefault('linewidth', 0)
    plot_kwargs.setdefault('label', 'Data')
    
    ax.plot(x_values, y_values, **plot_kwargs)
    
    # Fit power law if requested
    if fit_power_law and len(x_values) >= 2:
        # Fit in log space: log(y) = log(A) + α * log(x)
        log_x = np.log(x_values)
        log_y = np.log(y_values)
        
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = coeffs[0]
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Plot fit
        x_fit = np.linspace(x_values.min(), x_values.max(), 100)
        y_fit = A * x_fit ** alpha
        
        ax.plot(x_fit, y_fit, '--', linewidth=config.line_width,
                label=f'Fit: $y = {A:.2e} \\times x^{{{alpha:.2f}}}$')
    
    # Set scale
    if scale == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif scale == 'semilogx':
        ax.set_xscale('log')
    elif scale == 'semilogy':
        ax.set_yscale('log')
    
    # Labels with defaults
    x_label = xlabel or x.split('.')[-1].replace('_', ' ').title()
    y_label = ylabel or (y.split('.')[-1].replace('_', ' ').title() + ' (s)')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Title with optional subtitle
    if title is None:
        title = f'{test_case}: {method} - Scaling Analysis'
    
    if subtitle:
        full_title = f'{title}\n{subtitle}'
    else:
        full_title = title
    
    ax.set_title(full_title)
    
    if config.grid:
        ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc=config.legend_loc)
    
    fig.tight_layout()
    return fig, ax


def plot_timeline(
    db: PerformanceDatabase,
    test_case: str,
    method: str,
    metric: str = 'median_time',
    smooth: bool = False,
    window: int = 5,
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None,
    **plot_kwargs
) -> Tuple[Figure, Axes]:
    """
    Performance over time (regression detection)
    
    Args:
        db: Performance database
        test_case: Test case name
        method: Numerical method name
        metric: Metric to plot (default: 'median_time')
        smooth: If True, apply moving average
        window: Window size for moving average
        ax: Matplotlib axes
        config: PlotConfig for styling
    
    Returns:
        fig, ax
    """
    _check_plotting()
    
    config = config or PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    # Extract data
    results = db.query(test_case=test_case, method=method)
    results = sorted(results, key=lambda r: r.timestamp)
    
    if not results:
        raise ValueError(f"No results found for {test_case}, {method}")
    
    timestamps = [r.timestamp for r in results]
    values = [getattr(r.metrics, metric) for r in results]
    
    # Plot raw data
    plot_kwargs.setdefault('marker', 'o')
    plot_kwargs.setdefault('markersize', config.marker_size)
    plot_kwargs.setdefault('alpha', 0.6 if smooth else 1.0)
    plot_kwargs.setdefault('label', 'Measurements')
    
    ax.plot(range(len(timestamps)), values, **plot_kwargs)
    
    # Apply smoothing if requested
    if smooth and len(values) >= window:
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(timestamps)), smoothed, 
                linewidth=config.line_width, label=f'Smoothed (window={window})')
    
    # Labels
    ax.set_xlabel('Benchmark Run')
    ax.set_ylabel(metric.replace('_', ' ').title() + ' (s)')
    ax.set_title(f'{test_case}: {method} - Performance Over Time')
    
    if config.grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc=config.legend_loc)
    
    fig.tight_layout()
    return fig, ax


def plot_multi_sweep(
    db: PerformanceDatabase,
    test_case: str,
    methods: List[str],
    x: str,
    y: str = 'median_time',
    error_bars: Optional[str] = 'std',
    scale: str = 'linear',
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None
) -> Tuple[Figure, Axes]:
    """
    Plot multiple methods on same parameter sweep
    
    Args:
        db: Performance database
        test_case: Test case name
        methods: List of method names
        x: Parameter to vary
        y: Metric to plot
        error_bars: Error bar style
        scale: Plot scale
        ax: Matplotlib axes
        config: PlotConfig for styling
    
    Returns:
        fig, ax
    """
    _check_plotting()
    
    config = config or PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    # Plot each method
    for method in methods:
        try:
            plot_sweep(db, test_case, method, x, y, error_bars, scale, ax, config)
        except ValueError:
            # Skip methods with no data
            pass
    
    # Update title
    ax.set_title(f'{test_case}: Multi-Method Comparison')
    
    return fig, ax
