"""
Schema definitions for performance benchmarking

All configuration and result dataclasses with extensible metadata support.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    device_type: str  # 'cpu', 'gpu', 'tpu'
    device_name: str  # e.g., 'NVIDIA A100', 'TPU v4'
    device_count: int = 1
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    
    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NumericsConfig:
    """Numerical method configuration"""
    method: str  # 'TT', 'FV', 'PLR', 'Spectral', etc.
    
    # For TT compression
    tt_rank: Optional[int] = None
    tt_compression_method: Optional[str] = None  # 'SVD', 'ALS', etc.
    
    # For finite volume
    fv_order: Optional[int] = None
    fv_limiter: Optional[str] = None
    
    # For PLR (Piecewise Linear Reconstruction)
    plr_order: Optional[int] = None
    
    # Discretization
    precision: str = 'float32'  # 'float32', 'float64', 'bfloat16'
    
    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemConfig:
    """Problem size and configuration"""
    test_case: str  # 'kessler', 'acoustic_solver', 'grey_radiation', etc.
    
    # Domain size
    n_horizontal: int  # Total horizontal grid points (or i*j)
    n_vertical: int    # Vertical levels
    n_columns: Optional[int] = None  # For column physics
    
    # Batch/block sizes
    batch_size: Optional[int] = None
    block_size_i: Optional[int] = None
    block_size_j: Optional[int] = None
    
    # Time stepping
    dt: Optional[float] = None
    n_steps: Optional[int] = None
    n_substeps: Optional[int] = None  # For acoustic solver
    
    # Physics-specific parameters
    physics_params: Dict[str, Any] = field(default_factory=dict)
    
    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Benchmark methodology configuration"""
    n_warmup: int = 5
    n_iterations: int = 20
    percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99])
    
    # Timing strategy
    use_jax_block: bool = True  # Use jax.block_until_ready()
    device_sync: bool = True     # Explicit device synchronization
    
    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    # Timing (all in seconds unless otherwise noted)
    mean_time: float
    median_time: float
    std_time: float
    min_time: float
    max_time: float
    percentiles: Dict[str, float]  # e.g., {'p50': 0.1, 'p90': 0.12, ...}
    
    # Throughput
    throughput: Optional[float] = None  # columns/sec or cells/sec
    
    # Memory (in MB)
    memory_allocated: Optional[float] = None
    memory_peak: Optional[float] = None
    
    # Accuracy metrics
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result with all metadata"""
    # Unique identifier
    benchmark_id: str
    timestamp: str
    
    # Configuration
    hardware: HardwareConfig
    numerics: NumericsConfig
    problem: ProblemConfig
    benchmark_config: BenchmarkConfig
    
    # Results
    metrics: PerformanceMetrics
    
    # Git/version info
    git_commit: Optional[str] = None
    jax_version: Optional[str] = None
    
    # Additional notes
    notes: str = ""
    
    # Schema version for future compatibility
    schema_version: str = "1.0.0"
    
    # Extensible metadata
    extra: Dict[str, Any] = field(default_factory=dict)
