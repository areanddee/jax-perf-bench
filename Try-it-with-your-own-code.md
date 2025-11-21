from perfbench import *
import jax.numpy as jnp

db = PerformanceDatabase("my_results.json")
runner = BenchmarkRunner()

# Your actual physics function
def my_physics():
    # Your code
    pass

hardware = HardwareConfig(device_type='cpu', device_name='MacBook Pro')
numerics = NumericsConfig(method='TT', tt_rank=4)
problem = ProblemConfig(test_case='kessler', n_horizontal=10000, n_vertical=100)

result = runner.run_full_benchmark(my_kessler_physics, hardware, numerics, problem)
db.add_result(result)
db.save()
