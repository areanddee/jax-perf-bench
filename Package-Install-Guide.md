# 0. Create jax venv, activate

# 1. Clone jax_perf_bench from repo

# 2. Navigate to package
cd ./jax-perf-bench

# 3. Install with plotting (use quotes!)
pip install -e '.[plotting]'

# 4. Test installation
python -c "from perfbench import *; print('✓ Success!')"

# 5. Test with plotting
python -c "from perfbench import plot_sweep; print('✓ Plotting available!')"

# 6. Run benchmarking examples
cd examples
python ./examples/atmospheric_examples.py

# 7. Run plotting examples:
python ./examples/plotting_demo.py

