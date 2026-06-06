# VRP Optimization Suite

This repository showcases the evolution of algorithms used to solve the **Vehicle Routing Problem (VRP)**. It demonstrates a complete progression from traditional heuristics to modern hardware-accelerated and AI-driven approaches.

## Included Algorithms

1. **`aco_basic.py`**: Traditional Ant Colony Optimization (ACO) using standard probability rules.
2. **`aco_hybrid_tabu.py`**: Ant Colony System (ACS) integrated with Tabu Search for advanced local optimization.
3. **`aco_adaptive_numpy.py`**: Adaptive parameters powered by a high-speed Numpy matrix engine.
4. **`aco_parallel_tensor.py`**: GPU-accelerated parallel processing using PyTorch (simulating 100 ants simultaneously).
5. **`deep_rl_qlearning.py`**: Deep Reinforcement Learning with a Neural Network policy for dynamic routing.

## Quick Start

**1. Clone & Install Dependencies:**
Ensure you have Python 3.8+ installed.
```bash
git clone [https://github.com/Cuong51206/vrp-optimization-suite.git](https://github.com/Cuong51206/vrp-optimization-suite.git)
cd VRP-Optimization-Suite
pip install -r requirements.txt