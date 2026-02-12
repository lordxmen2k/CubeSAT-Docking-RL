# 🛰️ Autonomous CubeSat Docking with Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep reinforcement learning framework for autonomous spacecraft rendezvous and docking operations using **Proximal Policy Optimization (PPO)** with curriculum learning. Features a 3D orbital mechanics simulation, fuel-efficient reward shaping, and an immersive VPython-based pilot interface with real-time HUD visualization.

![Success Rate](https://img.shields.io/badge/Success%20Rate-100%25-brightgreen)
![Fuel Efficiency](https://img.shields.io/badge/Fuel%20Remaining-93%25-blue)
![Convergence](https://img.shields.io/badge/Mean%20Steps-25.8-orange)

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Training](#-training)
- [Testing](#-testing)
- [Pilot Interface](#-pilot-interface)
- [Results](#-results)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## ✨ Features

### 🤖 Reinforcement Learning
- **PPO Algorithm**: State-of-the-art on-policy actor-critic method
- **Curriculum Learning**: Three-phase training (Easy → Normal → Hard)
- **Reward Shaping**: Multi-objective optimization for fuel, time, and safety
- **100% Success Rate**: Validated across 50 independent test episodes

### 🌍 Physics Simulation
- **3D Orbital Mechanics**: Realistic gravity, drag, and fuel dynamics
- **Semi-Implicit Euler Integration**: Energy-preserving numerical solver
- **Soft-Docking Constraints**: Position (< 1m) and velocity (< 0.5 m/s) requirements

### 🎮 Visualization & Interface
- **VPython 3D HUD**: Real-time pilot seat visualization
- **Matplotlib Trajectory Plots**: 3D trajectory with XY/XZ projections
- **Live Telemetry**: Range, closure velocity, and fuel indicators

### 📊 Evaluation Suite
- **Robustness Testing**: 50-episode seed-based evaluation
- **Fuel Profiling**: Detailed consumption analysis
- **3D Visualization**: Automated trajectory plotting

---

## 🔧 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd cubesat-docking-rl

# Or download and extract the source files
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Core RL and ML libraries
pip install stable-baselines3==2.0.0
pip install gymnasium==0.28.1
pip install numpy==1.24.0

# Visualization libraries
pip install vpython==7.6.4
pip install matplotlib==3.7.0

# Optional: For GPU acceleration (if CUDA available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation
```bash
python -c "import stable_baselines3; import gymnasium; import vpython; print('✅ All dependencies installed successfully!')"
```

---

## 📁 Project Structure

```
cubesat-docking-rl/
│
├── cubesat_train_3D.py          # Main training environment & PPO implementation
├── test_model_cubesat_3D.py     # Evaluation and testing suite
├── sim_docking_mission_simulation_pilot_seat.py  # VPython HUD interface
│
├── models/                      # Saved model checkpoints (created during training)
│   ├── cubesat_3d_phase1.zip
│   ├── cubesat_3d_phase2.zip
│   └── cubesat_3d_final.zip
│
├── outputs/                     # Generated plots and visualizations
│   └── 3d_trajectory_seed42.png
│
├── cubesat_docking_research.html # Research publication (view in browser)
├── README.md                    # This file
└── requirements.txt             # Python dependencies (optional)
```

---

## 🚀 Quick Start

### 1. Train the Model
```bash
python cubesat_train_3D.py
```
Training takes approximately 10-30 minutes depending on hardware. The curriculum learning automatically progresses through three difficulty phases.

### 2. Test the Trained Model
```bash
python test_model_cubesat_3D.py
```
This runs the full evaluation suite including:
- 50-episode robustness test
- Fuel consumption analysis
- 3D trajectory visualization

### 3. Launch Pilot Interface
```bash
# Terminal 1: Start the RL server
python sim_docking_mission_simulation_pilot_seat.py

# The VPython window will open automatically
# Watch the autonomous docking in 3D with live HUD!
```

---

## 📖 Usage

### Training the Agent

#### Basic Training
```bash
python cubesat_train_3D.py
```

The training process:
1. **Phase 1 (Easy)**: Fixed target, low initial velocity → 200k timesteps
2. **Phase 2 (Normal)**: Fixed target, moderate velocity → 300k timesteps  
3. **Phase 3 (Hard)**: Randomized target ±2m, fine-tuning → 200k timesteps

#### Training Output
```
models/
├── cubesat_3d_phase1.zip    # After Phase 1
├── cubesat_3d_phase2.zip    # After Phase 2
└── cubesat_3d_final.zip     # Final trained model
```

#### Resume Training
```python
from stable_baselines3 import PPO

# Load existing model
model = PPO.load("models/cubesat_3d_phase2.zip")
model.set_env(new_env)
model.learn(total_timesteps=200000)
```

---

### Testing and Evaluation

#### Run Full Test Suite
```bash
python test_model_cubesat_3D.py
```

#### Expected Output
```
============================================================
3D CubeSat Navigation - Test Suite
============================================================

Quick diagnostic - 5 seeds:
Seed  0: ✅ SUCCESS at step  26, dist=0.83, fuel=92.9
Seed  1: ✅ SUCCESS at step  27, dist=0.86, fuel=92.6
...

============================================================
RESULTS: 50 EPISODES
============================================================
Success rate: 50/50 (100.0%)

Successful episodes (50):
  Mean steps: 25.8
  Mean final distance: 0.905m
  Mean final speed: 0.223m/s

============================================================
3D FUEL CONSUMPTION
============================================================
Episodes: 20
Mean fuel used: 6.97
Mean remaining: 93.03%
```

#### Individual Test Components
```python
from test_model_cubesat_3D import test_robustness_3d, measure_fuel_3d, plot_3d_trajectory

# Test specific functionality
test_robustness_3d()       # 50-episode robustness test
measure_fuel_3d()          # Fuel consumption analysis
plot_3d_trajectory(seed=42)  # Generate 3D plot
```

---

### Pilot Interface (VPython HUD)

#### Launch the Interface
```bash
python sim_docking_mission_simulation_pilot_seat.py
```

#### Features
- **3D Scene**: Real-time CubeSat and space station visualization
- **Heads-Up Display**: Range, closure velocity, fuel remaining
- **Camera Following**: Automatic tracking of satellite motion
- **Docking Ring**: Visual indicator for successful approach

#### Controls
The interface runs autonomously using the trained RL policy. No manual input required—sit back and watch the AI pilot!

#### HUD Elements
```
┌─────────────────────────────────────────┐
│  RANGE: 5.23 M                          │
│                                         │
│         [Satellite]  →  [Station]       │
│                                         │
│  CLOSURE: 0.45 M/S                      │
│  FUEL: 87.3%                            │
└─────────────────────────────────────────┘
```

---

## 📊 Results

### Performance Metrics (Validated)

| Metric | Value | Description |
|--------|-------|-------------|
| **Success Rate** | 100% (50/50) | Perfect reliability across all test seeds |
| **Mean Steps** | 25.8 | Average time to convergence |
| **Fuel Remaining** | 93.03% | Exceptional fuel efficiency |
| **Fuel Used** | 6.97 ± 0.61 | Low variance indicates consistency |
| **Final Distance** | 0.905m ± 0.05m | Precise positioning |
| **Final Speed** | 0.223 m/s | Conservative, safe approach |

### Trajectory Visualization
The test suite generates `3d_trajectory_seed42.png` showing:
- 3D trajectory path
- XY projection (top view)
- XZ projection (side view)
- Distance profile over time

---

## ⚙️ Configuration

### Environment Parameters
Edit `cubesat_train_3D.py` to modify physics:

```python
# Physics constants
self.dt = 1.0              # Time step (seconds)
self.max_thrust = 0.15     # Maximum thrust (m/s²)
self.gravity = 0.02        # Gravity constant
self.drag = 0.01           # Atmospheric drag coefficient

# Docking constraints
self.success_dist = 1.0    # Success distance threshold (m)
self.success_speed = 0.5   # Success velocity threshold (m/s)
self.max_fuel = 100.0      # Initial fuel capacity
```

### Training Hyperparameters
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Adjust for faster/slower learning
    n_steps=2048,            # Rollout buffer size
    batch_size=64,           # Minibatch size
    n_epochs=10,             # Optimization epochs
    ent_coef=0.01,           # Entropy coefficient (exploration)
    verbose=1
)
```

### Reward Weights
```python
# In _calculate_reward()
reward = (
    progress * 10.0          # Progress toward target
    - dist * 0.5             # Distance penalty
    - (thrust_mag ** 2) * 2.0 # Fuel penalty
    - 0.1                    # Time penalty
    + alignment * 2.0        # Velocity alignment
)
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'gymnasium'"
```bash
pip install gymnasium
```

### Issue: "No module named 'vpython'"
```bash
pip install vpython
# Note: VPython requires a display. On headless servers, use xvfb:
# sudo apt-get install xvfb
# xvfb-run python sim_docking_mission_simulation_pilot_seat.py
```

### Issue: "CUDA out of memory"
```python
# Force CPU usage
model = PPO("MlpPolicy", env, device='cpu')
```

### Issue: Model not found during testing
```bash
# Train first to generate models
python cubesat_train_3D.py

# Or check models/ directory exists
mkdir -p models
```

### Issue: VPython window doesn't appear
```bash
# Ensure you have a display available
# For remote servers, use X11 forwarding:
ssh -X user@server

# Or use a virtual display:
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
```

---

## 📚 Additional Resources

### Research Publication
Open `cubesat_docking_research.html` in your browser for the full technical paper including:
- Mathematical derivations of orbital dynamics
- PPO algorithm details
- Reward shaping analysis
- Comprehensive experimental results

### Key Papers
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*
- Bengio, Y., et al. (2009). Curriculum Learning. *ICML*

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add attitude dynamics (6-DOF control)
- [ ] Implement sensor noise and actuator uncertainty
- [ ] Add obstacle avoidance capabilities
- [ ] Optimize for real-time embedded systems
- [ ] Sim-to-real transfer for physical CubeSats

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{cubesat_docking_rl,
  author = {Gerald Enrique Nelson Mc Kenzie},
  title = {Autonomous CubeSat Docking with Deep Reinforcement Learning},
  year = {2026},
  url = {https://github.com/your-repo/cubesat-docking-rl}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for the PPO implementation
- [VPython](https://vpython.org/) for 3D visualization
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the environment interface

---

<p align="center">
  <strong>🛰️ Safe docking, efficient fuel, autonomous future. 🚀</strong>
</p>
