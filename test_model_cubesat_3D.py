"""
Test script for 3D CubeSat Navigation Model - FIXED
Handles variable observation space sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from cubesat_train_3D import CubeSat3DEnv


def parse_observation(obs):
    """Parse observation array - handles different sizes"""
    # Check for 1D vs 2D array (batch dim)
    if obs.ndim > 1:
        obs = obs[0]
        
    if len(obs) == 8:
        # Standard + distance: [dx, dy, dz, vx, vy, vz, fuel, dist]
        # Note: first 3 are RELATIVE position
        dx, dy, dz, vx, vy, vz, fuel, dist_val = obs
        # Convert to absolute assuming target is at [10, 0, 0]
        # This matches the training env's default target
        x = dx + 10.0
        y = dy
        z = dz
        return x, y, z, vx, vy, vz, fuel
    elif len(obs) == 7:
        # Standard: [x, y, z, vx, vy, vz, fuel]
        # Assuming these are relative too if coming from same env
        dx, dy, dz, vx, vy, vz, fuel = obs
        x = dx + 10.0
        y = dy
        z = dz
        return x, y, z, vx, vy, vz, fuel
    else:
        # Unknown format - safe fallback
        print(f"⚠️  Unknown obs size: {len(obs)}, returning zeros")
        return 0, 0, 0, 0, 0, 0, 100


def test_robustness_3d():
    """Test success rate across many seeds with detailed metrics."""
    print("=" * 60)
    print("3D ROBUSTNESS TEST")
    print("=" * 60)
    
    try:
        model = PPO.load("models/cubesat_3d_final.zip")
        print("✅ Loaded: cubesat_3d_final.zip")
    except:
        try:
            model = PPO.load("models/best_model.zip")
            print("✅ Loaded: best_model.zip")
        except:
            print("❌ Error: No model found. Train first with train_3d.py")
            return
    
    env = CubeSat3DEnv()
    
    # Check observation space
    obs, _ = env.reset()
    print(f"📊 Observation space: {len(obs)} dimensions")
    
    results = []
    
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        
        trajectory = []
        success = False
        final_dist = None
        final_speed = None
        steps = 0
        
        for step in range(300):
            x, y, z, vx, vy, vz, fuel = parse_observation(obs)
            
            # Distance from target [10, 0, 0]
            dist = np.linalg.norm([x-10, y, z])
            speed = np.linalg.norm([vx, vy, vz])
            
            trajectory.append({
                'step': step,
                'pos': (x, y, z),
                'dist': dist,
                'speed': speed,
                'fuel': fuel
            })
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps = step
            
            if terminated:
                success = True
                final_dist = info.get('distance_to_target', dist)
                final_speed = info.get('velocity', speed)
                break
            if truncated:
                final_dist = info.get('distance_to_target', dist)
                final_speed = info.get('velocity', speed)
                break
        
        results.append({
            'seed': seed,
            'success': success,
            'steps': steps,
            'final_dist': final_dist,
            'final_speed': final_speed,
            'trajectory': trajectory
        })
    
    # Analysis
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"RESULTS: 50 EPISODES")
    print(f"{'='*60}")
    print(f"Success rate: {len(successes)}/50 ({100*len(successes)/50:.1f}%)")
    
    if successes:
        avg_steps = np.mean([r['steps'] for r in successes])
        avg_dist = np.mean([r['final_dist'] for r in successes])
        avg_speed = np.mean([r['final_speed'] for r in successes])
        print(f"\nSuccessful episodes ({len(successes)}):")
        print(f"  Mean steps: {avg_steps:.1f}")
        print(f"  Mean final distance: {avg_dist:.3f}m")
        print(f"  Mean final speed: {avg_speed:.3f}m/s")
    
    if failures:
        print(f"\nFailed episodes ({len(failures)}):")
        for f in failures[:5]:
            print(f"  Seed {f['seed']}: dist={f['final_dist']:.2f}, "
                  f"speed={f['final_speed']:.2f}, steps={f['steps']}")
    
    # Best and worst
    if successes:
        best = min(successes, key=lambda x: x['final_dist'])
        print(f"\nBest episode (seed {best['seed']}):")
        print(f"  Final distance: {best['final_dist']:.4f}m")
        print(f"  Final speed: {best['final_speed']:.4f}m/s")
        print(f"  Steps: {best['steps']}")
    
    env.close()
    return results


def plot_3d_trajectory(seed=42):
    """Plot 3D trajectory for visualization."""
    print(f"\n{'='*60}")
    print(f"3D Trajectory Visualization (Seed {seed})")
    print(f"{'='*60}")
    
    try:
        model = PPO.load("models/cubesat_3d_final.zip")
    except:
        model = PPO.load("models/best_model.zip")
    
    # Force hard difficulty for visualization to see interesting trajectories
    env = CubeSat3DEnv()
    obs, _ = env.reset(seed=seed)
    
    xs, ys, zs = [], [], []
    distances = []
    speeds = []
    
    for step in range(300):
        x, y, z, vx, vy, vz, fuel = parse_observation(obs)
        dist = np.linalg.norm([x-10, y, z])
        speed = np.linalg.norm([vx, vy, vz])
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
        distances.append(dist)
        speeds.append(speed)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(xs, ys, zs, 'b-', linewidth=2, label='Trajectory')
    ax1.plot(xs[::3], ys[::3], zs[::3], 'bo', markersize=4)
    ax1.plot([0], [0], [0], 'go', markersize=10, label='Start')
    ax1.plot([10], [0], [0], 'r*', markersize=15, label='Target')
    
    # Success sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = 10 + 1.0 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0 + 1.0 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0 + 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='green')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Trajectory (Seed {seed})')
    ax1.legend()
    
    # XY projection
    ax2 = fig.add_subplot(222)
    ax2.plot(xs, ys, 'b-', linewidth=2)
    ax2.plot(0, 0, 'go', markersize=10, label='Start')
    ax2.plot(10, 0, 'r*', markersize=15, label='Target')
    circle = plt.Circle((10, 0), 1.0, color='r', fill=False, linestyle='--')
    ax2.add_patch(circle)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection (Top View)')
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend()
    
    # XZ projection
    ax3 = fig.add_subplot(223)
    ax3.plot(xs, zs, 'b-', linewidth=2)
    ax3.plot(0, 0, 'go', markersize=10, label='Start')
    ax3.plot(10, 0, 'r*', markersize=15, label='Target')
    circle = plt.Circle((10, 0), 1.0, color='r', fill=False, linestyle='--')
    ax3.add_patch(circle)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection (Side View)')
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.legend()
    
    # Distance profile
    ax4 = fig.add_subplot(224)
    ax4.plot(distances, 'g-', linewidth=2)
    ax4.axhline(y=1.0, color='r', linestyle='--', label='Success threshold')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Distance to target (m)')
    ax4.set_title('Distance Profile')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'3d_trajectory_seed{seed}.png', dpi=150)
    plt.show()
    print(f"💾 Saved to: 3d_trajectory_seed{seed}.png")
    print(f"Episode length: {len(xs)} steps")
    print(f"Success: {distances[-1] < 1.0 and speeds[-1] < 0.5}")


def measure_fuel_3d():
    """Measure fuel consumption in 3D."""
    print(f"\n{'='*60}")
    print("3D FUEL CONSUMPTION")
    print(f"{'='*60}")
    
    try:
        model = PPO.load("models/cubesat_3d_final.zip")
    except:
        model = PPO.load("models/best_model.zip")
    
    env = CubeSat3DEnv()
    
    fuel_used_list = []
    steps_list = []
    
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        initial_fuel = parse_observation(obs)[6]
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                final_fuel = parse_observation(obs)[6]
                fuel_used = initial_fuel - final_fuel
                fuel_used_list.append(fuel_used)
                steps_list.append(step + 1)
                break
    
    env.close()
    
    print(f"Episodes: {len(fuel_used_list)}")
    print(f"Mean steps: {np.mean(steps_list):.1f}")
    print(f"Mean fuel used: {np.mean(fuel_used_list):.2f}")
    print(f"Std fuel used: {np.std(fuel_used_list):.2f}")
    print(f"Min/Max fuel: {np.min(fuel_used_list):.2f} / {np.max(fuel_used_list):.2f}")
    print(f"Mean remaining: {100 - np.mean(fuel_used_list):.2f}%")
    print(f"Fuel per step: {np.mean(fuel_used_list)/np.mean(steps_list):.3f}")
    print(f"Fuel per meter: {np.mean(fuel_used_list)/10:.2f}")


def quick_diagnostic():
    """Check multiple seeds for consistency."""
    try:
        model = PPO.load("models/cubesat_3d_final.zip")
    except:
        try:
            model = PPO.load("models/best_model.zip")
        except:
            print("❌ No model found!")
            return
    
    env = CubeSat3DEnv()
    
    print("Quick diagnostic - 5 seeds:")
    for seed in [0, 1, 2, 42, 99]:
        obs, _ = env.reset(seed=seed)
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                status = "✅ SUCCESS" if terminated else "❌ FAILED"
                dist = info.get('distance_to_target', 'N/A')
                fuel = info.get('fuel_remaining', 'N/A')
                
                # Careful f-string typing
                if isinstance(dist, (int, float)):
                    dist_str = f"{dist:.2f}"
                else:
                    dist_str = str(dist)
                    
                if isinstance(fuel, (int, float)):
                    fuel_str = f"{fuel:.1f}"
                else:
                    fuel_str = str(fuel)
                
                print(f"Seed {seed:2d}: {status} at step {step:3d}, "
                      f"dist={dist_str}, fuel={fuel_str}")
                break
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("3D CubeSat Navigation - Test Suite")
    print("=" * 60)
    
    # Run diagnostic first to check observation space
    quick_diagnostic()
    
    # Run full tests
    print("\n" + "="*60)
    results = test_robustness_3d()
    
    if results:
        print("\n" + "="*60)
        measure_fuel_3d()
        
        # Visualize one trajectory
        print("\n" + "="*60)
        print("Generating 3D visualization...")
        plot_3d_trajectory(seed=42)
    
    print(f"\n{'='*60}")
    print("✅ All tests complete!")
    print("=" * 60)