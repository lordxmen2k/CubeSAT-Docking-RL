"""
Train 3D CubeSat Navigation Model (Fixed)
Run with: python train_3d_fixed.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os


class CubeSat3DEnv(gym.Env):
    """
    3D CubeSat navigation environment with orbital mechanics.
    Target: Learn fuel-efficient rendezvous with gravity and drag.
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, difficulty='normal'):
        super().__init__()
        
        self.dt = 1.0
        self.max_thrust = 0.15  # Increased for better control authority
        self.max_fuel = 100.0
        self.fuel_burn_rate = 0.5  # Reduced from 1.0 (more fuel efficient)
        
        # Orbital parameters (critical for stability)
        self.gravity = 0.02  # Weak central gravity toward origin
        self.drag = 0.01  # Velocity damping (stabilizes oscillations)
        
        self.target = np.array([15.0, 0.0, 0.0])
        self.success_dist = 1.0
        self.success_speed = 0.5
        
        # Curriculum: easy/medium/hard
        self.difficulty = difficulty
        self.init_velocity_scale = 0.1 if difficulty == 'easy' else 0.3
        
        # Action space: 3D thrust vector
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation: [dx, dy, dz, vx, vy, vz, fuel, dist_to_target]
        # Using relative position is more stable than absolute coordinates
        self.observation_space = spaces.Box(
            low=np.array([-30.0, -30.0, -30.0, -5.0, -5.0, -5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 30.0, 30.0, 5.0, 5.0, 5.0, self.max_fuel, 40.0], dtype=np.float32),
            dtype=np.float32
        )   
        
        self.render_mode = render_mode
        self.step_count = 0
        self.max_steps = 300  # Reduced from 300 (force faster solutions)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize target position slightly for robustness
        if self.difficulty == 'hard':
            self.target = np.array([
                15.0 + self.np_random.uniform(-2.0, 2.0),
                self.np_random.uniform(-2.0, 2.0),
                self.np_random.uniform(-2.0, 2.0)
            ])
        else:
            self.target = np.array([15.0, 0.0, 0.0])
        
        # Start at origin with small random velocity
        pos = np.array([0.0, 0.0, 0.0])
        vel = self.np_random.uniform(
            -self.init_velocity_scale, 
            self.init_velocity_scale, 
            size=3
        )
        
        self.state = {
            'pos': pos.astype(np.float32),
            'vel': vel.astype(np.float32),
            'fuel': self.max_fuel
        }
        
        self.step_count = 0
        self.prev_dist = np.linalg.norm(pos - self.target)
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        
        # Deadzone: very small thrusts are set to zero (coasting)
        # This encourages bang-bang control and saves fuel
        deadzone = 0.05
        action[np.abs(action) < deadzone] = 0.0
        
        pos = self.state['pos'].copy()
        vel = self.state['vel'].copy()
        fuel = self.state['fuel']
        
        # Calculate thrust with fuel constraints
        thrust = action * self.max_thrust
        thrust_magnitude = np.linalg.norm(action)
        
        # Fuel consumption proportional to thrust (quadratic for realism)
        fuel_consumed = (thrust_magnitude ** 2) * self.fuel_burn_rate * self.dt
        
        if fuel_consumed > fuel:
            # Scale thrust if not enough fuel
            scale = fuel / (fuel_consumed + 1e-8)
            thrust *= scale
            fuel_consumed = fuel
        
        # Physics integration with gravity and drag
        # Gravity pulls toward origin (optional: change to pull toward target)
        r = np.linalg.norm(pos)
        if r > 0.1:
            gravity_vec = -self.gravity * pos / r  # Central gravity
        else:
            gravity_vec = np.zeros(3)
        
        # Drag force (damps oscillations)
        drag_force = -self.drag * vel
        
        # Acceleration
        accel = thrust + gravity_vec + drag_force
        
        # Semi-implicit Euler integration (more stable)
        vel_new = vel + accel * self.dt
        pos_new = pos + vel_new * self.dt  # Use new velocity for position
        
        fuel_new = max(0.0, fuel - fuel_consumed)
        
        # Update state
        self.state['pos'] = pos_new.astype(np.float32)
        self.state['vel'] = vel_new.astype(np.float32)
        self.state['fuel'] = fuel_new
        
        # Calculate metrics
        dist = np.linalg.norm(pos_new - self.target)
        speed = np.linalg.norm(vel_new)
        
        # Reward calculation
        reward = self._calculate_reward(pos_new, vel_new, fuel_new, dist, speed, thrust_magnitude)
        
        # Termination conditions
        success = (dist < self.success_dist) and (speed < self.success_speed)
        terminated = bool(success)
        
        # Truncation conditions
        out_of_bounds = np.any(np.abs(pos_new) > 35.0)
        fuel_empty = fuel_new <= 0.0
        timeout = self.step_count >= self.max_steps
        
        truncated = bool(out_of_bounds or fuel_empty or timeout)
        
        info = {
            'distance_to_target': float(dist),
            'fuel_remaining': float(fuel_new),
            'velocity': float(speed),
            'success': success,
            'steps': self.step_count
        }
        
        self.prev_dist = dist
        
        return self._get_obs(), float(reward), terminated, truncated, info
    
    def _get_obs(self):
        """Return observation vector."""
        pos = self.state['pos']
        vel = self.state['vel']
        fuel = self.state['fuel']
        
        # Relative position to target (more meaningful than absolute)
        rel_pos = pos - self.target
        
        dist = np.linalg.norm(rel_pos)
        
        obs = np.concatenate([
            rel_pos,  # dx, dy, dz
            vel,      # vx, vy, vz
            [fuel],   # remaining fuel
            [dist]    # distance to target (useful feature)
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, pos, vel, fuel, dist, speed, thrust_mag):
        """
        Reward shaping for efficient rendezvous.
        Key principles:
        1. Progress toward target (dense reward)
        2. Fuel efficiency (punish unnecessary thrust)
        3. Terminal velocity matching (soft docking)
        4. Survival bonus (encourage fuel conservation)
        """
        
        # 1. Progress reward: improvement in distance (most important)
        progress = self.prev_dist - dist
        reward = progress * 10.0  # Scale up for significance
        
        # 2. Distance penalty (shaped: closer = exponentially better)
        reward -= dist * 0.5
        
        # 3. Fuel efficiency: heavy penalty for thrust
        # Quadratic penalty encourages coasting
        reward -= (thrust_mag ** 2) * 2.0
        
        # 4. Time penalty (encourage speed)
        reward -= 0.1
        
        # 5. Velocity alignment: prefer velocity toward target
        if dist > 0.1:
            direction_to_target = (self.target - pos) / dist
            velocity_alignment = np.dot(vel, direction_to_target)
            # Reward moving toward target, penalize moving away
            reward += velocity_alignment * 2.0
        
        # 6. Terminal guidance: slow down near target
        # --- REWARD SHAPING TWEAK ---
        if dist < 5.0:
            # Tighten the ideal speed profile
            # Old: ideal_speed = dist * 0.2 
            ideal_speed = dist * 0.1  # At 1m, target 0.1 m/s instead of 0.2
            speed_error = abs(speed - ideal_speed)
            reward -= speed_error * 5.0 # Increased penalty from 2.0 to 5.0
            
            # Add a "Speed Limit" hard penalty
            if speed > 0.5:
                reward -= 10.0 # Heavy penalty for approaching too fast
            
            # Bonus for being close AND slow
            if dist < 2.0 and speed < 1.0:
                reward += (2.0 - dist) * 5.0
        
        # 7. Success bonus with fuel efficiency
        if dist < self.success_dist and speed < self.success_speed:
            reward += 500.0  # Big bonus
            # Extra reward for remaining fuel (efficiency matters)
            reward += fuel * 5.0
        
        # 8. Soft penalty for high Y/Z deviation (keep aligned)
        lateral_deviation = np.sqrt(pos[1]**2 + pos[2]**2)
        reward -= lateral_deviation * 0.1
        
        return reward
    
    def render(self):
        pass
    
    def close(self):
        pass


def make_env(rank=0, difficulty='normal'):
    """Factory function for vectorized environments."""
    def _init():
        env = CubeSat3DEnv(difficulty=difficulty)
        env = Monitor(env)
        return env
    return _init


def main():
    """Main training function with curriculum learning."""
    print("=" * 60)
    print("3D CubeSat Navigation - Training (Fixed)")
    print("=" * 60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Phase 1: Train on easy mode (deterministic target, low initial velocity)
    print("\n" + "=" * 60)
    print("PHASE 1: Easy Curriculum")
    print("=" * 60)
    
    env = CubeSat3DEnv(difficulty='easy')
    check_env(env, warn=True)
    
    eval_env = CubeSat3DEnv(difficulty='easy')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/phase1/',
        log_path='./logs/phase1/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log="./tensorboard_3d/"
    )
    
    print("\nTraining Phase 1 (200k steps)...")
    model.learn(total_timesteps=200000, callback=eval_callback)
    model.save("models/cubesat_3d_phase1")
    
    # Phase 2: Train on normal mode
    print("\n" + "=" * 60)
    print("PHASE 2: Normal Curriculum")
    print("=" * 60)
    
    env = CubeSat3DEnv(difficulty='normal')
    eval_env = CubeSat3DEnv(difficulty='normal')
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/phase2/',
        log_path='./logs/phase2/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    # Load from phase 1
    model.set_env(env)
    
    print("\nTraining Phase 2 (300k steps)...")
    model.learn(total_timesteps=300000, callback=eval_callback)
    model.save("models/cubesat_3d_phase2")
    
    # Phase 3: Fine-tune with hard difficulty
    print("\n" + "=" * 60)
    print("PHASE 3: Hard Curriculum")
    print("=" * 60)
    
    env = CubeSat3DEnv(difficulty='hard')
    eval_env = CubeSat3DEnv(difficulty='hard')
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    model.set_env(env)
    
    # Lower learning rate for fine-tuning
    model.learning_rate = 1e-4
    
    print("\nTraining Phase 3 (200k steps)...")
    model.learn(total_timesteps=200000, callback=eval_callback)
    
    # Save final
    model.save("models/cubesat_3d_final")
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Models saved:")
    print("  - models/cubesat_3d_phase1.zip (easy)")
    print("  - models/cubesat_3d_phase2.zip (normal)")
    print("  - models/cubesat_3d_final.zip (hard)")
    print("  - models/best_model.zip (best from final phase)")
    print("=" * 60)
    
    # Final evaluation
    print("\nFinal Evaluation (20 episodes)...")
    successes = 0
    total_fuel = []
    total_steps = []
    
    for seed in range(20):
        obs, _ = eval_env.reset(seed=seed)
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            step_count += 1
            done = terminated or truncated
            
            if terminated:
                successes += 1
                total_fuel.append(info['fuel_remaining'])
                total_steps.append(step_count)
                break
    
    print(f"\nResults: {successes}/20 successes ({successes*5}%)")
    if successes > 0:
        print(f"Mean fuel remaining: {np.mean(total_fuel):.1f}")
        print(f"Mean steps: {np.mean(total_steps):.1f}")
    
    eval_env.close()


if __name__ == "__main__":
    main()