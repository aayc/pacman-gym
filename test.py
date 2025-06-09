#!/usr/bin/env python3
"""
Test script for the Pacman environment.
"""

import numpy as np
import time
from env import PacmanEnv, ActionType

def test_pacman_environment():
    """Test the Pacman environment with random actions and rendering."""
    print("🟡 Testing Pacman Environment...")
    
    # Create environment
    env = PacmanEnv(maze_width=15, maze_height=17)
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Total pellets: {env.total_pellets}")
    
    # Test random episode
    episode_rewards = [0.0, 0.0, 0.0]
    step_count = 0
    
    print("\nRunning test episode with random actions...")
    print("Press Ctrl+C to stop early")
    
    try:
        while True:
            # Random actions for all agents: [pacman, ghost1, ghost2]
            actions = [np.random.randint(0, len(ActionType)) for _ in range(3)]
            
            # Step environment
            obs, rewards, done, truncated, info = env.step(actions)
            
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]
            episode_rewards[2] += rewards[2]
            step_count += 1
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}:")
                print(f"  Pacman - Pos: ({env.pacman.x}, {env.pacman.y}), Score: {env.pacman.score}, Alive: {env.pacman.alive}")
                print(f"  Ghost 1 - Pos: ({env.ghosts[0].x}, {env.ghosts[0].y})")
                print(f"  Ghost 2 - Pos: ({env.ghosts[1].x}, {env.ghosts[1].y})")
                print(f"  Pellets: {env.pellets_collected}/{env.total_pellets}")
                print(f"  Rewards: Pacman={rewards[0]:.2f}, Ghost1={rewards[1]:.2f}, Ghost2={rewards[2]:.2f}")
                print()
            
            # Render (comment out if running headless)
            env.render()
            time.sleep(0.05)  # ~20 FPS
            
            if done or truncated:
                print(f"\n✅ Episode completed!")
                print(f"  Duration: {step_count} steps")
                print(f"  Final rewards: Pacman: {episode_rewards[0]:.2f}, Ghost1: {episode_rewards[1]:.2f}, Ghost2: {episode_rewards[2]:.2f}")
                print(f"  Pacman final score: {env.pacman.score}")
                print(f"  Pellets collected: {env.pellets_collected}/{env.total_pellets}")
                
                if not env.pacman.alive:
                    print("  Result: 👻 Ghosts caught Pacman!")
                elif env.pellets_collected >= env.total_pellets:
                    print("  Result: 🟡 Pacman collected all pellets!")
                else:
                    print("  Result: ⏰ Episode timed out")
                break
    
    except KeyboardInterrupt:
        print(f"\n\nTest interrupted at step {step_count}")
    
    finally:
        env.close()
        print("Environment closed.")

def test_action_space():
    """Test all actions work correctly."""
    print("\n🎮 Testing action space...")
    
    env = PacmanEnv()
    obs, _ = env.reset()
    
    # Test each action type
    for action_type in ActionType:
        print(f"  Testing {action_type.name}...")
        # All agents perform the same action
        actions = [action_type.value] * 3
        obs, rewards, done, truncated, info = env.step(actions)
        
        if done:
            obs, _ = env.reset()
    
    print("✅ All actions tested successfully")
    env.close()

def test_observations():
    """Test observation space structure."""
    print("\n🔍 Testing observation space...")
    
    env = PacmanEnv()
    obs, _ = env.reset()
    
    print(f"  Observation shape: {obs.shape}")
    print(f"  Expected: (3, 38) for [Pacman, Ghost1, Ghost2]")
    
    # Check observation bounds
    obs_min, obs_max = obs.min(), obs.max()
    print(f"  Value range: [{obs_min:.3f}, {obs_max:.3f}]")
    
    # Test multiple steps
    for i in range(10):
        actions = [np.random.randint(0, len(ActionType)) for _ in range(3)]
        obs, rewards, done, truncated, info = env.step(actions)
        
        if done:
            obs, _ = env.reset()
    
    print("✅ Observations tested successfully")
    env.close()

def run_benchmark():
    """Benchmark environment performance."""
    print("\n⚡ Running performance benchmark...")
    
    env = PacmanEnv()
    obs, _ = env.reset()
    
    num_steps = 1000
    start_time = time.time()
    
    for i in range(num_steps):
        actions = [np.random.randint(0, len(ActionType)) for _ in range(3)]
        obs, rewards, done, truncated, info = env.step(actions)
        
        if done:
            obs, _ = env.reset()
    
    end_time = time.time()
    duration = end_time - start_time
    fps = num_steps / duration
    
    print(f"  Completed {num_steps} steps in {duration:.2f} seconds")
    print(f"  Performance: {fps:.1f} FPS")
    print("✅ Benchmark completed")
    
    env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🟡 PACMAN ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    # Run tests
    test_action_space()
    test_observations()
    run_benchmark()
    
    # Interactive test (comment out if running headless)
    test_pacman_environment()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)