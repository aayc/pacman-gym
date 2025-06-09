#!/usr/bin/env python3
"""
Quick training script for Pacman with optimized parameters for faster learning.
"""

import argparse

from env import PacmanEnv
from train import PacmanPPOTrainer, plot_pacman_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick Pacman PPO training with optimized parameters"
    )
    parser.add_argument("--show", action="store_true", help="Show visualization")
    parser.add_argument(
        "--timesteps", type=int, default=150000, help="Training timesteps"
    )

    args = parser.parse_args()

    print("🟡 Quick Pacman Training Mode - Optimized for Fast Learning")
    print("=" * 60)

    # Create smaller environment for faster learning
    env = PacmanEnv(maze_width=15, maze_height=17)

    # Create trainer with optimized hyperparameters for Pacman
    trainer = PacmanPPOTrainer(
        env,
        learning_rate=2e-3,  # Higher learning rate for faster learning
        gamma=0.95,  # Lower discount for more immediate rewards
        lambda_gae=0.9,  # GAE lambda
        clip_epsilon=0.2,  # PPO clip
        epochs_per_update=3,  # Fewer epochs per update
        batch_size=512,  # Large batch size for stability
        value_loss_coef=0.5,
        entropy_coef=0.1,  # High entropy for exploration
        render=args.show,
        render_freq=10 if args.show else 1,
    )

    try:
        print(f"Training for {args.timesteps:,} timesteps...")
        print("Pacman Optimizations:")
        print("- Smaller maze (15x17)")
        print("- Shorter episodes (2000 steps max)")
        print("- Higher learning rate (5e-4)")
        print("- Better reward shaping for pellet collection")
        print("- Multi-agent cooperative/competitive learning")
        print("- Higher entropy for exploration")
        print("=" * 60)

        # Train
        episode_rewards, pacman_scores, pellets_collected = trainer.train(
            total_timesteps=args.timesteps,
            update_frequency=1024,  # Smaller update frequency
            save_frequency=50000,
        )

        # Plot results
        plot_pacman_results(episode_rewards, pacman_scores, pellets_collected)

        # Save models
        trainer.save_models("quick_trained_models")
        print("🎉 Quick training complete! Models saved as 'quick_trained_models.pth'")

        # Print quick analysis
        if episode_rewards:
            recent_episodes = (
                episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
            )
            recent_scores = (
                pacman_scores[-20:] if len(pacman_scores) >= 20 else pacman_scores
            )
            recent_pellets = (
                pellets_collected[-20:]
                if len(pellets_collected) >= 20
                else pellets_collected
            )

            print("\n📊 Quick Analysis (Last 20 Episodes):")
            if recent_scores:
                print(
                    f"Avg Pacman Score: {sum(recent_scores) / len(recent_scores):.1f}"
                )
            if recent_pellets:
                print(
                    f"Avg Pellets Collected: {sum(recent_pellets) / len(recent_pellets):.1f}"
                )

            pacman_rewards = [r[0] for r in recent_episodes]
            ghost_rewards = [(r[1] + r[2]) / 2 for r in recent_episodes]
            print(f"Avg Pacman Reward: {sum(pacman_rewards) / len(pacman_rewards):.1f}")
            print(f"Avg Ghost Reward: {sum(ghost_rewards) / len(ghost_rewards):.1f}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        trainer.save_models("quick_interrupted_models")
        print("Models saved as 'quick_interrupted_models.pth'")

    finally:
        env.close()


if __name__ == "__main__":
    main()
