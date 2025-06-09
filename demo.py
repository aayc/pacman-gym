#!/usr/bin/env python3
"""
Demo script to watch trained Pacman agents play.
"""

import argparse
import time
from typing import List, Tuple

import torch

from env import PacmanEnv
from train import PacmanPPOAgent


def load_trained_pacman_agents(
    model_path: str, device: torch.device
) -> Tuple[PacmanPPOAgent, List[PacmanPPOAgent]]:
    """Load trained Pacman agents from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create agents with same architecture as training
    pacman_agent = PacmanPPOAgent(obs_size=47, action_size=5, hidden_size=256).to(
        device
    )
    ghost_agents = [
        PacmanPPOAgent(obs_size=47, action_size=5, hidden_size=256).to(device),
        PacmanPPOAgent(obs_size=47, action_size=5, hidden_size=256).to(device),
    ]

    # Load trained weights
    pacman_agent.load_state_dict(checkpoint["pacman_agent_state_dict"])
    for i, ghost_agent in enumerate(ghost_agents):
        ghost_agent.load_state_dict(checkpoint[f"ghost_{i}_agent_state_dict"])

    # Set to evaluation mode
    pacman_agent.eval()
    for ghost_agent in ghost_agents:
        ghost_agent.eval()

    return pacman_agent, ghost_agents


def run_pacman_demo_episode(
    env: PacmanEnv,
    pacman_agent: PacmanPPOAgent,
    ghost_agents: List[PacmanPPOAgent],
    device: torch.device,
    max_steps: int = 2000,
    speed: float = 0.1,
) -> Tuple[List[float], int]:
    """Run a single demo episode with trained agents."""
    obs, _ = env.reset()
    episode_rewards = [0.0, 0.0, 0.0]
    step_count = 0

    print("\n🟡 Starting Pacman demo episode...")
    print("Close the pygame window or press Ctrl+C to stop")

    while step_count < max_steps:
        obs_tensor = torch.FloatTensor(obs).to(device)

        # Get actions from trained agents
        with torch.no_grad():
            pacman_action, _, _, _ = pacman_agent.get_action_and_value(obs_tensor[0])
            ghost_actions = []
            for ghost_agent in ghost_agents:
                action, _, _, _ = ghost_agent.get_action_and_value(
                    obs_tensor[len(ghost_actions) + 1]
                )
                ghost_actions.append(action)

        actions = [pacman_action.cpu().numpy()] + [
            action.cpu().numpy() for action in ghost_actions
        ]

        # Step environment
        obs, rewards, done, truncated, _ = env.step(actions)

        episode_rewards[0] += rewards[0]
        episode_rewards[1] += rewards[1]
        episode_rewards[2] += rewards[2]
        step_count += 1

        # Render
        env.render()
        time.sleep(speed)  # Use configurable speed

        # Print status every 100 steps
        if step_count % 100 == 0:
            print(
                f"Step {step_count}: Pacman Score={env.pacman.score}, Pellets={env.pellets_collected}/{env.total_pellets}, Alive={env.pacman.alive}"
            )

        if done or truncated:
            print(f"\n🎮 Episode finished after {step_count} steps!")
            print(
                f"Final rewards: Pacman: {episode_rewards[0]:.1f}, Ghost1: {episode_rewards[1]:.1f}, Ghost2: {episode_rewards[2]:.1f}"
            )
            print(f"Pacman final score: {env.pacman.score}")
            print(f"Pellets collected: {env.pellets_collected}/{env.total_pellets}")

            if not env.pacman.alive:
                print("👻 Ghosts caught Pacman!")
            elif env.pellets_collected >= env.total_pellets:
                print("🟡 Pacman collected all pellets!")
            else:
                print("⏰ Episode timed out")

            return episode_rewards, step_count

    print(f"\n⏰ Episode reached max steps ({max_steps})")
    return episode_rewards, step_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo trained Pacman agents")
    parser.add_argument(
        "--model",
        type=str,
        default="final_models.pth",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument("--maze-width", type=int, default=19, help="Maze width")
    parser.add_argument("--maze-height", type=int, default=21, help="Maze height")
    parser.add_argument(
        "--speed", type=float, default=0.1, help="Time delay between steps (seconds)"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize env to None for proper cleanup
    env = None

    try:
        # Load trained agents
        print(f"Loading trained Pacman agents from {args.model}...")
        pacman_agent, ghost_agents = load_trained_pacman_agents(args.model, device)
        print("✅ Agents loaded successfully!")

        # Create environment
        env = PacmanEnv(maze_width=args.maze_width, maze_height=args.maze_height)

        # Run demo episodes
        total_rewards = [0.0, 0.0, 0.0]
        total_scores = 0
        total_pellets = 0
        pacman_wins = 0
        ghost_wins = 0
        timeouts = 0

        print(f"\n🎮 Running {args.episodes} Pacman demo episodes...")
        print("=" * 60)

        for episode in range(args.episodes):
            print(f"\n🟡 Episode {episode + 1}/{args.episodes}")

            try:
                episode_rewards, steps = run_pacman_demo_episode(
                    env, pacman_agent, ghost_agents, device, speed=args.speed
                )

                total_rewards[0] += episode_rewards[0]
                total_rewards[1] += episode_rewards[1]
                total_rewards[2] += episode_rewards[2]
                total_scores += env.pacman.score
                total_pellets += env.pellets_collected

                # Track outcomes
                if not env.pacman.alive:
                    ghost_wins += 1
                elif env.pellets_collected >= env.total_pellets:
                    pacman_wins += 1
                else:
                    timeouts += 1

                # Wait between episodes
                if episode < args.episodes - 1:
                    print("Starting next episode in 3 seconds...")
                    time.sleep(3)

            except KeyboardInterrupt:
                print(f"\n⏹️ Demo interrupted at episode {episode + 1}")
                break

        # Print summary
        episodes_played = episode + 1
        print("\n" + "=" * 60)
        print("🟡 PACMAN DEMO SUMMARY")
        print("=" * 60)
        print(f"Episodes played: {episodes_played}")
        print(f"Pacman wins: {pacman_wins} ({pacman_wins / episodes_played:.1%})")
        print(f"Ghost wins: {ghost_wins} ({ghost_wins / episodes_played:.1%})")
        print(f"Timeouts: {timeouts} ({timeouts / episodes_played:.1%})")
        print(f"Avg Pacman score: {total_scores / episodes_played:.1f}")
        print(f"Avg pellets collected: {total_pellets / episodes_played:.1f}")
        print(
            f"Avg rewards - Pacman: {total_rewards[0] / episodes_played:.1f}, Ghost1: {total_rewards[1] / episodes_played:.1f}, Ghost2: {total_rewards[2] / episodes_played:.1f}"
        )

    except FileNotFoundError:
        print(f"❌ Error: Model file '{args.model}' not found!")
        print("Make sure you've trained the agents first using train.py")
        return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    finally:
        if env is not None:
            env.close()

    return 0


if __name__ == "__main__":
    exit(main())
