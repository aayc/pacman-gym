import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import pygame
from env import PacmanEnv
from typing import List, Tuple, Dict, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

class PacmanPPOAgent(nn.Module):
    def __init__(self, obs_size: int = 47, action_size: int = 5, hidden_size: int = 256) -> None:
        super(PacmanPPOAgent, self).__init__()
        
        # Larger, simpler feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy) - simpler
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Critic head (value function) - simpler
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

class PacmanPPOTrainer:
    def __init__(self, env: PacmanEnv, learning_rate: float = 1e-3, gamma: float = 0.99, lambda_gae: float = 0.95, 
                 clip_epsilon: float = 0.2, epochs_per_update: int = 4, batch_size: int = 256, 
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.05, render: bool = False, render_freq: int = 1) -> None:
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get observation and action dimensions
        obs_shape = env.observation_space.shape
        self.obs_size = obs_shape[1]  # Size per agent
        self.action_size = env.action_space.nvec[0]  # Actions per agent
        self.num_agents = obs_shape[0]  # Total agents (Pacman + Ghosts)
        
        # Create agents (Pacman + 2 Ghosts)
        self.pacman_agent = PacmanPPOAgent(self.obs_size, self.action_size).to(self.device)
        self.ghost_agents = [
            PacmanPPOAgent(self.obs_size, self.action_size).to(self.device)
            for _ in range(self.num_agents - 1)
        ]
        
        # Optimizers
        self.pacman_optimizer = optim.Adam(self.pacman_agent.parameters(), lr=learning_rate)
        self.ghost_optimizers = [
            optim.Adam(agent.parameters(), lr=learning_rate)
            for agent in self.ghost_agents
        ]
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.pacman_scores = []
        self.pellets_collected = []
        self.ghost_catches = []
        
        # Loss tracking for all agents
        self.actor_losses = [[] for _ in range(self.num_agents)]
        self.critic_losses = [[] for _ in range(self.num_agents)]
        self.entropy_losses = [[] for _ in range(self.num_agents)]
        self.value_estimates = [[] for _ in range(self.num_agents)]
        self.policy_ratios = [[] for _ in range(self.num_agents)]
        self.explained_variance = [[] for _ in range(self.num_agents)]
        
        # Rendering options
        self.render = render
        self.render_freq = render_freq
        self.render_episode_count = 0
        
        # Rich console
        self.console = Console()
        
    def collect_trajectories(self, num_steps: int = 2048) -> Dict[str, Dict[str, List[Any]]]:
        """Collect trajectories for all agents."""
        trajectories = {
            f'agent_{i}': {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 
                          'values': [], 'log_probs': []}
            for i in range(self.num_agents)
        }
        
        obs, _ = self.env.reset()
        episode_rewards = [0.0] * self.num_agents
        episode_length = 0
        
        for _ in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # Get actions from all agents
            with torch.no_grad():
                # Pacman action
                pacman_action, pacman_log_prob, _, pacman_value = self.pacman_agent.get_action_and_value(obs_tensor[0])
                
                # Ghost actions
                ghost_actions = []
                ghost_log_probs = []
                ghost_values = []
                
                for i, ghost_agent in enumerate(self.ghost_agents):
                    action, log_prob, _, value = ghost_agent.get_action_and_value(obs_tensor[i + 1])
                    ghost_actions.append(action)
                    ghost_log_probs.append(log_prob)
                    ghost_values.append(value)
            
            # Combine actions
            actions = [pacman_action.cpu().numpy()] + [action.cpu().numpy() for action in ghost_actions]
            
            # Step environment
            next_obs, rewards, done, truncated, _ = self.env.step(actions)
            
            # Render if requested
            should_render = (self.render and 
                           self.render_episode_count % self.render_freq == 0)
            if should_render:
                self.env.render()
                time.sleep(0.05)  # Slower for Pacman visibility
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.render = False
                        break
            
            # Store trajectories for Pacman
            trajectories['agent_0']['observations'].append(obs[0])
            trajectories['agent_0']['actions'].append(pacman_action.cpu().numpy())
            trajectories['agent_0']['rewards'].append(rewards[0])
            trajectories['agent_0']['dones'].append(done)
            trajectories['agent_0']['values'].append(pacman_value.cpu().numpy())
            trajectories['agent_0']['log_probs'].append(pacman_log_prob.cpu().numpy())
            
            # Store trajectories for Ghosts
            for i, (action, log_prob, value) in enumerate(zip(ghost_actions, ghost_log_probs, ghost_values)):
                agent_key = f'agent_{i + 1}'
                trajectories[agent_key]['observations'].append(obs[i + 1])
                trajectories[agent_key]['actions'].append(action.cpu().numpy())
                trajectories[agent_key]['rewards'].append(rewards[i + 1])
                trajectories[agent_key]['dones'].append(done)
                trajectories[agent_key]['values'].append(value.cpu().numpy())
                trajectories[agent_key]['log_probs'].append(log_prob.cpu().numpy())
            
            # Update episode statistics
            for i in range(self.num_agents):
                episode_rewards[i] += rewards[i]
            episode_length += 1
            
            obs = next_obs
            
            if done or truncated:
                # Store episode statistics
                self.episode_rewards.append(episode_rewards.copy())
                self.episode_lengths.append(episode_length)
                self.pacman_scores.append(self.env.pacman.score)
                self.pellets_collected.append(self.env.pellets_collected)
                
                # Count ghost catches
                ghost_catch_count = sum(1 for ghost in self.env.ghosts if hasattr(ghost, 'caught_pacman') and ghost.caught_pacman)
                self.ghost_catches.append(ghost_catch_count)
                
                self.render_episode_count += 1
                obs, _ = self.env.reset()
                episode_rewards = [0.0] * self.num_agents
                episode_length = 0
        
        return trajectories
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_val = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update_agent(self, agent: PacmanPPOAgent, optimizer: torch.optim.Optimizer, trajectory: Dict[str, List[Any]]) -> Dict[str, float]:
        """Update a single agent and return statistics."""
        observations = np.array(trajectory['observations'])
        actions = np.array(trajectory['actions'])
        old_log_probs = np.array(trajectory['log_probs'])
        rewards = np.array(trajectory['rewards'])
        dones = np.array(trajectory['dones'])
        values = np.array(trajectory['values']).flatten()
        
        if len(observations) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy_loss': 0, 
                   'policy_ratio': 1, 'value_estimate': 0, 'explained_variance': 0}
        
        # Get next value for GAE computation
        last_obs = torch.FloatTensor(observations[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = agent(last_obs)
            next_value = next_value.cpu().numpy().flatten()[0]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Convert to tensors
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get all value predictions for explained variance calculation
        with torch.no_grad():
            _, _, _, all_values = agent.get_action_and_value(observations, actions)
            all_values = all_values.flatten()
        
        # Update policy and collect statistics
        dataset_size = len(observations)
        epoch_stats = {'actor_loss': [], 'critic_loss': [], 'entropy_loss': [], 
                      'policy_ratio': [], 'value_estimate': []}
        
        for _ in range(self.epochs_per_update):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = agent.get_action_and_value(batch_obs, batch_actions)
                
                # Compute ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.flatten(), batch_returns)
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Collect statistics
                epoch_stats['actor_loss'].append(actor_loss.item())
                epoch_stats['critic_loss'].append(value_loss.item())
                epoch_stats['entropy_loss'].append(entropy_loss.item())
                epoch_stats['policy_ratio'].append(ratios.mean().item())
                epoch_stats['value_estimate'].append(values.mean().item())
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # Compute explained variance
        explained_var = 1 - torch.var(returns - all_values) / (torch.var(returns) + 1e-8)
        
        return {
            'actor_loss': np.mean(epoch_stats['actor_loss']),
            'critic_loss': np.mean(epoch_stats['critic_loss']),
            'entropy_loss': np.mean(epoch_stats['entropy_loss']),
            'policy_ratio': np.mean(epoch_stats['policy_ratio']),
            'value_estimate': np.mean(epoch_stats['value_estimate']),
            'explained_variance': explained_var.item()
        }
    
    def create_training_layout(self) -> Layout:
        """Create rich layout for training display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="progress", size=5)
        )
        
        layout["main"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="losses", ratio=1)
        )
        
        return layout
    
    def create_stats_table(self, timesteps: int, update_count: int, elapsed_time: float, fps: float) -> Table:
        """Create statistics table."""
        table = Table(title="🟡 Pacman Training Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Pacman 🟡", style="yellow")
        table.add_column("Ghost 1 🔴", style="red")
        table.add_column("Ghost 2 🟣", style="magenta")
        
        # Recent episode statistics
        recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
        if recent_rewards:
            avg_rewards = [np.mean([r[i] for r in recent_rewards]) for i in range(self.num_agents)]
            avg_scores = np.mean(self.pacman_scores[-20:]) if self.pacman_scores else 0
            avg_pellets = np.mean(self.pellets_collected[-20:]) if self.pellets_collected else 0
            avg_catches = np.mean(self.ghost_catches[-20:]) if self.ghost_catches else 0
            
            table.add_row("Avg Reward (20ep)", f"{avg_rewards[0]:.1f}", f"{avg_rewards[1]:.1f}", f"{avg_rewards[2]:.1f}")
            table.add_row("Pacman Score", f"{avg_scores:.1f}", "-", "-")
            table.add_row("Pellets Eaten", f"{avg_pellets:.1f}", "-", "-")
            table.add_row("Ghost Catches", "-", f"{avg_catches:.1f}", f"{avg_catches:.1f}")
            table.add_row("Episodes Played", str(len(self.episode_rewards)), str(len(self.episode_rewards)), str(len(self.episode_rewards)))
        
        table.add_row("", "", "", "")
        table.add_row("Timesteps", f"{timesteps:,}", f"{timesteps:,}", f"{timesteps:,}")
        table.add_row("Updates", str(update_count), str(update_count), str(update_count))
        table.add_row("Training Time", f"{elapsed_time:.0f}s", f"{elapsed_time:.0f}s", f"{elapsed_time:.0f}s")
        table.add_row("FPS", f"{fps:.1f}", f"{fps:.1f}", f"{fps:.1f}")
        
        return table
    
    def create_losses_table(self) -> Table:
        """Create losses table."""
        table = Table(title="📉 Training Losses", box=box.ROUNDED)
        table.add_column("Loss Type", style="cyan", no_wrap=True)
        table.add_column("Pacman 🟡", style="yellow")
        table.add_column("Ghost 1 🔴", style="red")
        table.add_column("Ghost 2 🟣", style="magenta")
        
        all_have_losses = all(len(losses) > 0 for losses in self.actor_losses)
        
        if all_have_losses:
            recent_actor = [np.mean(losses[-5:]) for losses in self.actor_losses]
            recent_critic = [np.mean(losses[-5:]) for losses in self.critic_losses]
            recent_entropy = [np.mean(losses[-5:]) for losses in self.entropy_losses]
            
            table.add_row("Actor Loss", f"{recent_actor[0]:.4f}", f"{recent_actor[1]:.4f}", f"{recent_actor[2]:.4f}")
            table.add_row("Critic Loss", f"{recent_critic[0]:.4f}", f"{recent_critic[1]:.4f}", f"{recent_critic[2]:.4f}")
            table.add_row("Entropy Loss", f"{recent_entropy[0]:.4f}", f"{recent_entropy[1]:.4f}", f"{recent_entropy[2]:.4f}")
            
            if all(len(values) > 0 for values in self.value_estimates):
                recent_values = [np.mean(values[-5:]) for values in self.value_estimates]
                recent_ratios = [np.mean(ratios[-5:]) for ratios in self.policy_ratios]
                recent_explained = [np.mean(var[-5:]) for var in self.explained_variance]
                
                table.add_row("", "", "", "")
                table.add_row("Value Estimate", f"{recent_values[0]:.2f}", f"{recent_values[1]:.2f}", f"{recent_values[2]:.2f}")
                table.add_row("Policy Ratio", f"{recent_ratios[0]:.3f}", f"{recent_ratios[1]:.3f}", f"{recent_ratios[2]:.3f}")
                table.add_row("Explained Var", f"{recent_explained[0]:.3f}", f"{recent_explained[1]:.3f}", f"{recent_explained[2]:.3f}")
        
        return table
    
    def create_progress_bar_text(self, timesteps: int, total_timesteps: int, elapsed_time: float) -> str:
        """Create progress bar text manually."""
        progress_pct = timesteps / total_timesteps
        bar_width = 40
        filled_width = int(bar_width * progress_pct)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        if timesteps > 0:
            time_per_step = elapsed_time / timesteps
            remaining_steps = total_timesteps - timesteps
            eta_seconds = remaining_steps * time_per_step
            eta_text = f"{eta_seconds/60:.0f}m {eta_seconds%60:.0f}s"
        else:
            eta_text = "calculating..."
        
        return f"[bold blue]Pacman Training Progress[/] {bar} {timesteps:,}/{total_timesteps:,} • ETA: {eta_text}"

    def train(self, total_timesteps: int = 500000, update_frequency: int = 1024, save_frequency: int = 50000, resume_from: int = 0) -> Tuple[List[List[float]], List[float], List[int]]:
        """Train with rich progress bars and statistics."""
        start_time = time.time()
        timesteps = resume_from
        update_count = resume_from // update_frequency if resume_from > 0 else 0
        self.current_step = timesteps  # Track current step for interrupt handling
        
        # Create layout for statistics
        layout = self.create_training_layout()
        
        # Header
        header_text = Text("🟡 PPO Pacman Training - Pacman vs Ghosts", style="bold magenta")
        layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # Initialize layout
        layout["stats"].update(Panel(Text("Starting training...", style="yellow"), title="🟡 Training Statistics", box=box.ROUNDED))
        layout["losses"].update(Panel(Text("Collecting initial data...", style="yellow"), title="📉 Training Losses", box=box.ROUNDED))
        layout["progress"].update(Panel(Text("Initializing...", style="blue"), title="Progress Info", box=box.ROUNDED))
        
        # Main training loop
        with Live(layout, console=self.console, refresh_per_second=2) as live:
            while timesteps < total_timesteps:
                # Collect trajectories
                trajectories = self.collect_trajectories(update_frequency)
                timesteps += update_frequency
                update_count += 1
                self.current_step = timesteps  # Update current step tracking
                
                # Update all agents
                agents_and_optimizers = [(self.pacman_agent, self.pacman_optimizer)] + list(zip(self.ghost_agents, self.ghost_optimizers))
                
                for i, (agent, optimizer) in enumerate(agents_and_optimizers):
                    stats = self.update_agent(agent, optimizer, trajectories[f'agent_{i}'])
                    
                    # Store statistics
                    self.actor_losses[i].append(stats['actor_loss'])
                    self.critic_losses[i].append(stats['critic_loss'])
                    self.entropy_losses[i].append(stats['entropy_loss'])
                    self.value_estimates[i].append(stats['value_estimate'])
                    self.policy_ratios[i].append(stats['policy_ratio'])
                    self.explained_variance[i].append(stats['explained_variance'])
                
                # Update display
                if update_count % 2 == 0 or not self.render:
                    elapsed_time = time.time() - start_time
                    fps = timesteps / elapsed_time if elapsed_time > 0 else 0
                    
                    layout["stats"].update(Panel(
                        self.create_stats_table(timesteps, update_count, elapsed_time, fps),
                        box=box.ROUNDED
                    ))
                    
                    layout["losses"].update(Panel(
                        self.create_losses_table(),
                        box=box.ROUNDED
                    ))
                    
                    progress_bar_text = self.create_progress_bar_text(timesteps, total_timesteps, elapsed_time)
                    progress_info = f"{progress_bar_text}\nTimestep: {timesteps:,} | Update: {update_count} | FPS: {fps:.1f}"
                    layout["progress"].update(Panel(
                        Text(progress_info, style="bold green"),
                        title="Progress Info",
                        box=box.ROUNDED
                    ))
                
                # Save models
                if timesteps % save_frequency == 0 and timesteps > resume_from:
                    self.save_models(f"models_timestep_{timesteps}", timesteps)
                    if not self.render:
                        save_msg = Text(f"💾 Pacman models saved at timestep {timesteps:,}", style="bold green")
                        layout["progress"].update(Panel(save_msg, title="Save Status", box=box.ROUNDED))
                        live.refresh()
                        time.sleep(1)
        
        elapsed_time = time.time() - start_time
        self.console.print(f"\n🎉 Pacman training completed in {elapsed_time:.2f} seconds!", style="bold green")
        return self.episode_rewards, self.pacman_scores, self.pellets_collected
    
    def save_models(self, filename_prefix: str, timesteps_completed: int = 0) -> None:
        """Save all models and training state."""
        save_dict = {
            'pacman_agent_state_dict': self.pacman_agent.state_dict(),
            'pacman_optimizer_state_dict': self.pacman_optimizer.state_dict(),
            'timesteps_completed': timesteps_completed,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'pacman_scores': self.pacman_scores,
            'pellets_collected': self.pellets_collected,
            'ghost_catches': self.ghost_catches,
        }
        
        # Add ghost models
        for i, (agent, optimizer) in enumerate(zip(self.ghost_agents, self.ghost_optimizers)):
            save_dict[f'ghost_{i}_agent_state_dict'] = agent.state_dict()
            save_dict[f'ghost_{i}_optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, f"{filename_prefix}.pth")
        print(f"Training checkpoint saved as {filename_prefix}.pth")
    
    def load_models(self, filename: str) -> int:
        """Load all models and training state."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        # Load model states
        self.pacman_agent.load_state_dict(checkpoint['pacman_agent_state_dict'])
        self.pacman_optimizer.load_state_dict(checkpoint['pacman_optimizer_state_dict'])
        
        for i, (agent, optimizer) in enumerate(zip(self.ghost_agents, self.ghost_optimizers)):
            agent.load_state_dict(checkpoint[f'ghost_{i}_agent_state_dict'])
            optimizer.load_state_dict(checkpoint[f'ghost_{i}_optimizer_state_dict'])
        
        # Load training statistics if available
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = checkpoint['episode_rewards']
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = checkpoint['episode_lengths']
        if 'pacman_scores' in checkpoint:
            self.pacman_scores = checkpoint['pacman_scores']
        if 'pellets_collected' in checkpoint:
            self.pellets_collected = checkpoint['pellets_collected']
        if 'ghost_catches' in checkpoint:
            self.ghost_catches = checkpoint['ghost_catches']
        
        timesteps_completed = checkpoint.get('timesteps_completed', 0)
        print(f"Training checkpoint loaded from {filename}")
        print(f"Resuming from {timesteps_completed:,} timesteps")
        print(f"Episodes completed so far: {len(self.episode_rewards)}")
        
        return timesteps_completed

def plot_pacman_results(episode_rewards: List[List[float]], pacman_scores: List[float], pellets_collected: List[int]) -> None:
    """Plot Pacman-specific training results."""
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    if episode_rewards and len(episode_rewards) > 0:
        # Episode rewards
        pacman_rewards = [r[0] for r in episode_rewards]
        ghost1_rewards = [r[1] for r in episode_rewards]
        ghost2_rewards = [r[2] for r in episode_rewards]
        
        ax1.plot(pacman_rewards, label='Pacman 🟡', color='gold', alpha=0.7)
        ax1.plot(ghost1_rewards, label='Ghost 1 🔴', color='red', alpha=0.7)
        ax1.plot(ghost2_rewards, label='Ghost 2 🟣', color='purple', alpha=0.7)
        ax1.set_title(f'Episode Rewards ({len(episode_rewards)} episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Pacman scores
        if pacman_scores:
            ax2.plot(pacman_scores, color='gold', linewidth=2)
            ax2.set_title('Pacman Scores')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Score')
            ax2.grid(True)
        
        # Pellets collected
        if pellets_collected:
            ax3.plot(pellets_collected, color='yellow', linewidth=2)
            ax3.set_title('Pellets Collected per Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Pellets')
            ax3.grid(True)
        
        # Moving averages
        if len(pacman_rewards) >= 10:
            window = min(20, len(pacman_rewards) // 4)
            ma_pacman = np.convolve(pacman_rewards, np.ones(window)/window, mode='valid')
            ma_ghost1 = np.convolve(ghost1_rewards, np.ones(window)/window, mode='valid')
            ma_ghost2 = np.convolve(ghost2_rewards, np.ones(window)/window, mode='valid')
            
            ax4.plot(range(window-1, len(pacman_rewards)), ma_pacman, label='Pacman (MA)', color='gold', linewidth=2)
            ax4.plot(range(window-1, len(ghost1_rewards)), ma_ghost1, label='Ghost 1 (MA)', color='red', linewidth=2)
            ax4.plot(range(window-1, len(ghost2_rewards)), ma_ghost2, label='Ghost 2 (MA)', color='purple', linewidth=2)
            ax4.set_title(f'Moving Average Rewards (window={window})')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.legend()
            ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('pacman_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    if episode_rewards:
        print("\n🟡 Pacman Training Summary:")
        print(f"Total Episodes: {len(episode_rewards)}")
        print(f"Pacman Avg Reward: {np.mean([r[0] for r in episode_rewards]):.2f}")
        print(f"Ghost 1 Avg Reward: {np.mean([r[1] for r in episode_rewards]):.2f}")
        print(f"Ghost 2 Avg Reward: {np.mean([r[2] for r in episode_rewards]):.2f}")
        if pacman_scores:
            print(f"Pacman Avg Score: {np.mean(pacman_scores):.2f}")
        if pellets_collected:
            print(f"Avg Pellets Collected: {np.mean(pellets_collected):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agents on Pacman environment')
    parser.add_argument('--show', action='store_true', help='Show training visualization')
    parser.add_argument('--render-freq', type=int, default=5, help='Render every N episodes')
    parser.add_argument('--timesteps', type=int, default=300000, help='Total training timesteps')
    parser.add_argument('--maze-width', type=int, default=19, help='Maze width')
    parser.add_argument('--maze-height', type=int, default=21, help='Maze height')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Resume training from checkpoint (e.g., models_timestep_100000.pth)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🟡 PPO PACMAN TRAINING")
    print("=" * 60)
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"Maze size: {args.maze_width}x{args.maze_height}")
    if args.show:
        print(f"Rendering enabled (every {args.render_freq} episode{'s' if args.render_freq != 1 else ''})")
    else:
        print("Rendering disabled (use --show to enable)")
    print("Agents: 1 Pacman vs 2 Ghosts")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    print("=" * 60)
    
    # Create environment
    env = PacmanEnv(maze_width=args.maze_width, maze_height=args.maze_height)
    
    # Create trainer
    trainer = PacmanPPOTrainer(env, render=args.show, render_freq=args.render_freq)
    
    # Handle resuming from checkpoint
    resume_timesteps = 0
    if args.resume:
        try:
            resume_timesteps = trainer.load_models(args.resume)
            print(f"✅ Successfully resumed from {resume_timesteps:,} timesteps")
        except FileNotFoundError:
            print(f"❌ Checkpoint file '{args.resume}' not found. Starting fresh.")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Starting fresh.")
    
    try:
        # Train agents
        episode_rewards, pacman_scores, pellets_collected = trainer.train(
            total_timesteps=args.timesteps, 
            resume_from=resume_timesteps
        )
        
        # Plot results
        plot_pacman_results(episode_rewards, pacman_scores, pellets_collected)
        
        # Save final models
        trainer.save_models("final_models", args.timesteps)
        
        print("🎉 Pacman training complete! Models saved and plots generated.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        trainer.save_models("interrupted_models", trainer.current_step if hasattr(trainer, 'current_step') else 0)
        print("Models saved as 'interrupted_models.pth'")
        
    finally:
        env.close()