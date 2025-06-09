import gymnasium as gym
import numpy as np
import pygame
import math
from typing import Tuple, List, Dict, Optional, Any
from gymnasium import spaces
from enum import Enum

class ActionType(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class EntityType(Enum):
    PACMAN = 0
    GHOST = 1

class Pacman:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.score = 0
        self.alive = True
        self.size = 12
        self.speed = 1  # Grid-based movement
        self.pellets_eaten = 0
        
    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def move(self, action: ActionType, maze: np.ndarray) -> None:
        """Move Pacman based on action, respecting maze walls."""
        new_x, new_y = self.x, self.y
        
        if action == ActionType.UP and self.y > 0:
            new_y = self.y - 1
        elif action == ActionType.DOWN and self.y < maze.shape[0] - 1:
            new_y = self.y + 1
        elif action == ActionType.LEFT and self.x > 0:
            new_x = self.x - 1
        elif action == ActionType.RIGHT and self.x < maze.shape[1] - 1:
            new_x = self.x + 1
        
        # Check if new position is not a wall
        if maze[new_y, new_x] != 1:  # 1 = wall
            self.x, self.y = new_x, new_y
    
    def eat_pellet(self, points: int = 10) -> None:
        """Eat a pellet and gain points."""
        self.score += points
        self.pellets_eaten += 1

class Ghost:
    def __init__(self, x: int, y: int, ghost_id: int) -> None:
        self.x = x
        self.y = y
        self.ghost_id = ghost_id
        self.size = 12
        self.speed = 1
        self.caught_pacman = False
        
        # Ghost colors for rendering
        self.colors = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]
        self.color = self.colors[ghost_id % len(self.colors)]
    
    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def move(self, action: ActionType, maze: np.ndarray) -> None:
        """Move Ghost based on action, respecting maze walls."""
        new_x, new_y = self.x, self.y
        
        if action == ActionType.UP and self.y > 0:
            new_y = self.y - 1
        elif action == ActionType.DOWN and self.y < maze.shape[0] - 1:
            new_y = self.y + 1
        elif action == ActionType.LEFT and self.x > 0:
            new_x = self.x - 1
        elif action == ActionType.RIGHT and self.x < maze.shape[1] - 1:
            new_x = self.x + 1
        
        # Check if new position is not a wall
        if maze[new_y, new_x] != 1:  # 1 = wall
            self.x, self.y = new_x, new_y

class PacmanEnv(gym.Env):
    def __init__(self, maze_width: int = 19, maze_height: int = 21) -> None:
        super().__init__()
        
        # Environment parameters
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.max_steps = 500  # Much shorter episodes for faster learning
        self.current_step = 0
        
        # Game entities
        self.pacman = None
        self.ghosts = []
        self.num_ghosts = 2
        
        # Maze and pellets
        self.maze = None
        self.pellets = None
        self.total_pellets = 0
        self.pellets_collected = 0
        
        # Action space: 3 agents (1 Pacman + 2 Ghosts), each with 5 actions
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * (1 + self.num_ghosts))
        
        # Observation space: [agent_data, other_agents, pellet_info, maze_info]
        # Agent's own info: 3 features (x, y, type)
        # Other agents info: 3 agents * 3 features = 9 features 
        # Local maze view: 5x5 = 25 features
        # Local pellets: 10 features
        # Total = 3 + 9 + 25 + 10 = 47 features
        obs_size = 47
        self.observation_space = spaces.Box(
            low=0, high=max(maze_width, maze_height), 
            shape=(1 + self.num_ghosts, obs_size), 
            dtype=np.float32
        )
        
        # Rendering
        self.screen = None
        self.clock = None
        self.render_mode = None
        self.cell_size = 20
        
    def _create_maze(self) -> np.ndarray:
        """Create a fixed Pacman-style maze with guaranteed connectivity."""
        maze = np.zeros((self.maze_height, self.maze_width), dtype=int)
        
        # Create border walls
        maze[0, :] = 1  # Top wall
        maze[-1, :] = 1  # Bottom wall
        maze[:, 0] = 1  # Left wall
        maze[:, -1] = 1  # Right wall
        
        # Create a fixed maze pattern that guarantees connectivity
        # This creates a classic Pacman-like layout
        
        # Horizontal corridors at regular intervals
        for y in [3, 6, 9, 12, 15, 18]:
            if y < self.maze_height:
                maze[y, 1:-1] = 0  # Clear horizontal paths
        
        # Vertical corridors at regular intervals  
        for x in [3, 6, 9, 12, 15]:
            if x < self.maze_width:
                maze[1:-1, x] = 0  # Clear vertical paths
        
        # Add some strategic walls for interesting gameplay
        wall_patterns = [
            # Corner blocks
            (2, 2), (2, self.maze_width-3),
            (self.maze_height-3, 2), (self.maze_height-3, self.maze_width-3),
            
            # Central obstacles (if space allows)
            (self.maze_height//2, self.maze_width//2),
            (self.maze_height//2-1, self.maze_width//2),
            (self.maze_height//2+1, self.maze_width//2),
            
            # Side obstacles
            (5, 8), (8, 5), (8, self.maze_width-6), (self.maze_height-6, 8),
        ]
        
        for y, x in wall_patterns:
            if (1 < y < self.maze_height-1 and 1 < x < self.maze_width-1 and
                y < self.maze_height and x < self.maze_width):
                maze[y, x] = 1
        
        # Ensure key areas remain clear
        key_clear_areas = [
            # Spawn areas
            (1, 1), (1, 2), (2, 1),  # Top-left spawn
            (self.maze_height-2, self.maze_width-2), 
            (self.maze_height-2, self.maze_width-3),
            (self.maze_height-3, self.maze_width-2),  # Bottom-right spawn
            
            # Center area
            (self.maze_height//2, self.maze_width//2-1),
            (self.maze_height//2, self.maze_width//2+1),
        ]
        
        for y, x in key_clear_areas:
            if 0 <= y < self.maze_height and 0 <= x < self.maze_width:
                maze[y, x] = 0
        
        return maze
    
    def _create_pellets(self) -> np.ndarray:
        """Create pellets in all empty spaces."""
        pellets = np.zeros_like(self.maze)
        
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y, x] == 0:  # Empty space
                    pellets[y, x] = 1
        
        # Remove pellets from starting positions
        pacman_y, pacman_x = self.pacman.y, self.pacman.x
        pellets[pacman_y, pacman_x] = 0
        
        for ghost in self.ghosts:
            pellets[ghost.y, ghost.x] = 0
        
        return pellets
    
    def _find_empty_positions(self, num_positions: int) -> List[Tuple[int, int]]:
        """Find fixed spawn positions for consistent gameplay."""
        # Use fixed spawn positions that are guaranteed to be clear
        fixed_positions = [
            (1, 1),  # Top-left for Pacman
            (self.maze_width-2, self.maze_height-2),  # Bottom-right for Ghost 1
            (1, self.maze_height-2),  # Bottom-left for Ghost 2
        ]
        
        # Return positions within maze bounds
        valid_positions = []
        for x, y in fixed_positions:
            if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
                valid_positions.append((x, y))
                if len(valid_positions) >= num_positions:
                    break
        
        # If we need more positions, add some safe ones
        if len(valid_positions) < num_positions:
            additional_positions = [
                (self.maze_width//2, 1),  # Top center
                (self.maze_width//2, self.maze_height-2),  # Bottom center
                (1, self.maze_height//2),  # Left center
            ]
            
            for x, y in additional_positions:
                if (0 <= x < self.maze_width and 0 <= y < self.maze_height and 
                    (x, y) not in valid_positions):
                    valid_positions.append((x, y))
                    if len(valid_positions) >= num_positions:
                        break
        
        return valid_positions[:num_positions]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        self.pellets_collected = 0
        
        # Create maze
        self.maze = self._create_maze()
        
        # Find spawn positions
        spawn_positions = self._find_empty_positions(1 + self.num_ghosts)
        
        # Create Pacman
        if spawn_positions:
            pacman_pos = spawn_positions[0]
            self.pacman = Pacman(pacman_pos[0], pacman_pos[1])
        else:
            # Fallback position
            self.pacman = Pacman(self.maze_width // 2, self.maze_height // 2)
        
        # Create Ghosts
        self.ghosts = []
        for i in range(self.num_ghosts):
            if i + 1 < len(spawn_positions):
                ghost_pos = spawn_positions[i + 1]
                ghost = Ghost(ghost_pos[0], ghost_pos[1], i)
            else:
                # Fallback positions
                ghost = Ghost(
                    self.maze_width // 2 + (i + 1) * 2, 
                    self.maze_height // 2, 
                    i
                )
            self.ghosts.append(ghost)
        
        # Create pellets
        self.pellets = self._create_pellets()
        self.total_pellets = np.sum(self.pellets)
        
        return self._get_observations(), {}
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # Process actions: [pacman_action, ghost1_action, ghost2_action]
        if self.pacman.alive:
            self.pacman.move(ActionType(actions[0]), self.maze)
        
        for i, ghost in enumerate(self.ghosts):
            ghost.move(ActionType(actions[i + 1]), self.maze)
        
        # Check pellet collection
        if self.pacman.alive and self.pellets[self.pacman.y, self.pacman.x] == 1:
            self.pellets[self.pacman.y, self.pacman.x] = 0
            self.pacman.eat_pellet()
            self.pacman.just_ate = True  # Flag for reward calculation
            self.pellets_collected += 1
        
        # Check ghost-pacman collisions
        for ghost in self.ghosts:
            if (self.pacman.alive and 
                ghost.x == self.pacman.x and 
                ghost.y == self.pacman.y):
                self.pacman.alive = False
                ghost.caught_pacman = True
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check if episode is done
        done = self._is_done()
        
        # Get observations
        observations = self._get_observations()
        
        return observations, rewards, done, False, {}
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        observations = []
        
        all_agents = [self.pacman] + self.ghosts
        
        for i, agent in enumerate(all_agents):
            obs = []
            
            # Agent's own position and type
            obs.extend([
                agent.x / self.maze_width,
                agent.y / self.maze_height,
                0.0 if isinstance(agent, Pacman) else 1.0  # Agent type
            ])
            
            # Other agents' positions and types
            for j, other_agent in enumerate(all_agents):
                if i != j:
                    obs.extend([
                        other_agent.x / self.maze_width,
                        other_agent.y / self.maze_height,
                        0.0 if isinstance(other_agent, Pacman) else 1.0
                    ])
                else:
                    obs.extend([0.0, 0.0, 0.0])  # Placeholder for self
            
            # Local maze view (5x5 around agent)
            local_view = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y, x = agent.y + dy, agent.x + dx
                    if 0 <= y < self.maze_height and 0 <= x < self.maze_width:
                        local_view.append(self.maze[y, x])  # Wall info
                    else:
                        local_view.append(1.0)  # Out of bounds = wall
            obs.extend(local_view)
            
            # Pellet information (local pellets in 5x5 view)
            local_pellets = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y, x = agent.y + dy, agent.x + dx
                    if 0 <= y < self.maze_height and 0 <= x < self.maze_width:
                        local_pellets.append(self.pellets[y, x])
                    else:
                        local_pellets.append(0.0)  # No pellets out of bounds
            
            # Take first 10 for observation size consistency
            obs.extend(local_pellets[:10])
            
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
    
    def _calculate_rewards(self) -> List[float]:
        """Calculate rewards for [Pacman, Ghost1, Ghost2]."""
        pacman_reward = 0.0
        ghost_rewards = [0.0] * self.num_ghosts
        
        # Pacman rewards
        if self.pacman.alive:
            # Small survival reward
            pacman_reward += 0.01
            
            # BIG pellet collection reward (main objective)
            if hasattr(self.pacman, 'just_ate') and self.pacman.just_ate:
                pacman_reward += 1.0  # Reduced but still significant
                self.pacman.just_ate = False
            
            # Reward based on pellets collected (progress reward)
            pellet_progress = self.pellets_collected / max(self.total_pellets, 1)
            pacman_reward += pellet_progress * 0.1
            
            # Penalty for staying still (encourage movement)
            if hasattr(self.pacman, 'prev_pos'):
                if self.pacman.get_position() == self.pacman.prev_pos:
                    pacman_reward -= 0.05
            self.pacman.prev_pos = self.pacman.get_position()
            
        else:
            # Death penalty
            pacman_reward = -1.0
        
        # Ghost rewards - simplified and balanced
        for i, ghost in enumerate(self.ghosts):
            if hasattr(ghost, 'caught_pacman') and ghost.caught_pacman:
                ghost_rewards[i] += 2.0  # Balanced reward for catching Pacman
                ghost.caught_pacman = False
            else:
                if self.pacman.alive:
                    # Reward for being close to Pacman (encourage pursuit)
                    distance = abs(ghost.x - self.pacman.x) + abs(ghost.y - self.pacman.y)
                    if distance <= 3:  # Very close
                        ghost_rewards[i] += 0.1
                    elif distance <= 6:  # Moderately close
                        ghost_rewards[i] += 0.05
                    
                    # Small movement reward
                    if hasattr(ghost, 'prev_pos'):
                        if ghost.get_position() != getattr(ghost, 'prev_pos', (0, 0)):
                            ghost_rewards[i] += 0.01
                    ghost.prev_pos = ghost.get_position()
        
        # Game completion bonuses
        if self.pellets_collected >= self.total_pellets:
            pacman_reward += 5.0  # Win bonus for Pacman
        elif self.pellets_collected >= self.total_pellets * 0.8:
            pacman_reward += 2.0  # Progress bonus
        
        return [pacman_reward] + ghost_rewards
    
    def _is_done(self) -> bool:
        """Check if episode should end."""
        # Episode ends if Pacman dies, all pellets eaten, or max steps reached
        return (not self.pacman.alive or 
                self.pellets_collected >= self.total_pellets or 
                self.current_step >= self.max_steps)
    
    def render(self, mode: str = 'human') -> None:
        if self.render_mode is None:
            self.render_mode = mode
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            screen_width = self.maze_width * self.cell_size
            screen_height = self.maze_height * self.cell_size + 100  # Extra space for UI
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Pacman RL Environment")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw maze
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                if self.maze[y, x] == 1:  # Wall
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Blue walls
                else:  # Empty space
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Black space
                    
                    # Draw pellets
                    if self.pellets[y, x] == 1:
                        pellet_center = (
                            x * self.cell_size + self.cell_size // 2,
                            y * self.cell_size + self.cell_size // 2
                        )
                        pygame.draw.circle(self.screen, (255, 255, 0), pellet_center, 2)
        
        # Draw Pacman
        if self.pacman.alive:
            pacman_center = (
                self.pacman.x * self.cell_size + self.cell_size // 2,
                self.pacman.y * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.screen, (255, 255, 0), pacman_center, self.cell_size // 3)
            
            # Draw Pacman "mouth" (simple line)
            mouth_end = (
                pacman_center[0] + self.cell_size // 4,
                pacman_center[1]
            )
            pygame.draw.line(self.screen, (0, 0, 0), pacman_center, mouth_end, 2)
        
        # Draw Ghosts
        for ghost in self.ghosts:
            ghost_center = (
                ghost.x * self.cell_size + self.cell_size // 2,
                ghost.y * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.screen, ghost.color, ghost_center, self.cell_size // 3)
            
            # Draw ghost eyes
            eye_size = 2
            left_eye = (ghost_center[0] - 4, ghost_center[1] - 2)
            right_eye = (ghost_center[0] + 4, ghost_center[1] - 2)
            pygame.draw.circle(self.screen, (255, 255, 255), left_eye, eye_size)
            pygame.draw.circle(self.screen, (255, 255, 255), right_eye, eye_size)
        
        # Draw UI information
        if pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            ui_y = self.maze_height * self.cell_size + 10
            
            # Pacman info
            score_text = f"Score: {self.pacman.score}"
            pellets_text = f"Pellets: {self.pellets_collected}/{self.total_pellets}"
            step_text = f"Step: {self.current_step}/{self.max_steps}"
            status_text = f"Status: {'ALIVE' if self.pacman.alive else 'CAUGHT'}"
            
            texts = [score_text, pellets_text, step_text, status_text]
            for i, text in enumerate(texts):
                color = (0, 255, 0) if self.pacman.alive else (255, 0, 0)
                if i == 3:  # Status text
                    color = (0, 255, 0) if self.pacman.alive else (255, 0, 0)
                else:
                    color = (255, 255, 255)
                
                text_surface = font.render(text, True, color)
                self.screen.blit(text_surface, (10 + i * 150, ui_y))
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS for smooth viewing
        
        return None
    
    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

# Example usage and testing
if __name__ == "__main__":
    env = PacmanEnv()
    
    # Test the environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    for step in range(200):
        # Random actions for testing: [pacman_action, ghost1_action, ghost2_action]
        actions = [np.random.randint(0, len(ActionType)) for _ in range(3)]
        obs, rewards, done, truncated, info = env.step(actions)
        
        if step % 20 == 0:
            print(f"Step {step}: Rewards = {rewards}, Done = {done}")
            print(f"Pacman: Pos=({env.pacman.x}, {env.pacman.y}), Score={env.pacman.score}, Alive={env.pacman.alive}")
            print(f"Pellets collected: {env.pellets_collected}/{env.total_pellets}")
        
        # Uncomment to see rendering
        # env.render()
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    env.close()