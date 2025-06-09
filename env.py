from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


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

        # Animation and movement
        self.visual_x = float(x)  # Smooth interpolated position
        self.visual_y = float(y)
        self.target_x = x  # Target grid position
        self.target_y = y
        self.last_direction = ActionType.STAY
        self.animation_progress = 1.0  # 0.0 to 1.0, 1.0 = at target

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
            # Update grid position and animation targets
            self.x, self.y = new_x, new_y
            self.target_x, self.target_y = new_x, new_y
            self.animation_progress = 0.0  # Start new animation

            # Track direction for sprite orientation
            if action != ActionType.STAY:
                self.last_direction = action

    def eat_pellet(self, points: int = 10) -> None:
        """Eat a pellet and gain points."""
        self.score += points
        self.pellets_eaten += 1

    def update_animation(self, dt: float = 0.1) -> None:
        """Update smooth movement animation."""
        if self.animation_progress < 1.0:
            # Animation speed - higher values = faster animation
            animation_speed = 8.0
            self.animation_progress = min(
                1.0, self.animation_progress + animation_speed * dt
            )

            # Interpolate visual position
            start_x = self.target_x - (self.target_x - self.visual_x) / (
                1.0 if self.animation_progress == 0 else self.animation_progress
            )
            start_y = self.target_y - (self.target_y - self.visual_y) / (
                1.0 if self.animation_progress == 0 else self.animation_progress
            )

            # Use easing for smoother animation
            t = self.animation_progress
            eased_t = t * t * (3.0 - 2.0 * t)  # Smoothstep

            self.visual_x = start_x + (self.target_x - start_x) * eased_t
            self.visual_y = start_y + (self.target_y - start_y) * eased_t
        else:
            # Animation complete
            self.visual_x = float(self.target_x)
            self.visual_y = float(self.target_y)


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

        # Animation and movement
        self.visual_x = float(x)  # Smooth interpolated position
        self.visual_y = float(y)
        self.target_x = x  # Target grid position
        self.target_y = y
        self.last_direction = ActionType.STAY
        self.animation_progress = 1.0  # 0.0 to 1.0, 1.0 = at target

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
            # Update grid position and animation targets
            self.x, self.y = new_x, new_y
            self.target_x, self.target_y = new_x, new_y
            self.animation_progress = 0.0  # Start new animation

            # Track direction for sprite orientation
            if action != ActionType.STAY:
                self.last_direction = action

    def update_animation(self, dt: float = 0.1) -> None:
        """Update smooth movement animation."""
        if self.animation_progress < 1.0:
            # Animation speed - higher values = faster animation
            animation_speed = 8.0
            self.animation_progress = min(
                1.0, self.animation_progress + animation_speed * dt
            )

            # Interpolate visual position
            start_x = self.target_x - (self.target_x - self.visual_x) / (
                1.0 if self.animation_progress == 0 else self.animation_progress
            )
            start_y = self.target_y - (self.target_y - self.visual_y) / (
                1.0 if self.animation_progress == 0 else self.animation_progress
            )

            # Use easing for smoother animation
            t = self.animation_progress
            eased_t = t * t * (3.0 - 2.0 * t)  # Smoothstep

            self.visual_x = start_x + (self.target_x - start_x) * eased_t
            self.visual_y = start_y + (self.target_y - start_y) * eased_t
        else:
            # Animation complete
            self.visual_x = float(self.target_x)
            self.visual_y = float(self.target_y)


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
        self.action_space = spaces.MultiDiscrete(
            [len(ActionType)] * (1 + self.num_ghosts)
        )

        # Observation space: [agent_data, other_agents, pellet_info, maze_info]
        # Agent's own info: 3 features (x, y, type)
        # Other agents info: 3 agents * 3 features = 9 features
        # Local maze view: 5x5 = 25 features
        # Local pellets: 10 features
        # Total = 3 + 9 + 25 + 10 = 47 features
        obs_size = 47
        self.observation_space = spaces.Box(
            low=0,
            high=max(maze_width, maze_height),
            shape=(1 + self.num_ghosts, obs_size),
            dtype=np.float32,
        )

        # Rendering
        self.screen = None
        self.clock = None
        self.render_mode = None
        self.cell_size = 20

    def _create_maze(self) -> np.ndarray:
        """Create a Pacman-style maze with proper layout."""
        maze = np.ones(
            (self.maze_height, self.maze_width), dtype=int
        )  # Start with all walls

        # Create the basic structure - clear everything first then add walls strategically
        maze[1:-1, 1:-1] = 0  # Clear interior, keep borders as walls

        # Create a symmetric Pacman-like layout
        mid_x = self.maze_width // 2
        mid_y = self.maze_height // 2

        # Horizontal corridors - main pathways
        main_corridors_y = [
            3,  # Top corridor
            mid_y,  # Center corridor
            self.maze_height - 4,  # Bottom corridor
        ]

        for y in main_corridors_y:
            if 1 <= y < self.maze_height - 1:
                maze[y, 1:-1] = 0  # Clear horizontal paths

        # Vertical connectors
        connector_xs = [
            3,  # Left side
            mid_x,  # Center
            self.maze_width - 4,  # Right side
        ]

        for x in connector_xs:
            if 1 <= x < self.maze_width - 1:
                maze[1:-1, x] = 0  # Clear vertical paths

        # Add wall blocks - classic Pacman obstacles
        wall_blocks = []

        # Corner blocks (scaled to current size)
        if self.maze_width >= 15 and self.maze_height >= 15:
            # Top corners
            wall_blocks.extend(
                [
                    (2, 2, 4, 4),  # Top-left
                    (2, self.maze_width - 5, 4, self.maze_width - 3),  # Top-right
                ]
            )

            # Bottom corners
            wall_blocks.extend(
                [
                    (self.maze_height - 5, 2, self.maze_height - 3, 4),  # Bottom-left
                    (
                        self.maze_height - 5,
                        self.maze_width - 5,
                        self.maze_height - 3,
                        self.maze_width - 3,
                    ),  # Bottom-right
                ]
            )

        # Central area obstacles
        if self.maze_width >= 10 and self.maze_height >= 10:
            # Central chamber (ghost house)
            ghost_house_size = min(4, self.maze_width // 5)
            wall_blocks.append(
                (
                    mid_y - ghost_house_size // 2,
                    mid_x - ghost_house_size // 2,
                    mid_y + ghost_house_size // 2,
                    mid_x + ghost_house_size // 2,
                )
            )

            # Side obstacles
            side_size = max(2, self.maze_width // 8)
            wall_blocks.extend(
                [
                    # Left side obstacles
                    (mid_y - 3, 5, mid_y - 1, 5 + side_size),
                    (mid_y + 1, 5, mid_y + 3, 5 + side_size),
                    # Right side obstacles
                    (
                        mid_y - 3,
                        self.maze_width - 5 - side_size,
                        mid_y - 1,
                        self.maze_width - 5,
                    ),
                    (
                        mid_y + 1,
                        self.maze_width - 5 - side_size,
                        mid_y + 3,
                        self.maze_width - 5,
                    ),
                ]
            )

        # Place wall blocks
        for y1, x1, y2, x2 in wall_blocks:
            # Clamp to valid bounds
            y1 = max(1, min(y1, self.maze_height - 2))
            y2 = max(1, min(y2, self.maze_height - 2))
            x1 = max(1, min(x1, self.maze_width - 2))
            x2 = max(1, min(x2, self.maze_width - 2))

            if y1 <= y2 and x1 <= x2:
                maze[y1 : y2 + 1, x1 : x2 + 1] = 1

        # Ensure borders are walls
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1

        # Create side tunnels (classic Pacman feature)
        tunnel_y = mid_y
        if 0 < tunnel_y < self.maze_height - 1:
            maze[tunnel_y, 0] = 0  # Left tunnel
            maze[tunnel_y, -1] = 0  # Right tunnel

        # Ensure key spawn areas are always clear
        spawn_clearances = [
            # Pacman spawn area (bottom center)
            (self.maze_height - 3, mid_x),
            (self.maze_height - 2, mid_x),
            (self.maze_height - 4, mid_x),
            # Ghost spawn areas (around center)
            (mid_y, mid_x - 1),
            (mid_y, mid_x + 1),
            (mid_y - 1, mid_x),
            (mid_y + 1, mid_x),
            # Ensure connectivity around center
            (mid_y, mid_x - 3),
            (mid_y, mid_x + 3),
            (mid_y - 3, mid_x),
            (mid_y + 3, mid_x),
        ]

        for y, x in spawn_clearances:
            if 0 < y < self.maze_height - 1 and 0 < x < self.maze_width - 1:
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
            (self.maze_width - 2, self.maze_height - 2),  # Bottom-right for Ghost 1
            (1, self.maze_height - 2),  # Bottom-left for Ghost 2
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
                (self.maze_width // 2, 1),  # Top center
                (self.maze_width // 2, self.maze_height - 2),  # Bottom center
                (1, self.maze_height // 2),  # Left center
            ]

            for x, y in additional_positions:
                if (
                    0 <= x < self.maze_width
                    and 0 <= y < self.maze_height
                    and (x, y) not in valid_positions
                ):
                    valid_positions.append((x, y))
                    if len(valid_positions) >= num_positions:
                        break

        return valid_positions[:num_positions]

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
                    self.maze_width // 2 + (i + 1) * 2, self.maze_height // 2, i
                )
            self.ghosts.append(ghost)

        # Create pellets
        self.pellets = self._create_pellets()
        self.total_pellets = np.sum(self.pellets)

        return self._get_observations(), {}

    def step(
        self, actions: List[int]
    ) -> Tuple[np.ndarray, List[float], bool, bool, Dict[str, Any]]:
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
            if (
                self.pacman.alive
                and ghost.x == self.pacman.x
                and ghost.y == self.pacman.y
            ):
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
            obs.extend(
                [
                    agent.x / self.maze_width,
                    agent.y / self.maze_height,
                    0.0 if isinstance(agent, Pacman) else 1.0,  # Agent type
                ]
            )

            # Other agents' positions and types
            for j, other_agent in enumerate(all_agents):
                if i != j:
                    obs.extend(
                        [
                            other_agent.x / self.maze_width,
                            other_agent.y / self.maze_height,
                            0.0 if isinstance(other_agent, Pacman) else 1.0,
                        ]
                    )
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
            if hasattr(self.pacman, "just_ate") and self.pacman.just_ate:
                pacman_reward += 1.0  # Reduced but still significant
                self.pacman.just_ate = False

            # Reward based on pellets collected (progress reward)
            pellet_progress = self.pellets_collected / max(self.total_pellets, 1)
            pacman_reward += pellet_progress * 0.1

            # Penalty for staying still (encourage movement)
            if hasattr(self.pacman, "prev_pos"):
                if self.pacman.get_position() == self.pacman.prev_pos:
                    pacman_reward -= 0.05
            self.pacman.prev_pos = self.pacman.get_position()

        else:
            # Death penalty
            pacman_reward = -1.0

        # Ghost rewards - simplified and balanced
        for i, ghost in enumerate(self.ghosts):
            if hasattr(ghost, "caught_pacman") and ghost.caught_pacman:
                ghost_rewards[i] += 2.0  # Balanced reward for catching Pacman
                ghost.caught_pacman = False
            else:
                if self.pacman.alive:
                    # Reward for being close to Pacman (encourage pursuit)
                    distance = abs(ghost.x - self.pacman.x) + abs(
                        ghost.y - self.pacman.y
                    )
                    if distance <= 3:  # Very close
                        ghost_rewards[i] += 0.1
                    elif distance <= 6:  # Moderately close
                        ghost_rewards[i] += 0.05

                    # Small movement reward
                    if hasattr(ghost, "prev_pos"):
                        if ghost.get_position() != getattr(ghost, "prev_pos", (0, 0)):
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
        return (
            not self.pacman.alive
            or self.pellets_collected >= self.total_pellets
            or self.current_step >= self.max_steps
        )

    def render(self, mode: str = "human") -> None:
        if self.render_mode is None:
            self.render_mode = mode

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            screen_width = self.maze_width * self.cell_size
            screen_height = (
                self.maze_height * self.cell_size + 100
            )  # Extra space for UI
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
                    self.cell_size,
                )

                if self.maze[y, x] == 1:  # Wall
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Blue walls
                else:  # Empty space
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Black space

                    # Draw pellets
                    if self.pellets[y, x] == 1:
                        pellet_center = (
                            x * self.cell_size + self.cell_size // 2,
                            y * self.cell_size + self.cell_size // 2,
                        )
                        pygame.draw.circle(self.screen, (255, 255, 0), pellet_center, 2)

        # Update animations
        self.pacman.update_animation()
        for ghost in self.ghosts:
            ghost.update_animation()

        # Draw Pacman with smooth animation and direction
        if self.pacman.alive:
            pacman_center = (
                int(self.pacman.visual_x * self.cell_size + self.cell_size // 2),
                int(self.pacman.visual_y * self.cell_size + self.cell_size // 2),
            )

            # Draw Pacman body
            pygame.draw.circle(
                self.screen, (255, 255, 0), pacman_center, self.cell_size // 3
            )

            # Draw directional mouth based on last direction
            mouth_size = self.cell_size // 4
            if self.pacman.last_direction == ActionType.RIGHT:
                # Mouth facing right
                mouth_points = [
                    pacman_center,
                    (pacman_center[0] + mouth_size, pacman_center[1] - mouth_size // 2),
                    (pacman_center[0] + mouth_size, pacman_center[1] + mouth_size // 2),
                ]
            elif self.pacman.last_direction == ActionType.LEFT:
                # Mouth facing left
                mouth_points = [
                    pacman_center,
                    (pacman_center[0] - mouth_size, pacman_center[1] - mouth_size // 2),
                    (pacman_center[0] - mouth_size, pacman_center[1] + mouth_size // 2),
                ]
            elif self.pacman.last_direction == ActionType.UP:
                # Mouth facing up
                mouth_points = [
                    pacman_center,
                    (pacman_center[0] - mouth_size // 2, pacman_center[1] - mouth_size),
                    (pacman_center[0] + mouth_size // 2, pacman_center[1] - mouth_size),
                ]
            elif self.pacman.last_direction == ActionType.DOWN:
                # Mouth facing down
                mouth_points = [
                    pacman_center,
                    (pacman_center[0] - mouth_size // 2, pacman_center[1] + mouth_size),
                    (pacman_center[0] + mouth_size // 2, pacman_center[1] + mouth_size),
                ]
            else:
                # Default mouth (right)
                mouth_points = [
                    pacman_center,
                    (pacman_center[0] + mouth_size, pacman_center[1] - mouth_size // 2),
                    (pacman_center[0] + mouth_size, pacman_center[1] + mouth_size // 2),
                ]

            # Draw the mouth
            pygame.draw.polygon(self.screen, (0, 0, 0), mouth_points)

        # Draw Ghosts with smooth animation and direction
        for ghost in self.ghosts:
            ghost_center = (
                int(ghost.visual_x * self.cell_size + self.cell_size // 2),
                int(ghost.visual_y * self.cell_size + self.cell_size // 2),
            )

            # Draw ghost body
            pygame.draw.circle(
                self.screen, ghost.color, ghost_center, self.cell_size // 3
            )

            # Draw ghost eyes based on direction
            eye_size = 2
            eye_offset = 3

            if ghost.last_direction == ActionType.LEFT:
                # Eyes looking left
                left_eye = (ghost_center[0] - eye_offset - 1, ghost_center[1] - 2)
                right_eye = (ghost_center[0] + eye_offset - 1, ghost_center[1] - 2)
            elif ghost.last_direction == ActionType.RIGHT:
                # Eyes looking right
                left_eye = (ghost_center[0] - eye_offset + 1, ghost_center[1] - 2)
                right_eye = (ghost_center[0] + eye_offset + 1, ghost_center[1] - 2)
            elif ghost.last_direction == ActionType.UP:
                # Eyes looking up
                left_eye = (ghost_center[0] - eye_offset, ghost_center[1] - 3)
                right_eye = (ghost_center[0] + eye_offset, ghost_center[1] - 3)
            elif ghost.last_direction == ActionType.DOWN:
                # Eyes looking down
                left_eye = (ghost_center[0] - eye_offset, ghost_center[1] - 1)
                right_eye = (ghost_center[0] + eye_offset, ghost_center[1] - 1)
            else:
                # Default eyes (center)
                left_eye = (ghost_center[0] - eye_offset, ghost_center[1] - 2)
                right_eye = (ghost_center[0] + eye_offset, ghost_center[1] - 2)

            # Draw white eye background
            pygame.draw.circle(self.screen, (255, 255, 255), left_eye, eye_size)
            pygame.draw.circle(self.screen, (255, 255, 255), right_eye, eye_size)

            # Draw black pupils
            pygame.draw.circle(self.screen, (0, 0, 0), left_eye, eye_size - 1)
            pygame.draw.circle(self.screen, (0, 0, 0), right_eye, eye_size - 1)

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
            print(
                f"Pacman: Pos=({env.pacman.x}, {env.pacman.y}), Score={env.pacman.score}, Alive={env.pacman.alive}"
            )
            print(f"Pellets collected: {env.pellets_collected}/{env.total_pellets}")

        # Uncomment to see rendering
        # env.render()

        if done:
            print(f"Episode finished at step {step}")
            break

    env.close()
