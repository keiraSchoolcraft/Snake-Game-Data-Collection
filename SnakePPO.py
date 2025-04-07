import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import pygame
import os
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# Import Snake game
from snake_game import SnakeGame
from config import GRID_SIZE, CELL_SIZE, FOOD, SNAKE_HEAD, SNAKE_BODY

# Force CUDA if available
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    # Set CUDA options for better performance
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class SnakeGameRLWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        
        # Enhanced observation space
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(7,), # Modified observation space with better features
            dtype=np.float32
        )
        
        # Create the original Snake game instance
        self.game = SnakeGame()
        
        # Temporarily disable logging
        self.game.logging_enabled = False
        if hasattr(self.game, 'logger'):
            self.game.logger.game_started = False
        
        # Tracking variables
        self.steps_without_food = 0
        self.max_steps_without_food = 200
        self.previous_distance = None
    
    def reset(self, seed=None):
        super().reset(seed=None)  # Don't use fixed seeds
        
        # Randomize initial snake position
        start_x = random.randint(2, GRID_SIZE - 3)
        start_y = random.randint(2, GRID_SIZE - 3)
        
        # Reset the game with more randomness
        self.game.reset_game()
        
        # Manually set initial snake position
        self.game.snake = [(start_x, start_y)]
        
        # Randomize initial direction
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.game.direction = random.choice(directions)
        
        # Randomize food placement
        self.game.food = self._generate_random_food()
        
        # Ensure logging remains disabled
        self.game.logging_enabled = False
        if hasattr(self.game, 'logger'):
            self.game.logger.game_started = False
        
        # Reset tracking variables
        self.steps_without_food = 0
        head = self.game.snake[0]
        food = self.game.food
        self.previous_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
        
        return self._get_observation(), {}
    
    def _generate_random_food(self):
        while True:
            food = (
                random.randint(0, GRID_SIZE - 1), 
                random.randint(0, GRID_SIZE - 1)
            )
            # Ensure food is not on the snake
            if food not in self.game.snake:
                return food
                
    def _get_observation(self):
        """Get an enhanced observation of the game state"""
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        
        # Normalize positions to [-1, 1]
        norm_head_x = (head_x / (GRID_SIZE - 1)) * 2 - 1
        norm_head_y = (head_y / (GRID_SIZE - 1)) * 2 - 1
        norm_food_x = (food_x / (GRID_SIZE - 1)) * 2 - 1
        norm_food_y = (food_y / (GRID_SIZE - 1)) * 2 - 1
        
        # Calculate danger in each direction (0 = safe, 1 = danger)
        danger_up = 1.0 if self._is_collision(head_x, head_y - 1) else 0.0
        danger_right = 1.0 if self._is_collision(head_x + 1, head_y) else 0.0
        danger_down = 1.0 if self._is_collision(head_x, head_y + 1) else 0.0
        danger_left = 1.0 if self._is_collision(head_x - 1, head_y) else 0.0
        
        # Current direction
        dir_up = 1.0 if self.game.direction == "UP" else 0.0
        dir_right = 1.0 if self.game.direction == "RIGHT" else 0.0
        dir_down = 1.0 if self.game.direction == "DOWN" else 0.0
        dir_left = 1.0 if self.game.direction == "LEFT" else 0.0
        
        # Food direction
        food_up = 1.0 if food_y < head_y else 0.0
        food_right = 1.0 if food_x > head_x else 0.0
        food_down = 1.0 if food_y > head_y else 0.0
        food_left = 1.0 if food_x < head_x else 0.0
        
        return np.array([
            # Normalized relative food position
            (food_x - head_x) / GRID_SIZE,  # Food x relative to head (normalized)
            (food_y - head_y) / GRID_SIZE,  # Food y relative to head (normalized)
            
            # Danger in each direction
            danger_up,
            danger_right,
            danger_down,
            danger_left,
            
            # Snake length (normalized)
            len(self.game.snake) / GRID_SIZE**2  # Normalized snake length
        ], dtype=np.float32)
    
    def _is_collision(self, x, y):
        """Check if a position would result in collision"""
        # Check if point is outside boundaries
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return True
        
        # Check if point is on snake body (excluding head)
        if (x, y) in self.game.snake[1:]:
            return True
            
        return False

    def step(self, action):
        # Map action to Pygame key event
        action_map = {
            0: pygame.K_UP,
            1: pygame.K_RIGHT,
            2: pygame.K_DOWN,
            3: pygame.K_LEFT,
        }
        
        # Create a mock Pygame event
        mock_event = pygame.event.Event(pygame.KEYDOWN, {'key': action_map[action]})
        
        # Store initial state
        initial_score = self.game.score
        initial_head = self.game.snake[0]
        initial_snake_length = len(self.game.snake)
        
        # Change direction
        self.game.change_direction(mock_event)
        
        # Move snake
        self.game.move_snake()
        
        # Check if game is still running
        done = not self.game.running
        
        # Calculate food distance
        head = self.game.snake[0]
        food = self.game.food
        current_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
        
        # Improved Reward Shaping
        reward = 0
        
        # 1. Survival reward - small positive reward for staying alive
        reward += 0.01
        
        # 2. Food distance reward - moderate reward for getting closer to food
        if self.previous_distance is not None:
            distance_change = self.previous_distance - current_distance
            reward += distance_change * 0.1
        
        # 3. Food reward - larger reward for eating food
        if self.game.score > initial_score:
            reward += 1.0
            self.steps_without_food = 0
        else:
            self.steps_without_food += 1
        
        # 4. Starvation penalty - penalty for not finding food for too long
        if self.steps_without_food > self.max_steps_without_food:
            reward -= 0.5
            # Move food to a new location if snake is struggling to find it
            self.game.food = self._generate_random_food()
            self.steps_without_food = 0
        
        # 5. Death penalty - penalty for dying
        if done:
            reward -= 1.0
        
        # Update previous distance for next step
        self.previous_distance = current_distance
        
        return (
            self._get_observation(),
            reward,
            done,
            False,  # truncated
            {}
        )

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        # A more specialized network architecture
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)  # No activation on the output layer for Q-values

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0005):
        self.device = device
        
        # Main network for actions
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        
        # Target network for stable Q-value predictions
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for inference only
        
        # Optimizer - RMSprop often works well for DQN
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.eps_start = 1.0  # Starting epsilon for exploration
        self.eps_end = 0.05  # Minimum epsilon
        self.eps_decay = 0.9999  # Decay rate for epsilon
        self.target_update = 1000  # Update target network every N steps
        self.batch_size = 64  # Batch size for training
        
        # Experience replay memory
        self.memory = ReplayMemory(100000)  # Large memory for diverse experiences
        
        # Current exploration rate
        self.epsilon = self.eps_start
        
        # Step counter
        self.steps_done = 0
    
    def select_action(self, state):
        # Decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            return random.randint(0, 3)
        else:
            # Convert state to tensor
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action with highest Q-value
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
    
    def store_experience(self, state, action, reward, next_state, done):
        # Store transition in replay memory
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def train(self):
        # Check if we have enough samples
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples yet
        
        # Sample a batch from memory
        experiences = self.memory.sample(self.batch_size)
        
        # Convert batch of experiences to tensors
        batch = Experience(*zip(*experiences))
        
        # Convert to appropriate tensor shapes
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net; 
        # selecting their best reward with max(1)[0].
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))
        
        # Compute Huber loss (more robust than MSE)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def train_dqn(env, agent, num_episodes=10000):
    """Train the agent using DQN algorithm with no step limit"""
    # Create logs directory
    os.makedirs('training_logs', exist_ok=True)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_scores = []
    
    # For tracking best performance
    best_avg_reward = -float('inf')
    best_avg_score = -float('inf')
    
    # Time tracking
    start_time = time.time()
    last_time = start_time
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        total_loss = 0
        done = False
        
        # Continue until the game naturally ends (no max_steps limit)
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            if loss > 0:
                total_loss += loss
            
            # Update state and tracking
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Average loss for the episode
        avg_loss = total_loss / max(1, episode_length)
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_losses.append(avg_loss)
        episode_scores.append(env.game.score)
        

        save_every = 1000
        # Periodic logging and saving
        if (episode + 1) % save_every == 0:
            current_time = time.time()
            elapsed = current_time - last_time
            total_elapsed = current_time - start_time
            last_time = current_time
            
            # Calculate moving averages
            recent_rewards = episode_rewards[-save_every:]
            recent_lengths = episode_lengths[-save_every:]
            recent_scores = episode_scores[-save_every:]
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            avg_score = np.mean(recent_scores)
            
            print(f"Episode {episode+1}/{num_episodes} (Time: {elapsed:.1f}s, Total: {total_elapsed:.1f}s)")
            print(f"  Avg Reward (last set): {avg_reward:.2f}")
            print(f"  Avg Episode Length (last set): {avg_length:.2f}")
            print(f"  Avg Score (last set): {avg_score:.2f}")
            print(f"  Max Score (last set): {np.max(recent_scores)}")
            print(f"  Exploration Rate: {agent.epsilon:.4f}")
            print(f"  Recent Loss: {avg_loss:.4f}")
            print("---")
            
            # Save model periodically
            checkpoint_filename = os.path.join('training_logs', f'dqn_snake_model_ep{episode+1}.pth')
            torch.save({
                'policy_net_state': agent.policy_net.state_dict(),
                'target_net_state': agent.target_net.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'episode': episode,
                'epsilon': agent.epsilon,
                'steps_done': agent.steps_done
            }, checkpoint_filename, _use_new_zipfile_serialization=True)  # Use this to ensure future compatibility
            
            # Save best model based on avg reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save({
                    'policy_net_state': agent.policy_net.state_dict(),
                    'target_net_state': agent.target_net.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict(),
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'avg_score': avg_score,
                    'epsilon': agent.epsilon
                }, os.path.join('best_models', 'best_reward_model.pth'), _use_new_zipfile_serialization=True)
                print(f"  New best avg reward: {best_avg_reward:.2f} - Model saved")
            
            # Save best model based on avg score
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save({
                    'policy_net_state': agent.policy_net.state_dict(),
                    'target_net_state': agent.target_net.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict(),
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'avg_score': avg_score,
                    'epsilon': agent.epsilon
                }, os.path.join('best_models', 'best_score_model.pth'), _use_new_zipfile_serialization=True)
                print(f"  New best avg score: {best_avg_score:.2f} - Model saved")
    
    return episode_rewards, episode_lengths, episode_losses, episode_scores

def play_with_trained_model(model_path='best_models/best_score_model.pth', num_games=5):
    """Play the game using a trained model"""
    # Initialize environment and agent
    env = SnakeGameRLWrapper()
    
    # DQN agent with observation space size
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=4
    )
    
    # Load the checkpoint (explicitly set weights_only=False)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state'])
        agent.target_net.load_state_dict(checkpoint['target_net_state'])
        print(f"Loaded model from: {model_path}")
        if 'avg_reward' in checkpoint and 'avg_score' in checkpoint:
            print(f"Model stats - Avg Reward: {checkpoint['avg_reward']:.2f}, Avg Score: {checkpoint['avg_score']:.2f}")
    else:
        print(f"No model found at {model_path}, using untrained model")
    
    # Set exploration to zero for deterministic play
    agent.epsilon = 0
    
    # Tracking game results
    game_scores = []
    game_lengths = []
    
    # Make sure environment variable is completely cleared
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    # Simple way to force display: play one invisible, super quick game first to "warm up" pygame
    # This should solve the first game not showing issue
    print("Initializing display...")
    pygame.init()
    dummy_screen = pygame.display.set_mode((1, 1))
    pygame.display.quit()
    pygame.quit()
    print("Display initialized")
    
    for game in range(num_games):
        print(f"Starting game {game+1}...")
        
        # Reset the environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Completely restart pygame for each game
        pygame.init()
        screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        pygame.display.set_caption(f"Snake Game - Trained Model (Game {game+1})")
        clock = pygame.time.Clock()
        
        # Give a small delay for the window to appear
        pygame.time.delay(100)
        
        # Continue until game is done (no maximum steps)
        while not done:
            # Slow down the game for visualization
            clock.tick(10)  # Adjust speed as needed
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Game closed by user")
                    return
            
            # Select action using trained policy
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Visualize the game state
            screen.fill((0, 0, 0))  # Black background
            
            # Draw snake
            for x, y in env.game.snake:
                color = (0, 255, 0) if (x, y) == env.game.snake[0] else (0, 200, 0)
                pygame.draw.rect(screen, color, 
                               (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            # Draw food
            food_x, food_y = env.game.food
            pygame.draw.rect(screen, (255, 0, 0), 
                           (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            # Update display
            pygame.display.flip()
            
            # Update state and tracking
            state = next_state
            total_reward += reward
            steps += 1
        
        # Print game results
        print(f"Game {game+1} complete:")
        print(f"  Score: {env.game.score}")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        
        # Store results
        game_scores.append(env.game.score)
        game_lengths.append(steps)
        
        # Completely shut down pygame after each game
        pygame.display.quit()
        pygame.quit()
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Average Score: {np.mean(game_scores):.2f}")
    print(f"Average Game Length: {np.mean(game_lengths):.2f}")
    print(f"Best Score: {np.max(game_scores)}")

def main():
    # Disable pygame display for training
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # Create the environment
    env = SnakeGameRLWrapper()
    
    # Create DQN agent with observation space size
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=4
    )
    
    # Choose mode: 'train' or 'play'
    mode = input("train or play?")
    
    if mode == 'train':
        print("Starting training with DQN...")
        # Train the agent
        rewards, lengths, losses, scores = train_dqn(env, agent, num_episodes=10000)
        
        # Save final training metrics
        np.savez('training_logs/final_training_metrics.npz', 
                rewards=rewards, 
                lengths=lengths, 
                losses=losses,
                scores=scores)
        
        # Plot training metrics
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        
        plt.subplot(2, 2, 2)
        plt.plot(lengths)
        plt.title('Episode Lengths')
        
        plt.subplot(2, 2, 3)
        plt.plot(scores)
        plt.title('Game Scores')
        
        plt.subplot(2, 2, 4)
        plt.plot(losses)
        plt.title('Training Losses')
        
        plt.tight_layout()
        plt.savefig('training_logs/training_metrics.png')
        
        print("Training completed!")
        
        # Ask user if they want to play with the trained model
        user_input = input("Do you want to play with the trained model? (y/n): ")
        if user_input.lower() == 'y':
            print("Starting gameplay...")
            # Remove the dummy display driver
            os.environ.pop("SDL_VIDEODRIVER", None)
            play_with_trained_model()
    
    else:  # mode == 'play'
        print("Starting gameplay with trained model...")
        # Remove the dummy display driver
        os.environ.pop("SDL_VIDEODRIVER", None)
        play_with_trained_model()

if __name__ == "__main__":
    main()