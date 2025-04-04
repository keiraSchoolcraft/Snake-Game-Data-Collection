import pygame
import random
import time
import numpy as np
from sheets_logger import GoogleSheetsLogger
from config import *

# Calculate dimensions
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("16x16 Snake Game")
        self.clock = pygame.time.Clock()
        self.running = False
        
        # Initialize Google Sheets logger
        self.logger = GoogleSheetsLogger()
        self.logging_enabled = self.logger.setup()
        
        self.reset_game()
    
    def reset_game(self):
        self.snake = [(6, 6)]
        self.food = self.generate_food()
        self.direction = "RIGHT"
        self.running = True
        self.base_speed = FPS_BASE  # Starting with a slower speed
        self.score = 0      # Initialize score
        self.move_count = 0
        
        # Reset logger move counter
        if self.logging_enabled:
            self.logger.reset_counter()
            
        # Log initial state
        self.log_game_state()
    
    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food
    
    def change_direction(self, event):
        keys = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT"}
        if event.key in keys:
            new_direction = keys[event.key]
            opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            if new_direction != opposite[self.direction]:
                self.direction = new_direction
    
    def move_snake(self):
        head_x, head_y = self.snake[0]
        dx, dy = DIRECTIONS[self.direction]
        new_head = (head_x + dx, head_y + dy)
        
        if (
            new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE
        ):
            self.running = False
            return
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.food = self.generate_food()
            self.score += 1  # Increase score when food is eaten
        else:
            self.snake.pop()
            
        # Increment move counter
        self.move_count += 1
        
        # Log game state after move
        self.log_game_state()
    
    def get_board_state(self):
        # Create a numpy array to represent the board
        board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        # Add food
        food_x, food_y = self.food
        board[food_y, food_x] = FOOD
        
        # Add snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # Snake head
                board[y, x] = SNAKE_HEAD
            else:  # Snake body
                board[y, x] = SNAKE_BODY
                
        return board
    
    def log_game_state(self):
        # Skip if logging is disabled
        if not self.logging_enabled:
            return
            
        board_state = self.get_board_state()
        
        # Log the move
        self.logger.log_move(
            self.move_count,
            self.score,
            self.direction,
            board_state,
            self.snake.copy(),  # Make a copy to avoid reference issues
            self.food
        )
    
    def calculate_speed(self):
        # Calculate speed based on snake length
        # Start slower and increase with length
        # Cap at a reasonable maximum speed
        snake_length = len(self.snake)
        speed = self.base_speed + (snake_length // FPS_INCREMENT)
        return min(speed, FPS_MAX)  # Cap the maximum speed
    
    def draw_elements(self):
        self.screen.fill(BLACK)
        
        # Draw snake body (all segments except head)
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # Snake head
                pygame.draw.rect(self.screen, DARK_GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            else:  # Snake body
                pygame.draw.rect(self.screen, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw food
        food_x, food_y = self.food
        pygame.draw.rect(self.screen, RED, (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Display score
        font = pygame.font.Font(None, 24)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
    
    def show_start_screen(self):
        font = pygame.font.Font(None, 48)
        text = font.render("Press SPACE to Play", True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        
        self.screen.fill(BLACK)
        self.screen.blit(text, text_rect)
        pygame.display.flip()
    
    def countdown(self):
        for i in range(3, 0, -1):
            self.screen.fill(BLACK)
            font = pygame.font.Font(None, 72)
            text = font.render(str(i), True, WHITE)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            time.sleep(1)
    
    def show_game_over_screen(self):
        # Save game data before showing game over screen
        if self.logging_enabled:
            self.logger.save_game_data()

        self.screen.fill(BLACK)

        # Game Over title (smaller font)
        font_game_over = pygame.font.Font(None, 36)
        text_game_over = font_game_over.render("Game Over!", True, WHITE)
        text_game_over_rect = text_game_over.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        
        # Instructions
        font_instructions = pygame.font.Font(None, 24)
        text_instructions = font_instructions.render("Press SPACE to Play Again", True, WHITE)
        text_instructions_rect = text_instructions.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        
        # Score display
        score_text = font_instructions.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        
        # Optional logging message
        log_text = font_instructions.render("Game data saved!" if self.logging_enabled else "", True, GREEN)
        log_rect = log_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))
        
        # Render all texts
        self.screen.blit(text_game_over, text_game_over_rect)
        self.screen.blit(text_instructions, text_instructions_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(log_text, log_rect)
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
                    return True
        return False
    
    def run(self):
        try:
            while True:
                self.show_start_screen()
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            if self.logging_enabled:
                                self.logger.close()
                            return
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            waiting = False
                
                self.countdown()
                self.reset_game()
                
                while self.running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            if self.logging_enabled:
                                self.logger.close()
                            return
                        elif event.type == pygame.KEYDOWN:
                            self.change_direction(event)
                    
                    self.move_snake()
                    self.draw_elements()
                    
                    # Calculate and set the game speed based on snake length
                    current_speed = self.calculate_speed()
                    self.clock.tick(current_speed)
                
                if not self.show_game_over_screen():
                    break
        finally:
            # Ensure we close the logger properly
            if self.logging_enabled:
                self.logger.close()
            pygame.quit()