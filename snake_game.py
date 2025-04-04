import pygame
import random
import time
import numpy as np
import sys
import os

from sheets_logger import GoogleSheetsLogger
from config import *

# Calculate dimensions
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

class SnakeGame:
    def __init__(self):
        pygame.init()
        
        # Create window with custom title bar
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + 30), pygame.NOFRAME)
        pygame.display.set_caption("16x16 Snake Game")
        
        # Create game surface
        self.game_surface = pygame.Surface((WIDTH, HEIGHT))
        
        # Title bar elements
        self.title_font = pygame.font.Font(None, 24)
        self.title_rect = pygame.Rect(0, 0, WIDTH, 30)
        
        # Close button
        self.close_rect = pygame.Rect(WIDTH - 30, 0, 30, 30)
        
        # Minimize button
        self.minimize_rect = pygame.Rect(WIDTH - 60, 0, 30, 30)
        
        self.clock = pygame.time.Clock()
        self.running = False
        
        # Track window dragging
        self.dragging = False
        self.drag_offset = (0, 0)
        
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
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Title bar interactions
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check close button
                if self.close_rect.collidepoint(event.pos):
                    return False
                
                # Check minimize button
                if self.minimize_rect.collidepoint(event.pos):
                    pygame.display.iconify()
                
                # Start dragging if clicked on title bar
                if self.title_rect.collidepoint(event.pos):
                    self.dragging = True
                    mouse_x, mouse_y = event.pos
                    window_x, window_y = pygame.display.get_window_position()
                    self.drag_offset = (mouse_x - window_x, mouse_y - window_y)
            
            # Handle mouse up to stop dragging
            if event.type == pygame.MOUSEBUTTONUP:
                self.dragging = False
            
            # Handle mouse motion for dragging
            if event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    mouse_x, mouse_y = event.pos
                    pygame.display.set_mode((WIDTH, HEIGHT + 30), pygame.NOFRAME)
                    pygame.display.move_window(
                        mouse_x - self.drag_offset[0], 
                        mouse_y - self.drag_offset[1]
                    )
            
            # Game-specific events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return 'SPACE'
                elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    return event
        
        return True
    
    def draw_title_bar(self):
        # Title bar background
        pygame.draw.rect(self.screen, (50, 50, 50), self.title_rect)
        
        # Title text
        title_text = self.title_font.render("Snake Game", True, (200, 200, 200))
        self.screen.blit(title_text, (10, 5))
        
        # Close button
        pygame.draw.rect(self.screen, (220, 50, 50), self.close_rect)
        close_text = self.title_font.render("X", True, (255, 255, 255))
        close_pos = (WIDTH - 20, 5)
        self.screen.blit(close_text, close_pos)
        
        # Minimize button
        pygame.draw.rect(self.screen, (100, 100, 100), self.minimize_rect)
        minimize_text = self.title_font.render("-", True, (255, 255, 255))
        minimize_pos = (WIDTH - 50, 5)
        self.screen.blit(minimize_text, minimize_pos)
    
    def draw_game(self):
        # Clear the game surface
        self.game_surface.fill(BLACK)
        
        # Draw snake body (all segments except head)
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # Snake head
                pygame.draw.rect(self.game_surface, DARK_GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            else:  # Snake body
                pygame.draw.rect(self.game_surface, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw food
        food_x, food_y = self.food
        pygame.draw.rect(self.game_surface, RED, (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Display score
        font = pygame.font.Font(None, 24)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.game_surface.blit(score_text, (10, 10))
        
        # Combine game surface with title bar
        self.screen.fill((30, 30, 30))  # Background color
        self.screen.blit(self.game_surface, (0, 30))
        self.draw_title_bar()
        
        pygame.display.flip()
    
    def show_start_screen(self):
        self.screen.fill((30, 30, 30))
        
        font = pygame.font.Font(None, 48)
        text = font.render("Press SPACE to Play", True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 15))
        
        self.screen.blit(text, text_rect)
        self.draw_title_bar()
        
        pygame.display.flip()
    
    def countdown(self):
        for i in range(3, 0, -1):
            self.screen.fill((30, 30, 30))
            font = pygame.font.Font(None, 72)
            text = font.render(str(i), True, WHITE)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 15))
            self.screen.blit(text, text_rect)
            self.draw_title_bar()
            pygame.display.flip()
            time.sleep(1)
    
    def show_game_over_screen(self):
        # Save game data before showing game over screen
        if self.logging_enabled:
            self.logger.save_game_data()

        self.screen.fill((30, 30, 30))

        # Game Over title (smaller font)
        font_game_over = pygame.font.Font(None, 36)
        text_game_over = font_game_over.render("Game Over!", True, WHITE)
        text_game_over_rect = text_game_over.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 35))
        
        # Instructions
        font_instructions = pygame.font.Font(None, 24)
        text_instructions = font_instructions.render("Press SPACE to Play Again", True, WHITE)
        text_instructions_rect = text_instructions.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 15))
        
        # Score display
        score_text = font_instructions.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 65))
        
        # Optional logging message
        log_text = font_instructions.render("Game data saved!" if self.logging_enabled else "", True, GREEN)
        log_rect = log_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 115))
        
        # Render all texts
        self.screen.blit(text_game_over, text_game_over_rect)
        self.screen.blit(text_instructions, text_instructions_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(log_text, log_rect)
        
        self.draw_title_bar()
        pygame.display.flip()
        
        waiting = True
        while waiting:
            event = self.handle_events()
            
            if event is False:
                pygame.quit()
                return False
            
            if event == 'SPACE':
                waiting = False
                return True
        
        return False
    
    def run(self):
        try:
            while True:
                self.show_start_screen()
                
                waiting = True
                while waiting:
                    event = self.handle_events()
                    if event is False:
                        pygame.quit()
                        if self.logging_enabled:
                            self.logger.close()
                        return
                    
                    if event == 'SPACE':
                        waiting = False
                
                self.countdown()
                self.reset_game()
                
                while self.running:
                    event = self.handle_events()
                    
                    if event is False:
                        pygame.quit()
                        if self.logging_enabled:
                            self.logger.close()
                        return
                    
                    if isinstance(event, pygame.event.Event) and event.type == pygame.KEYDOWN:
                        self.change_direction(event)
                    
                    self.move_snake()
                    self.draw_game()
                    
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