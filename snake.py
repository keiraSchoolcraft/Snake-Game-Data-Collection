import pygame
import random
import time

# Constants
GRID_SIZE = 16
CELL_SIZE = 30
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
        self.reset_game()
    
    def reset_game(self):
        self.snake = [(6, 6)]
        self.food = self.generate_food()
        self.direction = "RIGHT"
        self.running = True
    
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
        else:
            self.snake.pop()
    
    def draw_elements(self):
        self.screen.fill((0, 0, 0))
        
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        food_x, food_y = self.food
        pygame.draw.rect(self.screen, (255, 0, 0), (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
    
    def show_start_screen(self):
        font = pygame.font.Font(None, 48)
        text = font.render("Press SPACE to Play", True, (255, 255, 255))
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.fill((0, 0, 0))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
    
    def countdown(self):
        for i in range(3, 0, -1):
            self.screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 72)
            text = font.render(str(i), True, (255, 255, 255))
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            time.sleep(1)
    
    def show_game_over_screen(self):
        font = pygame.font.Font(None, 48)
        text = font.render("Game Over! Press SPACE to Play Again", True, (255, 255, 255))
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.fill((0, 0, 0))
        self.screen.blit(text, text_rect)
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
        while True:
            self.show_start_screen()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        waiting = False
            
            self.countdown()
            self.reset_game()
            
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        self.change_direction(event)
                
                self.move_snake()
                self.draw_elements()
                self.clock.tick(10)
            
            if not self.show_game_over_screen():
                break
        
        pygame.quit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()