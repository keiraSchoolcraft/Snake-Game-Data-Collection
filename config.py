# Game settings
GRID_SIZE = 16
CELL_SIZE = 30
FPS_BASE = 5
FPS_MAX = 20
FPS_INCREMENT = 5  # Snake length divisor for speed increase

# Colors
DARK_GREEN = (0, 100, 0)  # Dark green for snake head
GREEN = (0, 255, 0)       # Green for snake body
RED = (255, 0, 0)         # Red for food
BLACK = (0, 0, 0)         # Black for background
WHITE = (255, 255, 255)   # White for text

# Google Sheets settings
SPREADSHEET_NAME = "Snake Game Data"
BUFFER_SIZE = 10  # Number of moves to buffer before writing to sheets
SERVICE_ACCOUNT_FILE = "service_account.json"
MASTER_SPREADSHEET_ID = "1wYXkTA0A2c4Ha23tXiLOhHZzcm904g_tS-g-9Ov62dU"  # ID of the centralized spreadsheet

# Board state representation
EMPTY = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2
FOOD = 3