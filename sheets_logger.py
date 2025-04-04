import gspread
from google.oauth2.service_account import Credentials
import numpy as np
from datetime import datetime
import os
import socket
import uuid
from config import SERVICE_ACCOUNT_FILE, BUFFER_SIZE, MASTER_SPREADSHEET_ID

# Google Sheets Scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

class GoogleSheetsLogger:
    def __init__(self, spreadsheet_name=None):
        # Path to your service account file
        self.service_account_file = SERVICE_ACCOUNT_FILE
        self.spreadsheet = None
        self.sheet = None
        self.client = None
        
        # Create a unique player ID based on machine info
        # This helps identify different players while maintaining privacy
        self.player_id = f"{socket.gethostname()}_{str(uuid.getnode())[-6:]}"
        self.game_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Store game data in memory
        self.game_data = {
            'moves': [],
            'scores': [],
            'directions': [],
            'board_states': [],
            'snake_positions': [],
            'food_positions': [],
            'timestamps': []
        }
        
        # Master spreadsheet ID from config
        self.master_spreadsheet_id = MASTER_SPREADSHEET_ID
        
    def setup(self):
        """
        Set up the Google Sheets connection and prepare the spreadsheet.
        Returns True if successful, False otherwise.
        """
        try:
            
            # Check if service account file exists
            if not os.path.exists(self.service_account_file):
                print(f"Service account file '{self.service_account_file}' not found.")
                return False
                
            # Set up credentials
            creds = Credentials.from_service_account_file(
                self.service_account_file, scopes=SCOPES)
            
            # Create client
            self.client = gspread.authorize(creds)
            
            try:
                # Open the central master spreadsheet by ID
                self.spreadsheet = self.client.open_by_key(self.master_spreadsheet_id)
                print(f"Connected to master spreadsheet: {self.spreadsheet.title}")
            except gspread.exceptions.APIError as e:
                print(f"Error accessing master spreadsheet: {e}")
                return False
            
            # Get or create a worksheet for board states
            board_states_worksheet_title = "Board_States"
            try:
                self.board_states_sheet = self.spreadsheet.worksheet(board_states_worksheet_title)
                # Check if headers exist
                values = self.board_states_sheet.get_all_values()
                if len(values) == 0:
                    self._create_board_states_headers()
            except gspread.exceptions.WorksheetNotFound:
                self.board_states_sheet = self.spreadsheet.add_worksheet(title=board_states_worksheet_title, rows=10000, cols=10)
                self._create_board_states_headers()
            
            # Get or create a worksheet for game summary data
            game_summary_worksheet_title = "Game_Summary"
            try:
                self.game_summary_sheet = self.spreadsheet.worksheet(game_summary_worksheet_title)
                # Check if headers exist
                values = self.game_summary_sheet.get_all_values()
                if len(values) == 0:
                    self._create_game_summary_headers()
            except gspread.exceptions.WorksheetNotFound:
                self.game_summary_sheet = self.spreadsheet.add_worksheet(title=game_summary_worksheet_title, rows=10000, cols=10)
                self._create_game_summary_headers()
            
            # Set the sheet for compatibility with other methods
            self.sheet = self.board_states_sheet
            
            print(f"Google Sheets setup complete. Data will be logged to the central database.")
            return True
            
        except Exception as e:
            print(f"Error setting up Google Sheets: {e}")
            return False
    
    def _create_board_states_headers(self):
        """Set up headers for the board states worksheet."""
        headers = ["Player ID", "Game ID", "Move Number", "Board State", 
                   "Direction", "Score", "Snake Positions", "Food Position", "Timestamp"]
        self.board_states_sheet.update('A1:I1', [headers])
        
    def _create_game_summary_headers(self):
        """Set up headers for the game summary worksheet."""
        headers = ["Player ID", "Game ID", "Start Time", "End Time", "Final Score", 
                   "Game Duration", "Number of Moves", "Final Snake Length"]
        self.game_summary_sheet.update('A1:H1', [headers])
    
    def reset_counter(self):
        """Reset for a new game."""
        # Generate a new game ID for the new game session
        self.game_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # Clear game data
        self.game_data = {
            'moves': [],
            'scores': [],
            'directions': [],
            'board_states': [],
            'snake_positions': [],
            'food_positions': [],
            'timestamps': []
        }
        # Record start time
        self.start_time = datetime.now()
        # Remove the game_saved attribute to allow saving
        if hasattr(self, 'game_saved'):
            delattr(self, 'game_saved')
    
    def log_move(self, move_num, score, direction, board_state, snake_positions, food_position):
        """
        Log a move to the game data collection.
        
        Args:
            move_num: The move number
            score: Current score
            direction: Current direction
            board_state: 2D numpy array representing the game board
            snake_positions: List of snake segment positions
            food_position: Food position tuple
        """
        # Add move data to the game data collection
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store data in memory for this game
        self.game_data['moves'].append(move_num)
        self.game_data['scores'].append(score)
        self.game_data['directions'].append(direction)
        self.game_data['board_states'].append(self._board_to_string(board_state))
        self.game_data['snake_positions'].append(str(snake_positions))
        self.game_data['food_positions'].append(str(food_position))
        self.game_data['timestamps'].append(timestamp)
    
    def _board_to_string(self, board_state):
        """Convert 2D numpy array to string representation."""
        return np.array2string(board_state)
    
    def save_game_data(self):
        """Save all the game data to Google Sheets when the game is complete."""
        if not self.spreadsheet:
            print("Spreadsheet not available. Game data not saved.")
            return False
            
        # Check if this game has already been saved
        if hasattr(self, 'game_saved') and self.game_saved:
            print(f"Game {self.game_id} already saved, skipping.")
            return True
            
        try:
            # Calculate end time and game duration
            end_time = datetime.now()
            game_duration = (end_time - self.start_time).total_seconds()
            
            # Get final score and snake length
            final_score = self.game_data['scores'][-1] if self.game_data['scores'] else 0
            final_snake_length = len(eval(self.game_data['snake_positions'][-1])) if self.game_data['snake_positions'] else 0
            
            # 1. First, save the game summary
            game_summary_row = [
                self.player_id,                               # Player ID
                self.game_id,                                 # Game ID
                self.start_time.strftime("%Y-%m-%d %H:%M:%S"),# Start Time
                end_time.strftime("%Y-%m-%d %H:%M:%S"),       # End Time
                final_score,                                  # Final Score
                game_duration,                                # Game Duration (seconds)
                len(self.game_data['moves']),                 # Number of Moves
                final_snake_length                            # Final Snake Length
            ]
            
            # Get next row for game summary
            try:
                next_summary_row = len(self.game_summary_sheet.get_all_values()) + 1
                # Use a specific range for the update
                cell_range = f'A{next_summary_row}:H{next_summary_row}'
                self.game_summary_sheet.update(cell_range, [game_summary_row])
                print(f"Game summary saved. Game ID: {self.game_id}")
            except Exception as e:
                print(f"Error saving game summary: {e}")
            
            # 2. Now save all the individual board states
            board_states_rows = []
            for i in range(len(self.game_data['moves'])):
                try:
                    board_states_rows.append([
                        self.player_id,                     # Player ID
                        self.game_id,                       # Game ID
                        self.game_data['moves'][i],         # Move Number
                        self.game_data['board_states'][i],  # Board State
                        self.game_data['directions'][i],    # Direction
                        self.game_data['scores'][i],        # Score
                        self.game_data['snake_positions'][i], # Snake Positions
                        self.game_data['food_positions'][i],  # Food Position
                        self.game_data['timestamps'][i]       # Timestamp
                    ])
                except IndexError:
                    # In case some lists have different lengths
                    continue
            
            if board_states_rows:
                try:
                    # Batch upload board states in chunks to avoid exceeding API limits
                    # Google Sheets API has a limit of 500 rows per update
                    chunk_size = 400
                    for i in range(0, len(board_states_rows), chunk_size):
                        chunk = board_states_rows[i:i+chunk_size]
                        next_board_row = len(self.board_states_sheet.get_all_values()) + 1
                        end_row = next_board_row + len(chunk) - 1
                        range_str = f'A{next_board_row}:I{end_row}'
                        self.board_states_sheet.update(range_str, chunk)
                        print(f"Board states batch saved: {i} to {i+len(chunk)}")
                except Exception as e:
                    print(f"Error saving board states: {e}")
            
            print(f"All game data saved successfully. Game ID: {self.game_id}, total moves: {len(self.game_data['moves'])}")
            
            # Mark this game as saved so we don't save it again
            self.game_saved = True
            return True
            
        except Exception as e:
            print(f"Error saving game data: {e}")
            return False
    
    def close(self):
        """Save game data and perform cleanup."""
        try:
            if len(self.game_data['moves']) > 0:
                self.save_game_data()
        except Exception as e:
            print(f"Error in close method: {e}")
        finally:
            print("Google Sheets logger closed.")