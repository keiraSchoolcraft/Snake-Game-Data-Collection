import pandas as pd
import numpy as np
import ast # For safely evaluating the string representation of the list
import re

def stack_frames_from_csv(csv_path, k=3, board_shape=(16, 16)):
    """
    Reads Snake game data from CSV, performs 3 frame stacking,
    and handles game restarts based on score resets.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'Board State', 'Direction', 'Score'.
        k (int): Number of frames to stack (default is 3).
        board_shape (tuple): The expected shape (Height, Width) of the game board.

    Returns:
        tuple: A tuple containing:
            - stacked_states (np.ndarray): Array of stacked board states (N, k, H, W).
            - corresponding_actions (np.ndarray): Array of actions corresponding
                                                   to the last frame in each stack (N,).
               Returns (None, None) if processing fails or yields no valid data.
    """

    df = pd.read_csv(csv_path, usecols=['Board State', 'Direction', 'Score'])

    # A game starts at row 0 or when score is 0 and previous score was > 0
    # Shift(1) gets the previous row's value and handles the first row's NaN shift.
    is_new_game = (df['Score'] == 0) & (df['Score'].shift(1, fill_value=500) > 0)
    df['game_id'] = is_new_game.cumsum() - 1

    n_games = df['game_id'].nunique()
    print(f"Identified {n_games} games.")
    if n_games == 0:
        print("Error: No games identified. Check score data.")
        return None, None

    print("Parsing board states...")
    parsed_boards = []
    inner_comma_regex = re.compile(r'(?<=\d)\s+(?=\d)')
    outer_comma_regex = re.compile(r']\s+\[')
    
    for i, board_str in enumerate(df['Board State']):
        corrected_board_str = inner_comma_regex.sub(',', board_str)
        corrected_board_str = outer_comma_regex.sub('],\n[', corrected_board_str)
        board_list = ast.literal_eval(corrected_board_str)
        # convert to np array
        board_array = np.array(board_list, dtype=np.float32)
        # Validate shape
        if board_array.shape != board_shape:
             raise ValueError(f"Parsed board at index {i} has shape {board_array.shape}, expected {board_shape}")
        parsed_boards.append(board_array)

    df['parsed_board'] = parsed_boards
    print("Board states parsed successfully.")



    print(f"Grouping by game and creating stacks (k={k})...")
    all_stacked_states = []
    all_corresponding_actions = []
    games_processed_count = 0
    frames_generated_count = 0
    frames_skipped_count = 0

    grouped = df.groupby('game_id')

    for _, game_df in grouped:
        game_len = len(game_df)

        # Skip games that are too short to form even one stack
        if game_len < k:
            frames_skipped_count += game_len
            continue

        games_processed_count += 1
        # Retrieve the pre-parsed boards and actions for this game
        # Using .values is often faster than .tolist()
        game_boards = game_df['parsed_board'].values # Array of HxW arrays
        game_actions = game_df['Direction'].values

        # Iterate starting from the k-th frame (index k-1) up to the end
        # This ensures we have k frames: [t-k+1, ..., t-1, t]
        for t in range(k - 1, game_len):
            # Stack the numpy arrays for frames t-k+1 up to t
            stack = np.stack(game_boards[t - k + 1 : t + 1], axis=0) # Creates (k, H, W)

            # Get the action corresponding to the *last* frame 't'
            action = game_actions[t]

            all_stacked_states.append(stack)
            all_corresponding_actions.append(action)
            frames_generated_count += 1

        # Add the first k-1 frames skipped at the beginning of this game
        frames_skipped_count += (k - 1)

    
    
    print("-" * 30)
    print(f"Processed {games_processed_count} games.")
    print(f"Generated {frames_generated_count} stacked frames.")
    print(f"Skipped {frames_skipped_count} frames (from game starts / short games).")

    if not all_stacked_states:
        print("Warning: No stacked frames were generated. Check data or k value.")
        return None, None

    # Convert the lists of results into single large NumPy arrays
    try:
        stacked_states_np = np.array(all_stacked_states, dtype=np.float32)
        # Ensure actions have a consistent type, e.g., integer
        corresponding_actions_np = np.array(all_corresponding_actions, dtype=np.int64)
    except Exception as e:
        print(f"Error converting results to NumPy arrays: {e}")
        return None, None

    print(f"Output shapes: States {stacked_states_np.shape}, Actions {corresponding_actions_np.shape}")
    print("-" * 30)

    return stacked_states_np, corresponding_actions_np

if __name__ == "__main__":
    
    csv_path = "filtered_snake_data.csv"
    stacked_data, actions = stack_frames_from_csv(csv_path, k=3, board_shape=(16, 16))
    
    # Save using np file format
    np.savez_compressed("processed_snake_data.npz", states=stacked_data, actions=actions)

    if stacked_data is not None:
        print("\nUsage Results:")
        print("Shape of stacked_data:", stacked_data.shape)
        print("Shape of actions:", actions.shape)
        print("First stacked state (shape):", stacked_data[0].shape)
        print("First action:", actions[0])
    else:
        print("No data returned from stack_frames_from_csv.")