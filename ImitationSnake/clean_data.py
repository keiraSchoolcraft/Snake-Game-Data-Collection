import pandas as pd

def remove_last_five_moves(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # remove the last 5 moves of each game
    filtered_df = df.groupby('Game ID').apply(lambda group: group.iloc[:-5]).reset_index(drop=True)
    
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

input_csv = 'better_snake_data.csv'  
output_csv = 'filtered_snake_data.csv' 
remove_last_five_moves(input_csv, output_csv)