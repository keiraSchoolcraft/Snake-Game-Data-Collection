First install the prereqs in requirements.txt
Then download the data from the google sheet (adjusting with the filter you want,
I was getting good results with filtering for 10-20+ score)
Run clean_data.py to remove the last 5 moves from each game (feel free to adjust)
Run train_model.py with the filtered dataset to train it, then run ai_main.py to 
have it play the game.

IMPORTANT***
Currently the snake_game.py ai move choice is a little scuffed because I was trying
to fix the going backwards problem, so if it would go backwards instead it turns left
get_ai_directions2 has the original version of the function. There is some printing
for debugging the actual values returned by the softmax.
