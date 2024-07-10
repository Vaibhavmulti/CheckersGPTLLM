from checkers.game import Game
import numpy as np
import random
from tqdm import tqdm
import multiprocessing

def possible_capture_move(move):
    x = move[0]
    y = move[1]
    if x<y and (((x-1) // 4) + 2) == ((y-1) //4):
        return True
    elif ((x-1) // 4)  == (((y-1) //4) +2):
        return True
    else:
        return False


def synthetic_games(game, moves):
    
    game_string = ";"
    for i in range(1, moves+1):
        game_string +=  str(i) + ". "
        for _ in range(2):
            possible_moves = game.get_possible_moves()
            random_index = random.randint(0, len(possible_moves) - 1)
            random_element = possible_moves[random_index]
            x = random_element[0]
            y = random_element[1]
            game.move([x, y])
            if game.is_over():
                return game_string
            if possible_capture_move(random_element):
                game_string+= str(x) + 'x' + str(y)
                double_capture = game.get_possible_moves()
                if (len(double_capture) != 1 or double_capture[0][0] != y):
                    game_string+=" "
                else:
                    while (len(double_capture) == 1 and double_capture[0][0] == y):
                        game.move([double_capture[0][0],double_capture[0][1]])
                        game_string+= "x"+str(double_capture[0][1])
                        y = double_capture[0][1]
                        double_capture = game.get_possible_moves()
                        if game.is_over():
                            return game_string+" "
                    game_string+=" "  
            else:
                game_string+= str(x) + '-' + str(y) + " "
    return game_string


def generate_data(filename, num_games):
    collate_games = ""
    for i, _ in enumerate(tqdm(range(num_games))):
        game = Game()
        collate_games += synthetic_games(game, 30) + '\n'
        if (i + 1)  % 1600 == 0:
            with open(filename, 'a') as file:
                file.write(collate_games)
            collate_games = ""



def main():
    num_entries = 16000000
    num_processes = 100
    chunk_size = num_entries // num_processes
    processes = []
    for i in range(num_processes):
        filename = f"datagen/data_part_{i}.txt"
        p = multiprocessing.Process(target=generate_data, args=(filename, chunk_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join() 

if __name__ == "__main__":
    main()
