import pandas as pd
import torch
from torch.nn import functional as F
from typing import Callable, Optional
from dataclasses import dataclass
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
from enum import Enum
from checkers.game import Game

PIECE_TO_ONE_HOT_MAPPING = {
    -1:0,
    0:1,
    1:2
}
# BLANK_INDEX = PIECE_TO_ONE_HOT_MAPPING[0]
ONE_HOT_TO_PIECE_MAPPING = {value: key for key, value in PIECE_TO_ONE_HOT_MAPPING.items()}


def board_to_random_state() -> torch.Tensor:
    """Given a checkers board object, return a 8x8 torch.Tensor.
    Every square should be randomly assigned to 1, -1, or 0.
    This is to sanity check the linear probe."""
    state = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        state[i // 8, i % 8] = torch.randint(-1, 2, (1,))

    return state


def board_to_skill_state(board: Game, skill: float) -> torch.Tensor:
    """Given a checkers board object, return a 1x1 torch.Tensor.
    The 1x1 array should tell what skill level the player is."""
    state = torch.zeros((1, 1), dtype=torch.int)
    state[0][0] = skill

    return state


#Gives the x,y of the checkers board corrosponding to the index.
move_translater = {
    1 : (7,6), 2 : (7,4), 3 : (7,2), 4 : (7,0),
    5 : (6,7), 6 : (6,5), 7 : (6,3), 8 : (6,1),
    9 : (5,6), 10: (5,4), 11: (5,2), 12: (5,0),
    13: (4,7), 14: (4,5), 15: (4,3), 16: (4,1),
    17: (3,6), 18: (3,4), 19: (3,2), 20: (3,0),
    21: (2,7), 22: (2,5), 23: (2,3), 24: (2,1),
    25: (1,6), 26: (1,4), 27: (1,2), 28: (1,0),
    29: (0,7), 30: (0,5), 31: (0,3), 32: (0,1)
    }

def board_to_piece_color_state(game) -> torch.Tensor:
    """Given a checkers board object, return a 8x8 torch.Tensor.
    The 8x8 array should tell if each square is black, white, or blank.
    White is 1, black is -1, and blank is 0.
    """
    state = torch.zeros((8, 8), dtype=torch.int)
    text_list = {}
    for piece in game.board.pieces:
	    if piece.position not in text_list and piece.position != None:
		    text_list[piece.position] = piece.player                 
    
    for key in text_list:
        x, y = move_translater[key]
        state[x][y] = 1 if text_list[key] == 1 else -1
    
    return state


def board_to_piece_state(game) -> torch.Tensor:
    """Given a checkers board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell what piece is on each square. A white piece could be 1, a black piece could be -1, etc.
    Blank squares should be 0.
    """

    return board_to_piece_color_state(game)


#Function for parsing multi step move
def multiple_moves(moves_split):
    new_moves = moves_split[:2]
    moves_split = moves_split[2:]
    for move in moves_split:
        new_moves.append(new_moves[-1])
        new_moves.append(move)
    return new_moves


def pgn_string_to_board(pgn_string: str) -> Game:
    """Convert a PDN string to a checkers Game object.
    We are making an assumption that the pdn string is in this format:
    ;1. 11-15 24-20 2. 8-11 28-24"""
    
    #print(pgn_string)
    loaded_list = pgn_string[1:].strip()
    loaded_list = loaded_list.split()
    #Remove the last string as it would be either indicating victory or it would be incomplete.
    loaded_list = loaded_list[:-1]
    #Remove the numbering of the moves
    loaded_list = [item for item in loaded_list if '.' not in item]

    game = loaded_list
    checker = Game()
    print("*"*100)
    print(game)
    print("*"*100)
    for moves in game:
        if '-' in moves:
            moves_split = moves.split('-')
        else:
            moves_split = moves.split('x')
        if len(moves_split) == 2:
            x = int(moves_split[0])
            y = int(moves_split[1])
            checker.move([x,y])
        else:
            moves_split = multiple_moves(moves_split) 
            itr_max = len(moves_split)
            itr = 0
            while(itr<itr_max):
                x = int(moves_split[itr])
                y = int(moves_split[itr+1])
                checker.move([x,y])
                itr+=2
    return checker


def create_state_stack(
    moves_string: str,
    custom_board_to_state_fn: Callable[[Game], torch.Tensor],
    skill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Given a string of PDN format moves, create an 8x8 torch.Tensor for every character in the string."""

    
    board = Game()
    initial_states = []
    count = 1

    # Scan 1: Creates states, with length = number of moves in the game
    initial_states.append(custom_board_to_state_fn(board)) #board, skill
    # Apply each move to the board
    moves_expander = moves_string
    moves_string = moves_string.strip()
    moves_string = moves_string.split()
    #Remove the last string as it would be either indicating victory or it would be incomplete.
    moves_string = moves_string[:-1]
    #Remove the numbering of the moves
    moves_string = [item for item in moves_string if '.' not in item]

    for moves in moves_string:
        if '-' in moves:
            moves_split = moves.split('-')
        else:
            moves_split = moves.split('x')
        if len(moves_split) == 2:
            x = int(moves_split[0])
            y = int(moves_split[1])
            if [x,y] in board.get_possible_moves():
                board.move([x,y])
                initial_states.append(custom_board_to_state_fn(board)) # (board, skill)
            else:
                break
        else:
            moves_split = multiple_moves(moves_split) 
            itr_max = len(moves_split)
            itr = 0
            while(itr<itr_max):
                x = int(moves_split[itr])
                y = int(moves_split[itr+1])
                if [x,y] in board.get_possible_moves():
                    board.move([x,y])
                    initial_states.append(custom_board_to_state_fn(board)) # (board, skill)
                else:
                    break
                itr+=2
    
    # Second Scan: Expand states to match the length of moves_string
    #For ;1. 11-15 23-19 2. 8-11 22-17 3. 11-16 24-20
    # ;1. 11-15 = idx 0 , 23-19 2 = idx 1, . 8-11 = idx 2
    # If I find x and immediately after 2/3 chars I again find x that means its the same move of that player increment move index

    expanded_states = []
    move_index = 0

    toogle21 = False
    hacky_first = 0
    for i, char in enumerate(moves_expander):
        if char == "x":
            if (i+2 < len(moves_expander) and moves_expander[i+2] == "x") or \
            (i+3 < len(moves_expander) and moves_expander[i+3] == "x"):
                move_index+=1
        if toogle21:
            if char == " ":
                hacky_first+=1
                if hacky_first == 2:
                    move_index += 1
                    hacky_first = 0
                    toogle21 = False
        else:
            if char == ".":
                move_index += 1
                toogle21 = True
        expanded_states.append(initial_states[min(move_index, len(initial_states) - 1)])
                
    return torch.stack(expanded_states) #expanded_states


def create_state_stacks(
    moves_strings: list[str],
    custom_board_to_state_fn: Callable[[Game], torch.Tensor],
    skill_array: Optional[torch.Tensor] = None,
) -> Float[Tensor, "modes sample_size pgn_str_length rows cols"]:
    """Given a list of strings of PDN format moves, create a tensor of shape (len(moves_strings), 8, 8).
    custom_board_to_state is a function that takes a checkers Game object and returns a 8x8 torch.Tensor for
    board state."""
    state_stacks = []
    skill = None

    for idx, pgn_string in enumerate(moves_strings):
        if skill_array is not None:
            skill = skill_array[idx]
        state_stack = create_state_stack(pgn_string, custom_board_to_state_fn, skill)
        state_stacks.append(state_stack)

    # Convert the list of tensors to a single tensor
    final_state_stack = torch.stack(state_stacks)
    final_state_stack = final_state_stack.unsqueeze(0)  # Add a dimension for the modes
    # Currently, there is just one mode and it isn't necessary. For now, I'm maintaining the dimension for future use.
    return final_state_stack


def state_stack_to_one_hot(
    num_modes: int,
    num_rows: int,
    num_cols: int,
    min_val: int,
    max_val: int,
    device: torch.device,
    state_stack: torch.Tensor,
    user_mapping: Optional[dict[int, int]] = None,
) -> Int[Tensor, "modes sample_size num_white_moves rows cols one_hot_range"]:
    """Input shape: assert(state_stacks_all_chars.shape) == (modes, sample_size, game_length, rows, cols)
    Output shape: assert(state_stacks_one_hot.shape) == (modes, sample_size, game_length, rows, cols, one_hot_range)
    """
    range_size = max_val - min_val + 1

    mapping = {}
    if user_mapping:
        mapping = user_mapping
        min_val = min(mapping.values())
        max_val = max(mapping.values())
        range_size = max_val - min_val + 1
    else:
        for val in range(min_val, max_val + 1):
            mapping[val] = val - min_val

    # Initialize the one-hot tensor
    one_hot = torch.zeros(
        state_stack.shape[0],  # num modes
        state_stack.shape[1],  # num games
        state_stack.shape[2],  # num moves
        num_rows,
        num_cols,
        range_size,
        device=device,
        dtype=torch.int,
    )

    for val in mapping:
        one_hot[..., mapping[val]] = state_stack == val

    return one_hot


def one_hot_to_state_stack(one_hot: torch.Tensor, min_val: int) -> torch.Tensor:
    """Input shape: assert(probe_out.shape) == (modes, sample_size, num_white_moves, rows, cols, one_hot_range)
    Output shape: assert(state_stacks_probe_outputs.shape) == (modes, sample_size, num_white_moves, rows, cols)
    """
    indices = torch.argmax(one_hot, dim=-1)
    state_stack = indices + min_val
    return state_stack


def square_to_coordinate(square) -> tuple[int, int]:
    row = move_translater[square][0]
    column = move_translater[square][1]
    return (row, column)


def find_dots_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return indices


def find_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every ' ' in the string."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    return indices


def find_odd_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of odd indices of every ' ' in the string.
    There is some duplicated logic but it simplifies using the Callable function."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    # Select only the odd indices: start from index 1, go till the end, step by 2
    odd_indices = indices[1::2]
    return odd_indices


def find_even_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of even indices of every ' ' in the string.
    There is some duplicated logic but it simplifies using the Callable function."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    # Select only the even indices: start from index 0, go till the end, step by 2
    even_indices = indices[::2]
    return even_indices


def find_dots_indices_offset_one(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]

    incremented_indices = [index + 1 for index in indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_even_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of even indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    even_indices = indices[::2]

    # Increment each even index by one, ensuring it doesn't exceed the string length
    incremented_indices = [index + 1 for index in even_indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_odd_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of odd indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    odd_indices = indices[1::2]

    # Increment each odd index by one, ensuring it doesn't exceed the string length
    incremented_indices = [index + 1 for index in odd_indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_custom_indices(
    custom_indexing_fn: Callable[[str], list[int]], df: pd.DataFrame
) -> torch.Tensor:
    indices_series = df["transcript"].apply(custom_indexing_fn)
    shortest_length = indices_series.apply(len).min()
    print("Shortest length:", shortest_length)

    indices_series = indices_series.apply(lambda x: x[:shortest_length])
    assert all(
        len(lst) == shortest_length for lst in indices_series
    ), "Not all lists have the same length"

    indices = torch.tensor(indices_series.apply(list).tolist(), dtype=torch.int)
    return indices


def encode_string(meta: dict, s: str) -> list[int]:
    """Encode a string into a list of integers."""
    stoi = meta["stoi"]
    return [stoi[c] for c in s]


def decode_list(meta: dict, l: list[int]) -> str:
    """Decode a list of integers into a string."""
    itos = meta["itos"]
    return "".join([itos[i] for i in l])


# Adapted from nanogpt
def get_model_move(
    model,
    meta: dict,
    idx: torch.Tensor,
    max_new_tokens: int = 7,
    temperature=1.0,
    block_size=399,
):
    """Generate new tokens from a trained language model. If temperature is 0.0, greedy decoding is used.
    Otherwise, standard temperature based sampling is used."""

    if temperature < 0:
        raise ValueError("temperature has to be non-negative")

    input_length = len(idx[0])
    space_idx = encode_string(meta, " ")[0]
    #If we encounter 2 spaces we have successfully parsed a move.
    quit_on_2space = 0
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            if temperature == 0.0:
                # greedy decoding
                # model(idx_cond) is a tensor of shape (batch_size, sequence_length, vocab_size)
                # logits is a tensor of shape (batch_size, vocab_size)
                # idx_next is a tensor of shape (batch_size, 1)
                logits = model(idx_cond)[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                # forward the model to get the logits for the index in the sequence
                logits = model(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(idx_cond)
                    print(logits)
                    raise ValueError("Logits contain NaNs or Infs")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next[0] == space_idx:
                quit_on_2space+=1
            if quit_on_2space ==2:
                break
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

    model_response = idx[:, input_length:]
    model_move = decode_list(meta, model_response[0].tolist())
    return model_move.strip()


class PlayerColor(Enum):
    WHITE = "White"
    BLACK = "Black"


@dataclass
class Config:
    min_val: int
    max_val: int
    custom_board_state_function: callable
    linear_probe_name: str
    custom_indexing_function: callable = find_dots_indices
    num_rows: int = 8
    num_cols: int = 8
    levels_of_interest: Optional[list[int]] = None
    column_name: str = None
    probing_for_skill: bool = False
    # pos_start indexes into custom_indexing_function. Example: if pos_start = 25, for find_dots_indices, selects everything after the first 25 moves
    pos_start: int = 0
    # If pos_end is None, it's set to the length of the shortest game in construct_linear_probe_data()
    pos_end: Optional[int] = None
    player_color: PlayerColor = PlayerColor.WHITE  #.WHITE


piece_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_piece_state,
    linear_probe_name="checkers_piece_probe",
)

#Chess specific
color_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_piece_color_state,
    linear_probe_name="chess_color_probe",
)


random_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_random_state,
    linear_probe_name="chess_random_probe",
)


skill_config = Config(
    min_val=-2,
    max_val=20,
    custom_board_state_function=board_to_skill_state,
    linear_probe_name="chess_skill_probe",
    num_rows=1,
    num_cols=1,
    levels_of_interest=[0, 5],
    probing_for_skill=True,
    pos_start=25,
)


def find_config_by_name(config_name: str) -> Config:
    """
    Finds and returns the Config instance with a matching linear_probe_name.
    """
    all_configs = [piece_config, color_config, random_config, skill_config]
    for config in all_configs:
        if config.linear_probe_name == config_name:
            return config
    raise ValueError(f"Config with name {config_name} not found")


def update_config_using_player_color(player_color: PlayerColor, config: Config) -> Config:
    """Player color will determine which indexing function we use. In addition, we set player to white by default.
    If player is black, then we update the probe name as well."""

    #PlayerColor.WHITE by default
    if player_color == PlayerColor.WHITE:
        config.custom_indexing_function = find_dots_indices
        config.player_color = PlayerColor.WHITE

    if player_color == PlayerColor.BLACK:
        config.linear_probe_name = config.linear_probe_name.replace("probe", "black_player_probe")
        config.custom_indexing_function = find_even_spaces_indices
        config.player_color = PlayerColor.BLACK

    return config


def set_config_min_max_vals_and_column_name(
    config: Config,
    input_dataframe_file: str,
    dataset_prefix: str,
) -> Config:
    if config.levels_of_interest is not None or config.probing_for_skill:
        if dataset_prefix == "stockfish_":
            config.column_name = "player_two"
        elif "lichess_" in dataset_prefix:
            config.column_name = "WhiteEloBinIndex"
    else:
        # We only probe for piece and not skill
        print("ALWAYS HERE RIGHT?")
        return config
    df = pd.read_csv(input_dataframe_file)
    config.min_val = df[config.column_name].min()
    config.max_val = df[config.column_name].max()

    return config
