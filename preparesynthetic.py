# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset  # huggingface datasets
import pickle
import torch

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc


INPUT_FILE_PATH = "checkersengine/checkers/merged_file.txt"
OUTPUT_CSV_PATH = "data/checkers_games/output.csv"
MODEL_PICKLE_NAME = "data/checkers_games/meta.pkl"
DATA_BIN_PATH = "data/checkers_games"

if __name__ == "__main__":
    

    # Define the path to your text file and the output CSV file
    input_file_path = INPUT_FILE_PATH
    output_csv_path = OUTPUT_CSV_PATH

    # Initialize a list to hold all transcripts
    transcripts = []

    # Read each line from the text file
    with open(input_file_path, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace and check if line is not empty
            cleaned_line = line.strip()
            if cleaned_line:
                # Add line to transcripts list
                transcripts.append(cleaned_line)

    # Create a DataFrame
    df = pd.DataFrame(transcripts, columns=['transcript'])
    
    def trim_to_400(text):
        if len(text) > 400:
            return text[:400]
        else:
            return text
    
    def pad_to_400(text):
        if len(text) < 400:
            padding_length = 400 - len(text)
            return text + ' ' * padding_length
        else:
            return text

    
    # Trim
    df['transcript'] = df['transcript'].apply(trim_to_400)
    df = df[df['transcript'].apply(len) >= 100]
    # Apply the function to pad the strings in the 'transcripts' column
    df['transcript'] = df['transcript'].apply(pad_to_400)
    
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    

    with open(INPUT_FILE_PATH, 'r') as f:
        data = f.read()
        dataset = data.split('\n')
    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

        
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    





    # Load the dataset
    dataset = load_dataset("csv", data_files={"train": OUTPUT_CSV_PATH})
    #dataset = load_dataset("csv", data_files="output.csv")
    # by default only contains the 'train' split, so create a test split
    
    split_dataset = dataset["train"].train_test_split(
        test_size=0.01, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 
    #     })
    # })
    
    our_pickle = {'vocab_size':16, 'itos':itos, 'stoi':stoi}
    # Pickle the dictionary
    with open(MODEL_PICKLE_NAME, 'wb') as f:
        pickle.dump(our_pickle, f)

    
    column_name = "transcript"

    def process(example):
        ids = np.array([stoi[c] for c in example[column_name]], dtype=dtype)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        #filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        filename = os.path.join(DATA_BIN_PATH, f"{split}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        print(arr.shape)
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            # print(batch[0])
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

