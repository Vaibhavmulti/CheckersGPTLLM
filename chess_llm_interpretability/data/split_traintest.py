DATA_DIR = "data/"
prefix = "checkers_"

import pandas as pd

df = pd.read_csv('../data/checkers_games/output.csv')
print(len(df))

df = df[df['transcript'].str.count('\.') > 13]
print(len(df))

# Split df into a train and test split
train = df.sample(frac=0.5, random_state=200)
test = df.drop(train.index)

print(len(train))
print(len(test))

# Save the train and test splits to csv
train.to_csv(f'{DATA_DIR}{prefix}train.csv', index=False)
test.to_csv(f'{DATA_DIR}{prefix}test.csv', index=False)