# chess_llm_interpretability
This evaluates LLMs trained on PDN format Checkers games through the use of linear probes. We can check the LLMs internal understanding of board state and ability to estimate the skill level of the players involved. We can also perform interventions on the model's internal board state by deleting pieces from its internal world model.

This repo can train, evaluate, and visualize linear probes on LLMs that have been trained to play Checkers with PDN strings. 

# Setup

```
pip install -r requirements.txt
python model_setup.py
```

Split the dataset for training the probes
```
python data/split_traintest.py
```

The `train_test_chess.py` script can be used to either train new linear probes or test a saved probe on the test set.

Command line arguments:

--mode: Specifies `train`  or `test`. Optional, defaults to `train`.

--probe: Determines the type of probe to be used. `piece` probes for the piece type on each square. Optional, defaults to `piece`. We only have piece in Checkers


Examples:

Train piece board state probes:
`python train_test_chess.py`


See all options: `python train_test_chess.py -h`


# Interventions

To perform board state interventions on one layer, run `python board_state_interventions.py`. It will record JSON results in `intervention_logs/`. To get better results, train a set of 8 (one per layer) board state probes using `train_test_chess.py` and rerun.


# Useful links


# References

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. Here are the main references used:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability