# chess_llm_interpretability
This evaluates LLMs trained on PDN format Checkers games through the use of linear probes. We can check the LLMs internal understanding of board state.

This repo can train, evaluate, and visualize linear probes on LLMs that have been trained to play Checkers with PDN strings. 

# Setup

```
python model_setup.py
```

Split the dataset for training the probes
```
python data/split_traintest.py
```

The `train_test_checkers.py` script can be used to either train new linear probes or test a saved probe on the test set.

Command line arguments:

--mode: Specifies `train`  or `test`. Optional, defaults to `train`.

--probe: Determines the type of probe to be used. `piece` probes for the piece type on each square. Optional, defaults to `piece`. We only have piece in Checkers


Examples:

Train piece board state probes:
`python train_test_checkers.py`


See all options: `python train_test_checkers.py -h`
# Visualization
    Run all cells in the chess_llm_interpretability/probe_output_visualization.ipynb for visualizing the probe_outputs.
# Useful links


# References

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. Here are the main references used:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability