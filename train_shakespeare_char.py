import time

out_dir = "out-shakespeare-char"
eval_interval = 1000 #4000
eval_iters = 100
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 50  # don't print too too often

always_save_checkpoint = True

# wandb_log = False
# wandb_project = "chess-gpt-batch"
# wandb_run_name = "8layer_lichess"

wandb_log = False  # disabled by default
wandb_project = 'Checkers'
wandb_run_name = '8layer'   
#wand_run_name = "9fus93ui"
#dataset = "lichess_hf_dataset"
gradient_accumulation_steps = 4 #1
batch_size = 100
block_size = 399  # context of up to 399 tokens (because dataset block size is 400)

# baby GPT model :)
n_layer = 8 #8
n_head = 8 #8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4 #3e-4
max_iters = 600000  #600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000  # not super necessary potentially
compile = True
