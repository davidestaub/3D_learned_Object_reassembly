# Config file for the neural network architecture and training.
# There are a lot of different settings to change the architecture and experiment. Before changing network related 
# parameters, better make a copy of the file. The current settings were the best ones found by Bayesian Optimization.

data_conf = {
    'desc': 'pillar',       # The used descriptors, used by the dataloader. Possibilities: [fpfh, pillar, fpfh_pillar]
    'kpts': 'hybrid',       # The used keypoints, used by the dataloader. Possibilities: [hybrid, sd, sticky]
    'gt_match_thresh': 0.0, # The threshold for matching inliers in percent considered by the dataloader (calculated from ground truth)
}


model_conf = {
    'weights': None,        # Path to weights to load. If None is specified, the networks is trained from scratch
    'pillar': True,         # Activates the pillar architecture by using a neighborhood encoder and a keypoint encoder 
    'descriptor_dim': 36,   # The descriptor dimension, must be dividable by num heads!
    'num_heads': 6,         # Number of attention heads
    'sep_encoder': True,    # Using separate encoders for both fragments
    'keypoint_encoder': [8, 16, 32, 64], # Hidden Layer dimensions of the MLP. The input and output dimensions are correctly set automatically
    'GNN_layers': 3,        # Number of GNN layers, where each layer is composed of self-attention followed by cross-attention
    'sinkhorn_iterations': 320, # Number of sinkhorn iterations
    'match_threshold': 0.28,    # Sensitivity of the network for predicting a match
    'nll_weight': 1.,           # Weight for negative log-likelihood. The NLL*nll_weights is the total loss
    'nll_balancing': 0.96,      # Balancing of the negative and positive matches. The loss is composed of nll_balance * pos_matches - (1-nll_balance) * negative matches
}


train_conf = {
    'seed': 42,                 # Random seed
    'epochs': 3,                # Number of training epochs
    'batch_size': 1,            # Training Batch size. WARNING, a lot of vram is used!
    'optimizer': 'adam',        # Optimizer used. Possibilities: [adam, sgd, rmsprop]
    'lr':0.004,                 # Learning rate used by the optimizer
    'lr_schedule': {'type': 'exp', 'start': 10e3, 'exp_div_10': 2e5}, # Learning rate scheduler. Only exp is supported atm.
    'overfit': False,           # If set to true, the network overfits on the very first batch
    'use_sd_score': False,      # Wheter to use a the score provided in the keypoints npy file.
    'match_inverted': False,    # Wheter to use inverted fpfh features.
    'train_fraction': 0.9,      # Splitting of train and validation set
    'normalize_data': True,     # Centering the pointcloud.
}


wandb_conf = {       
    'key': None,        # Your API key for wandb logging. If None is provided, wandb is not used.
    'entity': None,     # The entity to login on wandb
    'project': None     # The wandb project name
}