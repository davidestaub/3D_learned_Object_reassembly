data_conf = {
    'desc': 'fpfh', # [fpfh, pillar, fpfh_pillar]
    'kpts': 'hybrid', # [hybrid, sd]
    'inference': False
}

model_conf = {
    'pillar': True,
    'descriptor_dim': 36, # the descriptor dimension, must be dividable by num heads!
    'num_heads': 6, # num of heads
    'sep_encoder': True, #sepparate encoders
    'weights': 'weights_01',
    'keypoint_encoder': [8, 16, 32, 64], # intermediate mlp dimensions. The first is automatically set to 4, last to descriptor_dim
    'GNN_layers': 3,
    'sinkhorn_iterations': 321,
    'match_threshold': 0.245,
    'nll_weight': 1000.,
    'nll_balancing': 0.96,
}

train_conf = {
    'seed': 42,  # training seed
    'epochs': 1000,  # number of epochs
    'batch_size': 2,  # training batch size
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr':0.0085,  # learning rate
    'lr_schedule': {'type': 'exp', 'start': 10e3, 'exp_div_10': 1e5},
    'eval_every_iter': 500,  # interval for evaluation on the validation set
    'log_every_iter': 500,  # interval for logging the loss to the console
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'output_dir': "output", # the 
    'load_weights': False,
    'overfit': False,
    'use_sd_score': False,
    'match_inverted': False,
    'train_fraction': 0.9,
    'normalize_data': True, # normalizing the pointcloud
}