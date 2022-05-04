model_conf = {
    'use_mlp': True, # using the mlp encoder, only works if pointnet is false
    'use_desc': True,
    'descriptor_dim': 32, # the descriptor dimension, must be dividable by 4!!
    'weights': 'weights_01',
    'keypoint_encoder': [32, 64, 128, 256], # intermediate mlp dimensions. The first is automatically set to 4, last to descriptor_dim
    'GNN_layers': ['self', 'cross'] * 6,
    'sinkhorn_iterations': 100,
    'match_threshold': 0.2,
    'loss': {
        'nll_weight': 1.,
        'nll_balancing': 0.999,
    },
}


train_conf = {
    'seed': 42,  # training seed
    'epochs': 1000,  # number of epochs
    'batch_size_train': 32,  # training batch size
    'batch_size_test': 32, #test batch size
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr':1e-4,  # learning rate
    'lr_schedule': {'type': 'exp', 'start': 250e3, 'exp_div_10': 50e3},
    'eval_every_iter': 500,  # interval for evaluation on the validation set
    'log_every_iter': 500,  # interval for logging the loss to the console
    'keep_last_checkpoints': 5,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'output_dir': "output", # the 
    'load_weights': False,
    'overfit': False,
    'use_sd_score': False,
    'match_inverted': False,
    'train_fraction': 0.8,
    'normalize_data': True, # normalizing the pointcloud
}