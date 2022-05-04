model_conf = {
    'descriptor_dim': 36, # the descriptor dimension, must be dividable by num heads!
    'num_heads': 4, # num of heads
    'sep_encoder': False, #sepparate encoders
    'weights': 'weights_01',
    'keypoint_encoder': [32, 64, 128, 256], # intermediate mlp dimensions. The first is automatically set to 4, last to descriptor_dim
    'GNN_layers': ['self', 'cross'] * 6,
    'sinkhorn_iterations': 100,
    'match_threshold': 0.3,
    'loss': {
        'nll_weight': 1.,
        'nll_balancing': 0.999,
    },
}

hyperpillar = True

train_conf = {
    'seed': 42,  # training seed
    'epochs': 1000,  # number of epochs
    'batch_size_train': 1,  # training batch size
    'batch_size_test': 1, #test batch size
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr':1e-4,  # learning rate
    'lr_schedule': {'type': 'exp', 'start': 1e3, 'exp_div_10': 1e4},
    'eval_every_iter': 500,  # interval for evaluation on the validation set
    'log_every_iter': 500,  # interval for logging the loss to the console
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'output_dir': "output", # the 
    'load_weights': False,
    'overfit': False,
    'use_sd_score': False,
    'match_inverted': True,
    'train_fraction': 0.8,
    'normalize_data': True, # normalizing the pointcloud
}