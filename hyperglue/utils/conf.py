model_conf = {
    'descriptor_dim': 336,
    'weights': 'weights_01',
    'keypoint_encoder': [32,64,128,256],
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 200,
    'match_threshold': 0.2,
    #'bottleneck_dim': None,
    'loss': {
        'nll_weight': 1.,
        'nll_balancing': 0.5,
        #'reward_weight': 0.,
        #'bottleneck_l2_weight': 0.,
    },
}

train_conf = {
    'seed': 42,  # training seed
    'epochs': 25,  # number of epochs
    'batch_size': 1, # yes
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr': 0.01,  # learning rate
    'lr_schedule': {'type': None, 'start': 0, 'exp_div_10': 1},
    'eval_every_iter': 100,  # interval for evaluation on the validation set
    'log_every_iter': 200,  # interval for logging the loss to the console
    'keep_last_checkpoints': 5,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'median_metrics': [],  # add the median of some metrics
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'dataset_callback_fn': None,  # data func called at the start of each epoch
    'output_dir': "output",
    'load_weights':False
}