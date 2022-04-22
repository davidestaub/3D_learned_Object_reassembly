model_conf = {
    'normalize_data': True, # normalizing the pointcloud
    'use_pointnet': False, # if true, using the pointnet as encoder and ignores the mlp one
    'use_mlp': True, # using the mlp encoder, only works if pointnet is false
    'use_desc': True,
    'descriptor_dim': 336, # the descriptor dimension, this is what SHOT gives us
    'weights': 'weights_01',
    'keypoint_encoder': [64, 128, 512, 1024], # intermediate mlp dimensions. The first is automatically set to 3, last to 336
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 200,
    'match_threshold': 0.2,
    # 'bottleneck_dim': None,
    'loss': {
        'nll_weight': 1.,
        'nll_balancing': 0.9,
        #'reward_weight': 0.,
        #'bottleneck_l2_weight': 0.,
    },
}
   

train_conf = {
    'seed': 42,  # training seed
    'epochs': 100000,  # number of epochs
    'batch_size_train': 24,  # training batch size
    'batch_size_test': 24, #test batch size
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr': 0.001,  # learning rate
    'lr_schedule': {'type': 'exp', 'start': 1e3, 'exp_div_10': 50e3},
    'eval_every_iter': 500,  # interval for evaluation on the validation set
    'log_every_iter': 100,  # interval for logging the loss to the console
    'keep_last_checkpoints': 5,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'median_metrics': ['match_recall'],  # add the median of some metrics
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'dataset_callback_fn': None,  # data func called at the start of each epoch
    'output_dir': "output",
    'load_weights': False
}