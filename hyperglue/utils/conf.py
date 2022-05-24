folder_conf = {
    'evaluation_data_path': 'data_evaluation'
}

data_conf = {
    'desc': 'pillar', # [fpfh, pillar, fpfh_pillar]
    'kpts': 'hybrid', # [hybrid, sd]
    'gt_match_thresh': 0.0,
}

model_conf = {
    'pillar': True,
    'descriptor_dim': 36, # the descriptor dimension, must be dividable by num heads!
    'num_heads': 6, # num of heads
    'sep_encoder': True, #sepparate encoders
    'keypoint_encoder': [8, 16, 32, 64], # intermediate mlp dimensions. The first is automatically set to 4, last to descriptor_dim
    'GNN_layers': 3,
    'sinkhorn_iterations': 1000,
    'match_threshold': 0.5,
    'nll_weight': 1000.,
    'nll_balancing': 0.96,
}

train_conf = {
    'seed': 42,  # training seed
    'epochs': 50,  # number of epochs
    'batch_size': 1,  # training batch size
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'lr':0.0085,  # learning rate
    'lr_schedule': {'type': 'exp', 'start': 10e3, 'exp_div_10': 2e5},
    'overfit': False,
    'use_sd_score': False,
    'match_inverted': False,
    'train_fraction': 0.9,
    'normalize_data': False, # normalizing the pointcloud
}