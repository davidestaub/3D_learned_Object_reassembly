metric = {
    "name": 'total',
    "goal": 'minimize'
}

sweep_config = {'method': 'bayes'}

terminator = {
    "type": 'hyperband',
    "min_iter": 100
}

param_dict = {
    'learning_rate': {
        'distribution': 'uniform',
        'min': 1e-3,
        'max': 1e-4
    },
    'optimizer': {
        'values': ["adam", "rmsprop"]
    },
    'sinkhorn_iterations': {
        'values':  [100,200,300]
    },
    'num_heads': {
        'values': [4, 6]},
    'sep_encoder': {
        'values': [True, False]},
    'use_sd_score': {
        'values': [False]},
    'match_threshold': {
        'distribution': 'uniform',
        'min': 0.18,
        'max': 0.28
    },
    'pillar': {
        'values': [True]},
    'match_inverted': {
        'values': [False]},
    'nll_balancing': {
        'distribution': 'uniform',
        'min': 0.9,
        'max': 0.95
    },

    'GNN_layers': {
        'values': [3, 4, 5, 6, 7]},
    'keypoint_encoder': {
        'values': [
            [8, 16, 32, 64],
            [8, 32, 64, 128],
        ]}
}

sweep_config['metric'] = metric
sweep_config['parameters'] = param_dict
sweep_config['early_terminate'] = terminator
