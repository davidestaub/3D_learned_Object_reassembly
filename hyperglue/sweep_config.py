metric = {
    "name": 'total',
    "goal": 'minimize'
}
sweep_config = {'method': 'bayes'}

terminator = {
  "type": 'hyperband',
  "min_iter": 3
}

param_dict = {
    'learning_rate': {
        'distribution': 'uniform',
        'min': 1e-5,
        'max': 1e-2
    },
    'optimizer': {
        'values': ["adam", "sgd"]
    },
    'sinkhorn_iterations': {
        'distribution': 'int_uniform',
        'min': 50,
        'max': 500
    },
    'num_heads': {
        'values': [4, 6]},
    'sep_encoder': {
        'values': [True, False]},
    'use_sd_score': {
        'values': [True, False]},
    'match_threshold': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.5
    },
    'pillar': {
        'values': [True, False]},
    'nll_balancing': {
        'distribution': 'uniform',
        'min': 0.9,
        'max': 0.99
    },

    'GNN_layers': {
        'values': [3, 4, 5, 6, 7]},
    'keypoint_encoder': {
        'values': [
            [8, 16, 32, 64],
            [32, 64, 128, 256],
        ]}
}

sweep_config['metric'] = metric
sweep_config['parameters'] = param_dict
sweep_config['early_terminate'] = terminator