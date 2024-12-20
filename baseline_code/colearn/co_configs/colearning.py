config = {
    'algorithm': 'Colearning',
    # dataset param
    'dataset': 'cifar-10',
    'input_channel': 3,
    'num_classes': 10,
    'root': './data',
    'noise_type': 'sym',
    'percent': 0.2,
    'seed': 1,
    # model param
    'hidden_ratio': 0.125,
    'feature_dim': 512,
    # train param
    'batch_size': 128,
    'num_workers': 1,
    'lr': 0.001,
    'warmup_lr': 0.001,
    'epochs': 200,
    'adjust_lr': 1,
    'epoch_decay_start': 80,
    # For two-model algorithms
    'model1_type': 'resnet18',
    'model2_type': 'resnet18',
    'save_result': True}