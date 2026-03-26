config = {
    'algorithm': 'DivideMix',
    # dataset param
    'dataset': 'cifar-10',
    'input_channel': 3,
    'num_classes': 10,
    'root': './data',
    'noise_type': 'sym',
    'percent': 0.2,
    'seed': 4398,
    # model param
    'model1_type': 'resnet18',
    'model2_type': 'resnet18',
    # train param
    'batch_size': 128,
    'lr': 0.02,
    'epochs': 15,
    'num_workers': 4,
    'adjust_lr': 0,
    # DivideMix-specific
    'warmup_epochs': 3,
    'lambda_u': 25.0,
    'T': 0.5,
    'alpha': 4.0,
    # result param
    'save_result': True}
