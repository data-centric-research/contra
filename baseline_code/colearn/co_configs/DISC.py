config = {
    "algorithm" :  'DISC',
    # dataset param
    "dataset" :  'cifar-10',
    "input_channel" :  3,
    "num_classes" :  10,
    "root" :  '/data/yfli/CIFAR10',
    "noise_type" :  'sym',
    "percent" :  0.8,
    "seed" :  1,
    "loss_type" :  'ce',
    # model param
    "model1_type" :  'resnet18',
    "model2_type" :  'none',
    # train param
    "gpu" :  '0',
    "batch_size" :  128,
    "lr" :  0.01,
    "momentum" : 0.9,
    # lr" :  0.0004
    "epochs" :  200,
    "num_workers" :  4,
    "adjust_lr" :  1,
    "epoch_decay_start" :  120,
    "alpha" :  5.0,
    # result param
    "save_result" :  True
}