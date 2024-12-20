config = {
    "algorithm":'ELR',
    # dataset param
    "dataset":'cifar-10',
    "input_channel":3,
    "num_classes":10,
    "root":'/data/yfli/CIFAR10',
    "noise_type":'sym',
    "percent":0.2,
    "seed":1,
    # model param
    "model1_type":'resnet18',
    "model2_type":'none',
    # train param
    "gpu":'0',
    "batch_size":128,
    "lr":0.1,
    "epochs":150,
    "num_workers":4,
    "adjust_lr":1,
    "epoch_decay_start":80,
    # optmizer params
    "save_result":True,
    "weight_decay":1e-3,
    "nesterov":True,
    "momentum":0.9,
    "loss_beta":0.7,
    "loss_lamd":3,
}