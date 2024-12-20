config={
    'algorithm':'PENCIL',
    # dataset param
    'dataset':'cifar-10',
    'input_channel':3,
    'num_classes':10,
    'root':'/data/yfli/CIFAR10',
    'noise_type':'sym',
    'percent':0.2,
    'seed':1,
    'loss_type':'ce',
    # model param
    'model1_type':'resnet18',
    'model2_type':'none',
    'stage1':70, #70
    'stage2':200, #200
    'epochs':200,
    'alpha':0.4,
    'beta':0.1,
    'lamd':600,
    # train param
    'gpu':'0',
    'batch_size':128,
    'lr':0.06,
    'lr2':0.2,
    'epochs':200,
    'num_workers':4,
    'adjust_lr':1,
    'epoch_decay_start':80,
    # result param
    'save_result':True,
    'output_idx':True
}