nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path ../data/cifar100/resnet18/retrain --shuffle > resnet18-cifar100-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path ../data/cifar100/resnet18/FT --shuffle > resnet18-cifar100-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path ../data/cifar100/resnet18/GA --shuffle > resnet18-cifar100-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path ../data/cifar100/resnet18/FF --shuffle > resnet18-cifar100-FF.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path ../data/cifar100/resnet18/IU --shuffle > resnet18-cifar100-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path ../data/cifar100/resnet18/FT_prune --shuffle > resnet18-cifar100-FT_prune.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar100/vgg16/retrain --shuffle --dataset cifar100 > vgg16-cifar100-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar100/vgg16/FT --shuffle --dataset cifar100 > vgg16-cifar100-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar100/vgg16/GA --shuffle --dataset cifar100 > vgg16-cifar100-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar100/vgg16/IU --shuffle --dataset cifar100 > vgg16-cifar100-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar100/vgg16/FT_prune --shuffle --dataset cifar100 > vgg16-cifar100-FT_prune.log 2>&1 &