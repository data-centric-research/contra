python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --shuffle
python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --shuffle
python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --shuffle
python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --shuffle
python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --shuffle
python -u main_forget.py --save_dir ./outputs/resnet18_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --shuffle
python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --shuffle --dataset cifar100
python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --shuffle --dataset cifar100
python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --shuffle --dataset cifar100
python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --shuffle --dataset cifar100
python -u main_forget.py --save_dir ./outputs/vgg16_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --shuffle --dataset cifar100