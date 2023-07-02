# Neural-Network-Final-hw

1.simclr
运行命令为：
python self-sv.py -out 128 -batch_size 128 -lr 1e-3 -t 0.5 -epoch 100 

python linear.py -batch_size 256 -pretrained ./self-supervised_model_pt/self-supervised99.pt -lr 0.05 -epoch 100

python sv.py -batch_size 128 -lr 0.01 -epoch 100

2.VIT-Resnet18
运行命令为：
（i）	python train.py --method cutmix --dataset cifar100 --model vit --epochs 200 --batch_size 128 --learning_rate 0.1
（ii）	python train.py --method cutout --dataset cifar100 --model vit --epochs 200 --batch_size 128 --learning_rate 0.1
（iii）	python train.py --method mixup --dataset cifar100 --model vit --epochs 200 --batch_size 128 --learning_rate 0.1

3.NeRF
运行命令为：
python run_nerf.py --config paper_configs/hotdog.txt
