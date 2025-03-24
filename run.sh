#CIFAR-10
python main.py --niid --partition dir --rg 1 --num_client 100 --alpha 0.1 --clean_reg --dataset cifar10 --num_class 10 --modelseed 1 --pretrained

#CIFAR-100
#python main.py --niid --partition dir --rg 1 --num_client 100 --alpha 0.1 --clean_reg --dataset cifar100 --num_class 100 --modelseed 1 --pretrained

#Tiny-ImageNet
#python main.py --niid --partition dir --rg 1 --num_client 100 --alpha 0.1 --clean_reg --dataset tinyimagenet --num_class 200 --modelseed 1 --pretrained
