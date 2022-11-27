# SNN_Attention_VGG

### Run CIFAR10 with VGG7

```python
python3 CIFAR10_VGG.py --T 1 --model spiking_vgg7 --data-path *Your CIFAR10 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```



### Run CIFAR10 with VGG7

```python
python3 CIFAR10_VGG_Att.py --T 1 --model spiking_vgg7 --data-path *Your CIFAR10 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```



### Run CIFAR10 with VGG11

```python
python3 CIFAR10_VGG.py --T 1 --model spiking_vgg11_bn --data-path *Your CIFAR10 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```





### Run CIFAR10 with VGG11 Attention

```python
python3 CIFAR10_VGG_Att.py --T 1 --model spiking_vgg11_bn --data-path *Your CIFAR10 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```



### Run CIFAR100 with VGG11

```python
python3 CIFAR100_VGG.py --T 1 --model spiking_vgg11_bn --data-path *Your CIFAR100 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```



### Run CIFAR100 with VGG11 Attention

```python
python3 CIFAR100_VGG_Att.py --T 1 --model spiking_vgg11_bn --data-path *Your CIFAR100 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```



### Run CIFAR100 with VGG13

```python
python3 CIFAR100_VGG.py --T 1 --model spiking_vgg13_bn --data-path *Your CIFAR100 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```



### Run CIFAR100 with VGG13 Attention

```python
python3 CIFAR100_VGG_Att.py --T 1 --model spiking_vgg13_bn --data-path *Your CIFAR100 Data Path* --batch-size 64 --lr 0.001 --lr-scheduler cosa --epochs 200 --lr-warmup-epochs 0 --opt adamw --train-crop-size 32 --val-crop-size 32 --val-resize-size 32
```

