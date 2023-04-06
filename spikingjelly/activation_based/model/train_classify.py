import datetime
import os
import time
import warnings
from .tv_ref_classify import presets, transforms, utils
import torch
import torch.utils.data
import torchvision
from .tv_ref_classify.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
from .. import functional
from .. import monitor, neuron
import pdb
import torch.distributed as dist
import torch.nn.functional as F


try:
    from torchvision import prototype
except ImportError:
    prototype = None

def set_deterministic(_seed_: int = 2020, disable_uda=False):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    
    if disable_uda:
        pass
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # 将调试环境变量CUBLAS_WORKSPACE_CONFIG设置为“：16:8”（可能会限制整体性能）或“：4096:8”（将使GPU内存中的库占用空间增加约24MiB）
        torch.use_deterministic_algorithms(False)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def cal_acc1_acc5(self, output, target):
        # 定义如何计算acc1和acc5
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        return acc1, acc5

    def preprocess_train_sample(self, args, x: torch.Tensor):
        # 定义如何在将训练样本发送到模型之前对其进行处理
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # 定义如何在将测试样本发送到模型之前对其进行处理
        return x

    def process_model_output(self, args, y: torch.Tensor):
        # 定义如何处理 y = model(x)
        return y

    def train_one_epoch(self, model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"

        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, -1, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                image = self.preprocess_train_sample(args, image)
                output = self.process_model_output(args, model(image))      # 脉冲发放频率
                # loss = criterion(output, target)
                targets = torch.argmax(target, dim=1)
                label_one_hot = F.one_hot(targets, 10).float()
                loss = F.mse_loss(output, label_one_hot)  # 输出层神经元的脉冲发放频率与真实类别的MSE



            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # 如果进行梯度剪裁，我们应该取消优化器指定参数的梯度缩放
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            else:                   # 如果在运行时注明了--disable-amp
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            functional.reset_net(model)

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # 重置ema缓冲区以在预热期间保持复制权重
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = self.cal_acc1_acc5(output, target)
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        # 收集所有进程的统计数据
        metric_logger.synchronize_between_processes()
        train_loss, train_acc1, train_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Train: train_acc1={train_acc1:.3f}, train_acc5={train_acc5:.3f}, train_loss={train_loss:.6f}, samples/s={metric_logger.meters["img/s"]}')
        return train_loss, train_acc1, train_acc5

    def evaluate(self, args, model, criterion, data_loader, device, log_suffix=""):
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"

        num_processed_samples = 0
        start_time = time.time()
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)
                output = self.process_model_output(args, model(image))
                # loss = criterion(output, target)
                label_one_hot = F.one_hot(target, 10).float()
                loss = F.mse_loss(output, label_one_hot)  # 输出层神经元的脉冲发放频率与真实类别的MSE


                acc1, acc5 = self.cal_acc1_acc5(output, target)
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = target.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size
                functional.reset_net(model)

        # 收集所有进程的统计数据
        num_processed_samples = utils.reduce_across_processes(num_processed_samples)

        # 加了会报错
        # if (
        #     hasattr(data_loader.dataset, "__len__")
        #     and len(data_loader.dataset) != num_processed_samples
        #     and torch.distributed.get_rank() == 0
        # ):
        #     # See FIXME above
        #     warnings.warn(
        #         f"看起来数据集有 {len(data_loader.dataset)} 样本，但是 {num_processed_samples} "
        #         "样本被用于验证，这可能会使结果产生偏差. "
        #         "尝试调整批量大小和/或世界大小. "
        #         "将世界大小设置为1总是一个安全的赌注."
        #     )

        metric_logger.synchronize_between_processes()

        test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
        return test_loss, test_acc1, test_acc5
    

    # 统计放电率
    def evaluate_firing_rate(self, args, model, data_loader, device, log_suffix=""):
        def cal_firing_rate(s_seq: torch.Tensor):
            # s_seq.shape = [T, N, *]
            return s_seq.flatten(1).mean(1)
        def cal_firing_num(s_seq: torch.Tensor):
            # s_seq.shape = [T, N, *]
            return (int(torch.sum(s_seq)), s_seq.numel())
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"
        mon = monitor.OutputMonitor(model, neuron.LIFNode, cal_firing_num)
        num_processed_samples = 0
        mon.enable()
        i = 0
        firing_rate_list = []
        start_time = time.time()
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, -1, header):
                if i >= 1:
                    break
                i += 1
                image = image.to(device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)
                output = self.process_model_output(args, model(image))
                functional.reset_net(model)
                stack = mon.records
                pdb.set_trace()
                mon.clear_recorded_data()
        # 收集所有进程的统计数据
        return firing_rate_list

    # 加载数据
    def load_data(self, args):
        return self.load_ImageNet(args)

    def load_CIFAR10(self, args):
        # Data loading code
        print("Loading data")
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            transform=presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,

            ),
            download=True
        )

        print("Took", time.time() - st)

        print("Loading validation data")

        dataset_test = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
            download=True
        )

        print("Creating data loaders")
        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)

        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)


        return dataset, dataset_test, train_sampler, test_sampler
    
    def load_CIFAR100(self, args):
        # Data loading code
        print("Loading data")
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=True,
            transform=presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,

            ),
            download = True
        )

        print("Took", time.time() - st)

        print("Loading validation data")

        dataset_test = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=False,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
            download = True
        )

        print("Creating data loaders")
        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)

        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)


        return dataset, dataset_test, train_sampler, test_sampler

    def load_ImageNet(self, args):
        # Data loading code
        traindir = os.path.join(args.data_path, "train")
        valdir = os.path.join(args.data_path, "val")
        print("Loading data")
        val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()
        cache_path = self._get_cache_path(traindir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_train from {cache_path}")
            dataset, _ = torch.load(cache_path)
        else:
            auto_augment_policy = getattr(args, "auto_augment", None)
            random_erase_prob = getattr(args, "random_erase", 0.0)
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                ),
            )
            if args.cache_dataset:
                print(f"Saving dataset_train to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset, traindir), cache_path)
        print("Took", time.time() - st)

        print("Loading validation data")
        cache_path = self._get_cache_path(valdir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_test from {cache_path}")
            dataset_test, _ = torch.load(cache_path)
        else:
            if not args.prototype:
                preprocessing = presets.ClassificationPresetEval(
                    crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
                )
            else:
                if args.weights:
                    weights = prototype.models.get_weight(args.weights)
                    preprocessing = weights.transforms()
                else:
                    preprocessing = prototype.transforms.ImageNetEval(
                        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
                    )

            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
            if args.cache_dataset:
                print(f"Saving dataset_test to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset_test, valdir), cache_path)

        print("Creating data loaders")
        loader_g = torch.Generator()
        loader_g.manual_seed(args.seed)

        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        return dataset, dataset_test, train_sampler, test_sampler

    def load_model(self, args, num_classes):
        raise NotImplementedError("Users should define this function to load model")

    def get_tb_logdir_name(self, args):
        tb_dir = f'{args.model}' \
                 f'_b{args.batch_size}' \
                 f'_e{args.epochs}' \
                 f'_{args.opt}' \
                 f'_lr{args.lr}' \
                 f'_wd{args.weight_decay}' \
                 f'_ls{args.label_smoothing}' \
                 f'_ma{args.mixup_alpha}' \
                 f'_ca{args.cutmix_alpha}' \
                 f'_sbn{1 if args.sync_bn else 0}' \
                 f'_ra{args.ra_reps if args.ra_sampler else 0}' \
                 f'_re{args.random_erase}' \
                 f'_aaug{args.auto_augment}' \
                 f'_size{args.train_crop_size}_{args.val_resize_size}_{args.val_crop_size}' \
                 f'_seed{args.seed}'
        return tb_dir


    def set_optimizer(self, args, parameters):
        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = None
        return optimizer

    def set_lr_scheduler(self, args, optimizer):
        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "step":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosa":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs
            )
        elif args.lr_scheduler == "exp":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            main_lr_scheduler = None
        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                warmup_lr_scheduler = None
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        return lr_scheduler

    def main(self, args):
        set_deterministic(args.seed, args.disable_uda)
        if args.prototype and prototype is None:
            raise ImportError("找不到原型模块。请每晚安装最新的torchvision.")
        if not args.prototype and args.weights:
            raise ValueError("权重参数仅在原型模式下工作。请传递--prototype参数.")
        if args.output_dir:
            utils.mkdir(args.output_dir)

        utils.init_distributed_mode(args)
        print(args)

        device = torch.device(args.device)

        dataset, dataset_test, train_sampler, test_sampler = self.load_data(args)

        collate_fn = None
        num_classes = len(dataset.classes)
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
                pass
            else:
                # TODO implement a CrossEntropyLoss to support for probabilities for each class.
                raise NotImplementedError("pytorch＜1.11.0中的交叉熵损失不支持每个类别的概率."
                                          "设置 mixup_alpha=0. 以避免此类问题或更新您的pytorch.")
            mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=not args.disable_pinmemory,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=not args.disable_pinmemory,
            worker_init_fn=seed_worker
        )

        print("Creating model")
        model = self.load_model(args, num_classes)
        model.to(device)
        print(model)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        if args.norm_weight_decay is None:
            parameters = model.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        optimizer = self.set_optimizer(args, parameters)

        if args.disable_amp:
            scaler = None
        else:
            scaler = torch.cuda.amp.GradScaler()

        lr_scheduler = self.set_lr_scheduler(args, optimizer)


        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        model_ema = None
        if args.model_ema:
            # 衰变调整旨在保持衰变独立于最初提出的其他超参数：
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

        # 确定目录文件名

        tb_dir = self.get_tb_logdir_name(args)
        pt_dir = os.path.join(args.output_dir, 'pt', tb_dir)
        tb_dir = os.path.join(args.output_dir, tb_dir)
        if args.print_logdir:
            print(tb_dir)
            print(pt_dir)
            exit()
        if args.clean:
            if utils.is_main_process():
                if os.path.exists(tb_dir):
                    os.remove(tb_dir)
                if os.path.exists(pt_dir):
                    os.remove(pt_dir)
                print(f'remove {tb_dir} and {pt_dir}.')

        if utils.is_main_process():
            os.makedirs(tb_dir, exist_ok=args.resume is not None)
            os.makedirs(pt_dir, exist_ok=args.resume is not None)

        if args.resume is not None:
            if args.resume == 'latest':
                checkpoint = torch.load(os.path.join(pt_dir, 'checkpoint_latest.pth'), map_location="cpu")
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])

            if utils.is_main_process():
                max_test_acc1 = checkpoint['max_test_acc1']
                if model_ema:
                    max_ema_test_acc1 = checkpoint['max_ema_test_acc1']

        if utils.is_main_process():
            tb_writer = SummaryWriter(tb_dir, purge_step=args.start_epoch)
            with open(os.path.join(tb_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
                args_txt.write(str(args))
                args_txt.write('\n')
                args_txt.write(' '.join(sys.argv))

            max_test_acc1 = -1.
            if model_ema:
                max_ema_test_acc1 = -1.


        if args.test_only:
            if model_ema:
                self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            else:
                self.evaluate(args, model, criterion, data_loader_test, device=device)
            return




        for epoch in range(args.start_epoch, args.epochs):
            start_time = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)

            self.before_train_one_epoch(args, model, epoch)
            train_loss, train_acc1, train_acc5 = self.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
            if utils.is_main_process():
                tb_writer.add_scalar('train_loss', train_loss, epoch)
                tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                tb_writer.add_scalar('train_acc5', train_acc5, epoch)

            lr_scheduler.step()
            self.before_test_one_epoch(args, model, epoch)
            test_loss, test_acc1, test_acc5 = self.evaluate(args, model, criterion, data_loader_test, device=device)
            if utils.is_main_process():
                tb_writer.add_scalar('test_loss', test_loss, epoch)
                tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                tb_writer.add_scalar('test_acc5', test_acc5, epoch)
            if model_ema:
                ema_test_loss, ema_test_acc1, ema_test_acc5 = self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
                if utils.is_main_process():
                    tb_writer.add_scalar('ema_test_loss', ema_test_loss, epoch)
                    tb_writer.add_scalar('ema_test_acc1', ema_test_acc1, epoch)
                    tb_writer.add_scalar('ema_test_acc5', ema_test_acc5, epoch)

            if utils.is_main_process():
                save_max_test_acc1 = False
                save_max_ema_test_acc1 = False

                if test_acc1 > max_test_acc1:
                    max_test_acc1 = test_acc1
                    save_max_test_acc1 = True

                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "max_test_acc1": max_test_acc1,
                }
                if model_ema:
                    if ema_test_acc1 > max_ema_test_acc1:
                        max_ema_test_acc1 = ema_test_acc1
                        save_max_ema_test_acc1 = True
                    checkpoint["model_ema"] = model_ema.state_dict()
                    checkpoint["max_ema_test_acc1"] = max_ema_test_acc1
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()

                utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(pt_dir, "checkpoint_latest.pth"))
                if save_max_test_acc1:
                    utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_test_acc1.pth"))
                if model_ema and save_max_ema_test_acc1:
                    utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_ema_test_acc1.pth"))

                if utils.is_main_process() and epoch > 0:
                    os.remove(os.path.join(pt_dir, f"checkpoint_{epoch - 1}.pth"))
            print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

            
    def main_firing_rate(self, args):
        set_deterministic(args.seed, args.disable_uda)
        if args.prototype and prototype is None:
            raise ImportError("找不到原型模块。请每晚安装最新的torchvision")
        if not args.prototype and args.weights:
            raise ValueError("权重参数仅在原型模式下工作。请传递--prototype参数")
        if args.output_dir:
            utils.mkdir(args.output_dir)

        utils.init_distributed_mode(args)
        print(args)

        device = torch.device(args.device)

        dataset, dataset_test, train_sampler, test_sampler = self.load_data(args)

        collate_fn = None
        num_classes = len(dataset.classes)
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
                pass
            else:
                # TODO implement a CrossEntropyLoss to support for probabilities for each class.
                raise NotImplementedError("CrossEntropyLoss in pytorch < 1.11.0 does not support for probabilities for each class."
                                          "Set mixup_alpha=0. to avoid such a problem or update your pytorch.")
            mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=not args.disable_pinmemory,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            drop_last=True
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=not args.disable_pinmemory,
            worker_init_fn=seed_worker,
            drop_last=True
        )

        # print(f"train_loader中的训练数据数量大致为:{len(data_loader) * args.batch_size}, "           # 数据加载无问题
        #       f"测试数据数量大致为:{len(data_loader_test) * args.batch_size}")


        print("Creating model")
        model = self.load_model(args, num_classes)
        model.to(device)
        print(model)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        if args.norm_weight_decay is None:
            parameters = model.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(model)
            wd_groups = [args.norm_weight_decay, args.weight_decay]
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]


        optimizer = self.set_optimizer(args, parameters)

        if args.disable_amp:
            scaler = None
        else:
            scaler = torch.cuda.amp.GradScaler()

        lr_scheduler = self.set_lr_scheduler(args, optimizer)


        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        model_ema = None
        if args.model_ema:
            # 衰变调整旨在保持衰变独立于最初提出的其他超参数：
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

        # 确定目录文件名
        tb_dir = args.resume_path
        pt_dir = os.path.join(args.output_dir, 'pt', tb_dir)        # logs/pt/resume
        tb_dir = os.path.join(args.output_dir, tb_dir)              # logs/resume
        if args.print_logdir:
            print(tb_dir)
            print(pt_dir)
            exit()
        if args.clean:
            if utils.is_main_process():
                if os.path.exists(tb_dir):
                    os.remove(tb_dir)
                if os.path.exists(pt_dir):
                    os.remove(pt_dir)
                print(f'remove {tb_dir} and {pt_dir}.')

        if utils.is_main_process():
            os.makedirs(tb_dir, exist_ok=True)          # 原本两个都为exist_ok=args.resume is not None
            os.makedirs(pt_dir, exist_ok=True)

        if args.resume is not None:
            if args.resume == 'latest':
                checkpoint = torch.load(os.path.join(pt_dir, 'checkpoint_latest.pth'), map_location="cpu")
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])

            if utils.is_main_process():
                max_test_acc1 = checkpoint['max_test_acc1']
                if model_ema:
                    max_ema_test_acc1 = checkpoint['max_ema_test_acc1']

        if utils.is_main_process():
            tb_writer = SummaryWriter(tb_dir, purge_step=args.start_epoch)
            with open(os.path.join(tb_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
                args_txt.write(str(args))
                args_txt.write('\n')
                args_txt.write(' '.join(sys.argv))

            max_test_acc1 = -1.
            if model_ema:
                max_ema_test_acc1 = -1.

        if args.test_only:
            if model_ema:
                self.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            else:
                self.evaluate(args, model, criterion, data_loader_test, device=device)
            return


        for epoch in range(args.start_epoch, args.epochs):
            start_time = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)

            self.before_train_one_epoch(args, model, epoch)
            train_loss, train_acc1, train_acc5 = self.train_one_epoch(model, criterion, optimizer, data_loader,
                                                                          device, epoch, args, model_ema, scaler)
            if utils.is_main_process():
                tb_writer.add_scalar('train_loss', train_loss, epoch)
                tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                tb_writer.add_scalar('train_acc5', train_acc5, epoch)

            lr_scheduler.step()
            self.before_test_one_epoch(args, model, epoch)
            test_loss, test_acc1, test_acc5 = self.evaluate(args, model, criterion, data_loader_test, device=device)
            if utils.is_main_process():
                tb_writer.add_scalar('test_loss', test_loss, epoch)
                tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                tb_writer.add_scalar('test_acc5', test_acc5, epoch)
            if model_ema:
                ema_test_loss, ema_test_acc1, ema_test_acc5 = self.evaluate(args, model_ema, criterion,
                                                                                data_loader_test, device=device,
                                                                                log_suffix="EMA")
                if utils.is_main_process():
                    tb_writer.add_scalar('ema_test_loss', ema_test_loss, epoch)
                    tb_writer.add_scalar('ema_test_acc1', ema_test_acc1, epoch)
                    tb_writer.add_scalar('ema_test_acc5', ema_test_acc5, epoch)

            if utils.is_main_process():
                save_max_test_acc1 = False
                save_max_ema_test_acc1 = False

                if test_acc1 > max_test_acc1:
                    max_test_acc1 = test_acc1
                    save_max_test_acc1 = True

                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "max_test_acc1": max_test_acc1,
                }
                if model_ema:
                    if ema_test_acc1 > max_ema_test_acc1:
                        max_ema_test_acc1 = ema_test_acc1
                        save_max_ema_test_acc1 = True
                    checkpoint["model_ema"] = model_ema.state_dict()
                    checkpoint["max_ema_test_acc1"] = max_ema_test_acc1
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()

                utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(pt_dir, "checkpoint_latest.pth"))
                if save_max_test_acc1:
                    utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_test_acc1.pth"))
                if model_ema and save_max_ema_test_acc1:
                    utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_ema_test_acc1.pth"))

                if utils.is_main_process() and epoch > 0:
                    os.remove(os.path.join(pt_dir, f"checkpoint_{epoch - 1}.pth"))
            print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

            # firing_rate = self.evaluate_firing_rate(args, model, data_loader_test, device=device)
            # return firing_rate

            

    def before_test_one_epoch(self, args, model, epoch):
        pass

    def before_train_one_epoch(self, args, model, epoch):
        pass

    def get_args_parser(self, add_help=True):

        parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
        parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
        parser.add_argument("--model", default="resnet18", type=str, help="model name")
        parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
        parser.add_argument("-b", "--batch-size", default=32, type=int, help="每个gpu的图像，总批量大小为$NGPU x batch_size")
        parser.add_argument("--epochs", default=90, type=int, metavar="N", help="epochs")
        parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="num_workers(默认为16)")
        parser.add_argument("--opt", default="sgd", type=str, help="优化器")
        parser.add_argument("--lr", default=0.1, type=float, help="学习率")
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="动量")
        parser.add_argument("--wd", "--weight-decay", default=0., type=float, metavar="W", help="weight decay (default: 0.)", dest="weight_decay")
        parser.add_argument("--norm-weight-decay", default=None, type=float, help="规格化层的权重衰减 (default: None, same value as --wd)")
        parser.add_argument("--label-smoothing", default=0.1, type=float, help="标签平滑 (default: 0.1)", dest="label_smoothing")
        parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)")
        parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 1.0)")
        parser.add_argument("--lr-scheduler", default="cosa", type=str, help="学习率衰减算法 (default: cosa)")
        parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="进行学习率预热的epoch数 (default: 5)")
        parser.add_argument("--lr-warmup-method", default="linear", type=str, help="学习率预热的预热方法 (default: linear)")
        parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="学习率预热中学习率的衰减")
        parser.add_argument("--lr-step-size", default=30, type=int, help="多少个epoch衰减一次lr")
        parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
        parser.add_argument("--output-dir", default="./logs", type=str, help="存储结果的位置")
        parser.add_argument("--resume", default=None, type=str, help="检查点的路径。如果设置为“latest”，它将尝试加载最新的检查点")
        parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="开始的epoch")
        parser.add_argument("--cache-dataset", dest="cache_dataset", help="缓存数据集以加快初始化。它还序列化转换", action="store_true")
        parser.add_argument("--auto-augment", default='ta_wide', type=str, help="自动增强策略（默认值：ta_wide）")
        parser.add_argument("--random-erase", default=0.1, type=float, help="随机擦除概率（默认值：0.1）")

        parser.add_argument("--sync-bn", dest="sync_bn", help="使用同步BN", action="store_true")
        parser.add_argument("--test-only", dest="test_only", help="仅测试模型", action="store_true")
        parser.add_argument("--pretrained", dest="pretrained", help="使用modelzoo中经过预训练的模型", action="store_true")


        # 混合精度训练参数
        # 分布式训练参数
        parser.add_argument("--world-size", default=1, type=int, help="分布式进程数")
        parser.add_argument("--dist-url", default="env://", type=str, help="用于设置分布式训练的url")

        parser.add_argument("--model-ema", action="store_true", help="启用跟踪模型参数的指数移动平均值")

        parser.add_argument("--model-ema-steps", type=int, default=32, help="控制EMA模型更新频率的迭代次数（默认值：32）")
        parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="模型参数的指数移动平均值的衰减因子（默认值：0.99998）")
        parser.add_argument("--interpolation", default="bilinear", type=str, help="插值方法(default: bilinear)")
        parser.add_argument("--val-resize-size", default=232, type=int, help="用于验证的调整大小 (default: 232)")
        parser.add_argument("--val-crop-size", default=224, type=int, help="用于验证的中心作物尺寸 (default: 224)")
        parser.add_argument("--train-crop-size", default=176, type=int, help="用于训练的随机作物大小 (default: 176)")
        parser.add_argument("--clip-grad-norm", default=None, type=float, help="最大梯度范数 (default None)")

        parser.add_argument("--ra-sampler", action="store_true", help="是否在训练中使用重复增强")
        parser.add_argument("--ra-reps", default=4, type=int, help="重复增强的重复次数 (default: 4)")

        # 仅原型模型
        parser.add_argument("--prototype", dest="prototype", help="使用原型模型构建器，而不是来自主要区域的模型构建器", action="store_true")

        parser.add_argument("--weights", default=None, type=str, help="要加载的权重枚举名称")
        parser.add_argument("--seed", default=2023, type=int, help="随机种子")

        parser.add_argument("--print-logdir", action="store_true", help="打印tensorboard日志和pt文件的目录并退出")
        parser.add_argument("--clean", action="store_true", help="删除tensorboard日志和pt文件的目录")
        parser.add_argument("--disable-pinmemory", action="store_true", help="不在数据加载器中使用引脚内存，这有助于减少内存消耗")
        parser.add_argument("--disable-amp", action="store_true", help="不使用自动混合精度训练")
        parser.add_argument("--local_rank", type=int, help="DDP的参数，不应由用户设置")
        parser.add_argument("--disable-uda", action="store_true", help="不要（没有）设置“torch.use_determistic_algorithms（True）”，这可以避免某些没有确定性实现的函数引发的错误")
        parser.add_argument("--resume-path", type=str, default='./resume', help="不要（没有）设置“torch.use_determistic_algorithms（True）”，这可以避免某些没有确定性实现的函数引发的错误")

        return parser

if __name__ == "__main__":
    trainer = Trainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)