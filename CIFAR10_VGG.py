import torch
import sys
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet, train_classify, spiking_vgg, parametric_lif_net
from spikingjelly.activation_based.monitor import GPUMonitor
from datetime import datetime


class SResNetTrainer(train_classify.Trainer):
    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--T', type=int, help="total time-steps")
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        return parser

    def get_tb_logdir_name(self, args):
        now_time = datetime.now()
        return super().get_tb_logdir_name(args) + f'_T{args.T}' + '_' + str(now_time.day) + '_' + str(now_time.hour) + '_' + str(now_time.minute)
    
    def load_data(self, args):
        return self.load_CIFAR10(args)

    def load_model(self, args, num_classes):
        model = spiking_vgg.__dict__[args.model](pretrained=args.pretrained, spiking_neuron=neuron.LIFNode,
                                                        surrogate_function=surrogate.ATan(), detach_reset=True, num_classes=num_classes)
        functional.set_step_mode(model, step_mode='m')
        if args.cupy:
            functional.set_backend(model, 'cupy', neuron.LIFNode)
            return model
        else:
            raise ValueError(f"args.model should be one of {spiking_vgg.__all__}")


if __name__ == "__main__":
    trainer = SResNetTrainer()
    args = trainer.get_args_parser().parse_args()
    #gm = GPUMonitor(interval=100)
    print(trainer.main_firing_rate(args))
    gm.stop()
    #print("hello")

