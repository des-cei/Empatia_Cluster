import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torch.quantization
from efficientnet import *
from torch.quantization import QuantStub, DeQuantStub
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    """ Computes the top 1 accuracy """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def load_model(quantized_model, model_path):
    """ Loads in the weights into an object meant for quantization """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k.replace("module.", '')
        new_state_dict[name] = v

    quantized_model.load_state_dict(new_state_dict)


def fuse_modules(model):
    """ Fuse together convolutions/linear layers and ReLU """
    torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                            ['conv2', 'relu2'],
                                            ['fc1', 'relu3'],
                                            ['fc2', 'relu4']], inplace=True)


if __name__ == '__main__':
    classes = ('not_fear', 'fear')
    qnet = EfficientNetB0(num_classes=len(classes), q=True)
    load_model(qnet, 'checkpoint/ckpt.pth')
    # fuse_modules(qnet)

    print_size_of_model(qnet)
    qnet.qconfig = torch.quantization.default_qconfig
    print(qnet.qconfig)
    torch.quantization.prepare(qnet, inplace=True)
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Conv1: After observer insertion \n\n', qnet.conv1)

    print('Post Training Quantization: Calibration done')
    torch.quantization.convert(qnet, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Conv1: After fusion and quantization \n\n', qnet.conv1)
    print("Size of model after quantization")
    print_size_of_model(qnet)

    print('Saving..')
    state = {
        'net': qnet.state_dict(),
        'acc': qnet,
        'epoch': qnet,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/quantization.pth')

