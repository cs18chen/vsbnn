import numpy as np
import sys
import os
import pickle
import argparse
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.bnnmodels import HorseshoeNet, BNN, HPBayesianNet, HPBayesianNet1, RegHPBayesianNet1, Dropout_MLP, MLP
#from models.allconv import AllConvNet
#from models.wrn import WideResNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import torchvision
from torchvision import transforms

from sklearn.metrics import roc_curve, auc

# go through rigamaroo to do ...util.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from util.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import util.svhn_loader as svhn
    #import util.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates a SVHN OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=100)
parser.add_argument('--num_to_avg', type=int, default=5, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate',  default=True, help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', default='entropy', help='Use entropy or MSP.')
parser.add_argument('--method_name',  type=str, default='RegHP1', help='BNN, RegHP1, HS, MLP')

parser.add_argument('--load', '-l', type=str, default='./checkpoints_800', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
args = parser.parse_args()
n_test_samples = 10
test_batch_size = 100
num_classes = 10
torch.manual_seed(1)
np.random.seed(1)

test_data = dset.SVHN('./svhn/', split='test',
                      transform=trn.ToTensor(), download=False)
num_classes = 10

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model


if args.method_name == 'RegHP1':
    net = RegHPBayesianNet1(3, 10)
elif args.method_name == 'HS':
    net = HorseshoeNet(3, 10)
elif args.method_name == 'Gauss':
    net = BNN(3, 10)
elif args.method_name == 'MLP':
    net = MLP(3, 10)
else:
    raise NotImplementedError('Invalid model')
start_epoch = 0

# Restore model


model_name = os.path.join(args.load, args.method_name + '.pt')
net.load_state_dict(torch.load(model_name))

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = test_data.data.shape[0] // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.data.shape[0])

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()
          #  print(data.shape)

           # output = net(data)
            #smax = to_np(F.softmax(output, dim=1))


            '''
            outputs = torch.zeros(n_test_samples + 1, test_batch_size,
                                  num_classes).cuda()
            for i in range(n_test_samples):
                outputs[i] = net(data, take_sample=True)
            outputs[n_test_samples] = net(data, take_sample=False)  # compute output with mean weights too
            output = outputs.mean(dim=0)  # take mean across test samples
            #preds = outputs.argmax(dim=2, keepdim=True)
            #mean_pred = mean_output.argmax(dim=1, keepdim=True)
            '''
            outputs = net.sample_predict(data, 10)
            output = outputs.mean(dim=0)  # take mean across test samples
            output = torch.exp(output)  #entropy

            #output = net(data) #MLP, MSP
            smax = to_np(output)
           # smax=output

            if args.use_xent == 'entropy':
                _score.append(to_np(torch.distributions.Categorical(probs=output).entropy())) #or probs=output+1e-5
            elif args.use_xent == 'MSP':
                _score.append(-np.max(smax, axis=1))
            else:
                raise NotImplementedError('Invalid model')



            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent == 'entropy':
                    _right_score.append(to_np(torch.distributions.Categorical(probs=output).entropy())[right_indices]) #or probs=output+1e-5
                    _wrong_score.append(to_np(torch.distributions.Categorical(probs=output).entropy())[wrong_indices])
                elif args.use_xent == 'MSP':
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
                else:
                    raise NotImplementedError('Invalid model')
    
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100*num_wrong/(num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing SVHN as typical data')

# /////////////// Error Detection ///////////////

#print('\n\nError Detection')
#show_performance(wrong_score, right_score, method_name=args.method_name)

# /////////////// OOD Detection ///////////////

auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)


# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples*args.num_to_avg)
ood_data = torch.from_numpy(
    np.clip(np.random.normal(size=(ood_num_examples*args.num_to_avg, 3, 32, 32),
                             loc=0.5, scale=0.5).astype(np.float32), 0, 1))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nGaussian Noise (mu = sigma = 0.5) Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_gauss.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_gauss.out'.format(args.method_name, args.use_xent), fpr)




# /////////////// CIFAR data ///////////////

ood_data = dset.CIFAR10('./cifarpy', train=False, transform=trn.ToTensor(), download=True)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_cifar10.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_cifar10.out'.format(args.method_name, args.use_xent), fpr)

# /////////////// CIFAR data ///////////////

ood_data = dset.CIFAR100('./cifarpy', train=False, transform=trn.ToTensor(), download=True)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nCIFAR-100 Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_cifar100.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_cifar100.out'.format(args.method_name, args.use_xent), fpr)


