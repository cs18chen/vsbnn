import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
#from outlierexposure.CIFAR.models.allconv import AllConvNet
#from outlierexposure.CIFAR.models.wrn import WideResNet
from models.bnnmodels import HorseshoeNet, BNN, HPBayesianNet, HPBayesianNet1,RegHPBayesianNet1, Dropout_MLP, MLP
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import torchvision
from sklearn.metrics import roc_curve, auc
# go through rigamaroo to do ...util.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from util.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import util.svhn_loader as svhn
    import util.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=100)
parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', default=True, help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', default='entropy',type=str, help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', type=str, default='cifar10_Gauss', help='Method name: cifar10_BNN, cifar10_RegHP1,cifar10_HS')
# Loading details
#parser.add_argument('--layers', default=40, type=int, help='total number of layers')
#parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
#parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', type=str, default='./checkpoints_800', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
args = parser.parse_args()

n_test_samples = 10
test_batch_size = 100
num_classes = 10
torch.manual_seed(1)
np.random.seed(1)



# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10('./cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 10
else:
    test_data = dset.CIFAR100('./cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 100


test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if args.method_name == 'cifar10_Gauss':
    net = BNN(3, num_classes)
elif args.method_name == 'cifar10_RegHP1':
    net = RegHPBayesianNet1(3, num_classes)
elif args.method_name == 'cifar10_HS':
    net = HorseshoeNet(3, num_classes)
elif args.method_name == 'cifar10_MLP':
    net = MLP(3, num_classes)
else:
    raise NotImplementedError('Invalid model')


start_epoch = 1

# Restore model
'''
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'baseline' in args.method_name:
            subdir = 'baseline'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        else:
            subdir = 'oe_scratch'

        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"
'''
model_name = os.path.join(args.load, args.method_name + '.pt')
net.load_state_dict(torch.load(model_name))
print('Model restored! Epoch:')

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

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

            data = data.view(-1, 3, 32, 32).cuda()


 

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
            outputs = net.sample_predict(data, n_test_samples)
            output = outputs.mean(dim=0)  # take mean across test samples
            output = torch.exp(output)  #entropy

            #output = net(data) #MLP, MSP
            smax = to_np(output)
 


            if args.use_xent == 'entropy':
                _score.append(to_np(torch.distributions.Categorical(probs=output).entropy()))
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
                    _right_score.append(to_np(torch.distributions.Categorical(probs=output).entropy())[right_indices])
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
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)

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

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_8/tpr_{}_{}_gauss.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_8/fpr_{}_{}_gauss.out'.format(args.method_name, args.use_xent), fpr)
# /////////////// SVHN ///////////////

ood_data = svhn.SVHN(root='./svhn/', split="test",
                     transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]), download=True)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_8/tpr_{}_{}_svhn.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_8/fpr_{}_{}_svhn.out'.format(args.method_name, args.use_xent), fpr)



# /////////////// CIFAR Data ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('./cifarpy', train=False, download=True, transform=test_transform)
else:
    ood_data = dset.CIFAR10('./cifarpy', train=False,  download=True, transform=test_transform)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)


print('\n\nCIFAR-100 Detection') if 'cifar100' in args.method_name else print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_8/tpr_{}_{}_cifar10.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_8/fpr_{}_{}_cifar10.out'.format(args.method_name, args.use_xent), fpr)


