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
from skimage.transform import resize
#pillow
from models.bnnmodels import HorseshoeNet, BNN, RegHPBayesianNet1,Dropout_MLP, MLP
from skimage import filters
#from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from sklearn.metrics import roc_curve, auc
# go through rigamaroo to do ...util.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from util.display_results import show_performance, get_measures, print_measures, print_measures_with_std
#    import util.svhn_loader as svhn
  #  import util.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates an MNIST OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=100)
parser.add_argument('--num_to_avg', type=int, default=2, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', default=True, help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', default='MSP',type=str, help='Use entropy or MSP.')
parser.add_argument('--method_name',  type=str, default='HS', help='Method name: BNN, RegHP1, HS, MCD,MLP')
# Loading details
parser.add_argument('--load', type=str, default='./checkpoint_800', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
args = parser.parse_args()

n_test_samples = 10
#test_batch_size = 100
num_classes = 10
torch.manual_seed(1)
np.random.seed(1)


test_data = dset.MNIST('./mnist', train=False, transform=trn.ToTensor(), download=True)
num_classes = 10

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model

if args.method_name == 'MCD':
    net = Dropout_MLP(1, 10)
#elif args.method_name == 'HP':
 #   net = HPBayesianNet(1, 10)
#elif args.method_name == 'HP1':
#    net = HPBayesianNet1(1, 10)
elif args.method_name == 'RegHP1':
    net = RegHPBayesianNet1(1, 10)
elif args.method_name == 'HS':
    net = HorseshoeNet(1, 10)
elif args.method_name == 'Gauss':
    net = BNN(1, 10)
elif args.method_name == 'MLP':
    net = MLP(1, 10)
else:
    raise NotImplementedError('Invalid model')

start_epoch = 1

# Restore model
'''
if args.load != '':
    for i in range(300 - 1, -1, -1):
        if 'Hs' in args.method_name:
            subdir = 'HS'
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

ood_num_examples = test_data.data.size(0) // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.data.size(0))

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

            data = data.view(-1, 28, 28).cuda()





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
                _score.append(to_np(torch.distributions.Categorical(probs=output+1e-5).entropy()))
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
                    _right_score.append(to_np(torch.distributions.Categorical(probs=output+1e-5).entropy())[right_indices])
                    _wrong_score.append(to_np(torch.distributions.Categorical(probs=output+1e-5).entropy())[wrong_indices])
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

print('\nUsing MNIST as typical data')

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

dummy_targets = torch.ones(ood_num_examples*args.num_to_avg)
ood_data = torch.from_numpy(
    np.clip(np.random.normal(size=(ood_num_examples*args.num_to_avg, 1, 28, 28),
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
np.savetxt('./roc_800/tpr_{}_{}_gauss.out'.format(args.method_name,args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_gauss.out'.format(args.method_name,args.use_xent), fpr)


# /////////////// Unifrom Noise ///////////////

dummy_targets = torch.ones(ood_num_examples*args.num_to_avg)
ood_data = torch.from_numpy(np.random.uniform( size=(ood_num_examples*args.num_to_avg, 1, 28, 28), low=0, high=1).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform Noise Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_uniform.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_uniform.out'.format(args.method_name, args.use_xent), fpr)

# /////////////// CIFAR data ///////////////

ood_data = dset.CIFAR10(
    './cifarpy', train=False,
    transform=trn.Compose([trn.Resize(28),
                           trn.Lambda(lambda x: x.convert('L', (0.2989, 0.5870, 0.1140, 0))),
                           trn.ToTensor()]), download=True)
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



# /////////////// Fashion-MNIST ///////////////

ood_data = dset.FashionMNIST('./fashion_mnist', train=False,
                             transform=trn.ToTensor(), download=True)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nFashion-MNIST Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_fashion.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_fashion.out'.format(args.method_name, args.use_xent), fpr)




# /////////////// Omniglot ///////////////

import scipy.io as sio
import scipy.misc as scimisc
# other alphabets have characters which look like digits
safe_list = [0, 2, 5, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 26]
m = sio.loadmat("./datasets/data_background.mat")

squished_set = []
for safe_number in safe_list:
    for alphabet in m['images'][safe_number]:
        for letters in alphabet:
            for letter in letters:
                for example in letter:
                    #squished_set.append(scimisc.imresize(1 - example[0], (28, 28)).reshape(1, 28 * 28))
                    squished_set.append(resize(1 - example[0], (28, 28)).reshape(1, 28 * 28))



omni_images = np.concatenate(squished_set, axis=0)

dummy_targets = torch.ones(min(ood_num_examples*args.num_to_avg, len(omni_images)))
ood_data = torch.utils.data.TensorDataset(torch.from_numpy(
    omni_images[:ood_num_examples*args.num_to_avg].astype(np.float32)), dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nOmniglot Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_omniglot.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_omniglot.out'.format(args.method_name, args.use_xent), fpr)


# /////////////// notMNIST ///////////////

pickle_file = './datasets/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    notMNIST_data = pickle.load(f, encoding='latin1')
    notMNIST_data = notMNIST_data['test_dataset'].reshape((-1, 28 * 28)) + 0.5

dummy_targets = torch.ones(min(ood_num_examples*args.num_to_avg, notMNIST_data.shape[0]))
ood_data = torch.utils.data.TensorDataset(torch.from_numpy(
    notMNIST_data[:ood_num_examples*args.num_to_avg].astype(np.float32)), dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nnotMNIST Detection')
get_and_print_results(ood_loader)

out_score = get_ood_scores(ood_loader)
pos = np.array(out_score[:]).reshape((-1, 1))
neg = np.array(in_score[:]).reshape((-1, 1))
examples = np.squeeze(np.vstack((pos, neg)))
labels = np.zeros(len(examples), dtype=np.int32)
labels[:len(pos)] += 1
fpr, tpr, _ = roc_curve(labels, examples)
np.savetxt('./roc_800/tpr_{}_{}_notmnist.out'.format(args.method_name, args.use_xent), tpr)
np.savetxt('./roc_800/fpr_{}_{}_notmnist.out'.format(args.method_name, args.use_xent), fpr)



