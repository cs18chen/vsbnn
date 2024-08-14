"""
File: model.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: All model classes
"""
import torch.utils.data
from .network_layers import HorseshoeLayer, RegHorseshoeLayer,BayesianLinear, BayesianLayer
import math
import numpy as np
import torch
num_batches=550
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 100

NUM_LABELS = 10
PI = 0.5

SIGMA1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA2 = torch.cuda.FloatTensor([math.exp(-6)])
IMG_HEIGHT, IMG_WIDTH = 28, 28
num_pixels = IMG_HEIGHT * IMG_WIDTH

NUM_SAMPLES = 20
classification = False
SIGMA = torch.tensor([math.exp(-2)])
def compute_log_likelihoods(classification, outputs, target, num_samples):
    if classification:
        log_likelihoods = - F.binary_cross_entropy_with_logits(outputs, target, reduction='none')
        log_likelihoods = torch.sum(log_likelihoods, dim = 1)
    else:
        log_likelihoods =  torch.sum(-0.5 * torch.log(2 * math.pi * SIGMA) - 0.5 * (target - outputs) ** 2 / SIGMA, dim=1)

    return log_likelihoods
'''
class Model(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass
'''

class HorseshoeNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.l1 = HorseshoeLayer(28*28*n_channels, 800)
        self.l2 = HorseshoeLayer(800, 800)
        self.l3 = HorseshoeLayer(800, 800)
        self.l4 = HorseshoeLayer(800, n_classes)



    def forward(self, x, take_sample=True):
       # x = x.view(-1, 28*28)
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x, take_sample))
        x = F.relu(self.l2(x, take_sample))
        x = F.relu(self.l3(x, take_sample))
        x = self.l4(x, take_sample)

        x =  F.log_softmax(x, dim=1)
        return x


    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], 10)

        for i in range(Nsamples):
            y = self.forward(x, take_sample=True)
            predictions[i] = y

        return predictions
   


    def initialize(self, nn_input_size):
        """
        Reset model parameters
        """
        self.__init__(self)
        return self

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior + self.l4.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior  + self.l4.log_variational_posterior

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        return self.l1.analytic_update(), self.l2.analytic_update(), self.l3.analytic_update(), self.l4.analytic_update()

    # sample a bunch of weights for the network
    # make predictions using sampled weights
    # output averaged predictions from different sampled weights

    def loss(self, input, target, num_samples, num_classes, num_batches):
        """Variational free energy/negative ELBO loss function, called
                   f(w, theta) in the paper
                   NB calling model.loss() does a forward pass, so in train() function
                   we don't need to call model(input)
                """
        criterion = nn.CrossEntropyLoss()
        batch_size = target.size()[0]
        outputs = torch.zeros(num_samples, batch_size, num_classes).cuda()  # create tensors on the GPU
        log_priors = torch.zeros(num_samples).cuda()
        log_variational_posteriors = torch.zeros(num_samples).cuda()
        for i in range(num_samples):
            outputs[i] = self(input)  # note 4
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        # the following line might be wrong -- change it if something's goin wrong
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target,
                                             reduction='sum')  # we want to sum over (y_i.\hat{y_i}); y_i = 0 for all units except true output. Pretty sure this is identical to size_average=False


        #negative_log_likelihood = criterion(outputs.mean(0), target)

        loss = (log_variational_posterior - log_prior) / num_batches  + negative_log_likelihood
        return loss  # they also return log_prior, log_variational_posterior, negative_log_likelihood



    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out


class BNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.layer1 = BayesianLayer(28*28*n_channels, 800)
        self.layer2 = BayesianLayer(800, 800)
        self.layer3 = BayesianLayer(800, 800)
        self.layer4 = BayesianLayer(800, n_classes)

    def forward(self, x, take_sample=True):  # note 5
       # x = x.view(-1, 28 * 28)  # dim: batch size x 784
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x, take_sample))
        x = F.relu(self.layer2(x, take_sample))
        x = F.relu(self.layer3(x, take_sample))
        x = self.layer4(x, take_sample)

        x =  F.log_softmax(x, dim=1)
        return x


    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], 10)

        for i in range(Nsamples):
            y = self.forward(x, take_sample=True)
            predictions[i] = y

        return predictions

    def log_prior(self):
        '''log probability of the current prior parameters is the sum
           of those parameters for each layer.
           These get updated here (*) each time we do a forward pass
           This implies forward() must be called before finding the log lik
           of the posterior and prior parameters (in the loss func)!
        '''
        return self.layer1.log_prior \
               + self.layer2.log_prior \
               + self.layer3.log_prior + self.layer4.log_prior

    def log_variational_posterior(self):
        return self.layer1.log_variational_posterior + \
               self.layer2.log_variational_posterior + \
               self.layer3.log_variational_posterior + self.layer4.log_variational_posterior

    def loss(self, input, target, num_samples, num_classes, num_batches):
        """Variational free energy/negative ELBO loss function, called
                   f(w, theta) in the paper
                   NB calling model.loss() does a forward pass, so in train() function
                   we don't need to call model(input)
                """
        criterion = nn.CrossEntropyLoss()
        batch_size = target.size()[0]
        outputs = torch.zeros(num_samples, batch_size, num_classes).cuda()  # create tensors on the GPU
        log_priors = torch.zeros(num_samples).cuda()
        log_variational_posteriors = torch.zeros(num_samples).cuda()
        for i in range(num_samples):
            outputs[i] = self(input)  # note 4
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        # the following line might be wrong -- change it if something's goin wrong
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target,
           reduction='sum')  # we want to sum over (y_i.\hat{y_i}); y_i = 0 for all units except true output. Pretty sure this is identical to size_average=False


        #negative_log_likelihood = criterion(outputs.mean(0), target)

        loss = (log_variational_posterior - log_prior)  / num_batches + negative_log_likelihood
        return loss  # they also return log_prior, log_variational_posterior, negative_log_likelihood



    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out


class HPBayesianNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.layer1 = HorseshoeLayer(28*28*n_channels, 800)
        #                                     (-1.5, 1.5, -5, -2, -1.5, 1.5, -5, -2) ) #(-1.27, 1.25, -1, -0.1, -1, 1, -2, -1) )
        self.layer2 = HorseshoeLayer(800, 800)
        self.layer3= HorseshoeLayer(800, 800)
        self.layer4 = BayesianLinear(800, n_classes)


    def forward(self, x, take_sample=True):  # note 5
        #x = x.view(-1, 28*28)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x, take_sample))
        x = F.relu(self.layer2(x, take_sample))
        x = F.relu(self.layer3(x, take_sample))
        x = self.layer4(x, take_sample)
        x =  F.log_softmax(x, dim=1)
        return x


    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], 10)

        for i in range(Nsamples):
            y = self.forward(x, take_sample=True)
            predictions[i] = y

        return predictions



    def initialize(self, nn_input_size):
        """
        Reset model parameters
        """
        self.__init__(self)
        return self

    def log_prior(self):
        '''log probability of the current prior parameters is the sum
           of those parameters for each layer.
           These get updated here (*) each time we do a forward pass
           This implies forward() must be called before finding the log lik
           of the posterior and prior parameters (in the loss func)!
        '''
        return self.layer1.log_prior \
               + self.layer2.log_prior \
               + self.layer3.log_prior+ self.layer4.log_prior

    def log_variational_posterior(self):
        '''log probability of the current posterior parameters is the sum
            of those parameters for each layer
        '''
        return self.layer1.log_variational_posterior + \
               self.layer2.log_variational_posterior + \
               self.layer3.log_variational_posterior + self.layer4.log_variational_posterior

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        return self.layer1.analytic_update(), self.layer2.analytic_update()

    def loss(self, input, target, num_samples, num_classes, num_batches):
        """Variational free energy/negative ELBO loss function, called
                   f(w, theta) in the paper
                   NB calling model.loss() does a forward pass, so in train() function
                   we don't need to call model(input)
                """
        criterion = nn.CrossEntropyLoss()
        batch_size = target.size()[0]
        outputs = torch.zeros(num_samples, batch_size, num_classes).cuda()  # create tensors on the GPU
        log_priors = torch.zeros(num_samples).cuda()
        log_variational_posteriors = torch.zeros(num_samples).cuda()
        for i in range(num_samples):
            outputs[i] = self(input)  # note 4
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        # the following line might be wrong -- change it if something's goin wrong
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target,
           reduction='sum')  # we want to sum over (y_i.\hat{y_i}); y_i = 0 for all units except true output. Pretty sure this is identical to size_average=False

        #negative_log_likelihood = criterion(outputs.mean(0), target)

        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss  # they also return log_prior, log_variational_posterior, negative_log_likelihood


    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

class HPBayesianNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.layer1 = HorseshoeLayer(28*28*n_channels, 800)
        #                                     (-1.5, 1.5, -5, -2, -1.5, 1.5, -5, -2) ) #(-1.27, 1.25, -1, -0.1, -1, 1, -2, -1) )
        self.layer2 = BayesianLinear(800, 800)
        self.layer3 = BayesianLinear(800, 800)
        self.layer4 = BayesianLinear(800, n_classes)


    def forward(self, x, take_sample=True):  # note 5
        #x = x.view(-1, 28*28)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x, take_sample))
        x = F.relu(self.layer2(x, take_sample))
        x = F.relu(self.layer3(x, take_sample))
        x = self.layer4(x, take_sample)
        x = F.log_softmax(x, dim=1)
        return x


    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], 10)

        for i in range(Nsamples):
            y = self.forward(x, take_sample=True)
            predictions[i] = y

        return predictions



    def initialize(self, nn_input_size):
        """
        Reset model parameters
        """
        self.__init__(self)
        return self

    def log_prior(self):
        '''log probability of the current prior parameters is the sum
           of those parameters for each layer.
           These get updated here (*) each time we do a forward pass
           This implies forward() must be called before finding the log lik
           of the posterior and prior parameters (in the loss func)!
        '''
        return self.layer1.log_prior \
               + self.layer2.log_prior \
               + self.layer3.log_prior + self.layer4.log_prior

    def log_variational_posterior(self):
        '''log probability of the current posterior parameters is the sum
            of those parameters for each layer
        '''
        return self.layer1.log_variational_posterior + \
               self.layer2.log_variational_posterior + \
               self.layer3.log_variational_posterior + self.layer4.log_variational_posterior

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        return self.layer1.analytic_update()

    def loss(self, input, target, num_samples, num_classes, num_batches):
        """Variational free energy/negative ELBO loss function, called
                   f(w, theta) in the paper
                   NB calling model.loss() does a forward pass, so in train() function
                   we don't need to call model(input)
                """
        criterion = nn.CrossEntropyLoss()
        batch_size = target.size()[0]
        outputs = torch.zeros(num_samples, batch_size, num_classes).cuda()  # create tensors on the GPU
        log_priors = torch.zeros(num_samples).cuda()
        log_variational_posteriors = torch.zeros(num_samples).cuda()
        for i in range(num_samples):
            outputs[i] = self(input)  # note 4
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        # the following line might be wrong -- change it if something's goin wrong
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target,
           reduction='sum')  # we want to sum over (y_i.\hat{y_i}); y_i = 0 for all units except true output. Pretty sure this is identical to size_average=False

        #negative_log_likelihood = criterion(outputs.mean(0), target)

        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss  # they also return log_prior, log_variational_posterior, negative_log_likelihood



    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

class RegHPBayesianNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.layer1 = RegHorseshoeLayer(28*28*n_channels, 800)
        #                                     (-1.5, 1.5, -5, -2, -1.5, 1.5, -5, -2) ) #(-1.27, 1.25, -1, -0.1, -1, 1, -2, -1) )
        self.layer2 = RegHorseshoeLayer(800, 800)
        self.layer3 = RegHorseshoeLayer(800, 800)
        self.layer4 = BayesianLinear(800, n_classes)


    def forward(self, x, take_sample=True):  # note 5
        #x = x.view(-1, 28*28)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x, take_sample))
        x = F.relu(self.layer2(x, take_sample))
        x = F.relu(self.layer3(x, take_sample))
        x = self.layer4(x, take_sample)
        x = F.log_softmax(x, dim=1)
        return x


    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], 10)

        for i in range(Nsamples):
            y = self.forward(x, take_sample=True)
            predictions[i] = y

        return predictions



    def initialize(self, nn_input_size):
        """
        Reset model parameters
        """
        self.__init__(self)
        return self

    def log_prior(self):

        '''log probability of the current prior parameters is the sum

           of those parameters for each layer.
           These get updated here (*) each time we do a forward pass
           This implies forward() must be called before finding the log lik
           of the posterior and prior parameters (in the loss func)!
        '''
        return self.layer1.log_prior \
               + self.layer2.log_prior \
               + self.layer3.log_prior + self.layer4.log_prior

    def log_variational_posterior(self):
        '''log probability of the current posterior parameters is the sum
            of those parameters for each layer
        '''
        return self.layer1.log_variational_posterior + \
               self.layer2.log_variational_posterior + \
               self.layer3.log_variational_posterior + self.layer4.log_variational_posterior

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        return self.layer1.analytic_update(),self.layer2.analytic_update()

    def loss(self, input, target, num_samples, num_classes, num_batches):
        """Variational free energy/negative ELBO loss function, called
                   f(w, theta) in the paper
                   NB calling model.loss() does a forward pass, so in train() function
                   we don't need to call model(input)
                """
        criterion = nn.CrossEntropyLoss()
        batch_size = target.size()[0]
        outputs = torch.zeros(num_samples, batch_size, num_classes).cuda()  # create tensors on the GPU
        log_priors = torch.zeros(num_samples).cuda()
        log_variational_posteriors = torch.zeros(num_samples).cuda()
        for i in range(num_samples):
            outputs[i] = self(input)  # note 4
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        # the following line might be wrong -- change it if something's goin wrong
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target,
           reduction='sum')  # we want to sum over (y_i.\hat{y_i}); y_i = 0 for all units except true output. Pretty sure this is identical to size_average=False

        #negative_log_likelihood = criterion(outputs.mean(0), target)

        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss  # they also return log_prior, log_variational_posterior, negative_log_likelihood



    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

class Dropout_MLP(nn.Module):
    '''MLP with dropout on both hidden layers
       p=0.5, as per the original paper
    '''
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(28*28*n_channels, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 800)
        self.fc4 = nn.Linear(800, n_classes)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.2)
    def forward(self, x, take_sample=True):
        #x = x.view(-1, 28*28) # dimensions should be batch size x 784
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        x = self.drop4(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x # note 2


    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], 10)

        for i in range(Nsamples):
            y = self.forward(x, take_sample=True)
            predictions[i] = y

        return predictions



  #  def eval(self, x, y, train=False):
   #     x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

  #      out = self.model(x)

   #     loss = F.cross_entropy(out, y, reduction='sum')

     #   probs = F.softmax(out, dim=1).data.cpu()

    #    pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
    #    err = pred.ne(y.data).sum()

  #      return loss.data, err, probs

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

class MLP(nn.Module):
    def __init__(self,n_channels, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(28*28*n_channels, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 800)
        self.fc4 = nn.Linear(800, n_classes)

    def forward(self, x):
        #x = x.view(-1, 28*28)  # dimensions should be batch size x 784
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        y_hat = self.fc4(h3)
        y_hat = F.log_softmax(y_hat, dim=1)

        return y_hat  # note 2

