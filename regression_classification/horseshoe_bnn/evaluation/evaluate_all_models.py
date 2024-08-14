"""
File: evaluate_all_models.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: Trains models LinearGaussian, LinearHorseshoe, GaussianBNN and HorseshoeBNN on a dataset and prints the results
"""

import torch
import pickle
import os
import numpy as np
import torch.optim as optim
import math
import ipdb
import pandas as pd
import datetime
import yaml

from horseshoe_bnn.parameters import BNNRegressionHyperparameters, LinearBNNHyperparameters, LinearHorseshoeHyperparameters, HorseshoeHyperparameters, EvaluationParameters
from horseshoe_bnn.models import LinearGaussian, GaussianBNN, LinearHorseshoe, HorseshoeBNN, GMMBNN, HYHorseshoeBNN1, ReHYHorseshoeBNN2
from horseshoe_bnn.metrics import AllMetrics
from horseshoe_bnn.data_handling.dataset import Dataset
from horseshoe_bnn.evaluation.evaluator import evaluate

#from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

root = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)



#data = np.loadtxt('UCI/boston.txt')
#data = np.loadtxt('UCI/concrete.txt')
#data = np.loadtxt('UCI/energy.txt')
#data = np.loadtxt('UCI/kin8nm.txt')
#data = np.loadtxt('UCI/naval.txt')
#data = np.loadtxt('UCI/power.txt')
#data = np.loadtxt('UCI/protein.txt')
#data = np.loadtxt('UCI/wine.txt')
data = np.loadtxt('UCI/yacht.txt')
#dataset = Dataset(features, labels, 'Year')

'''
#classification
#data.CSV:banana, EEG, HTRU_2, magic04, MiniBooNE_PID, ionosphere, wdbc
filen = os.path.join("data", "banana", "banana.csv")
data = np.loadtxt(filen, delimiter=",")

#filen = os.path.join("UCI", "wdbc.csv")
#data = np.loadtxt(filen, delimiter=",")

'''

features = data[ :, range(data.shape[ 1 ] - 1) ]
labels = data[ :, data.shape[ 1 ] - 1 ]
print(features.shape)
print(labels.shape)

#dataset = Dataset(features, labels, 'boston')
#dataset = Dataset(features, labels, 'concrete')
#dataset = Dataset(features, labels, 'energy')
#dataset = Dataset(features, labels, 'kin8nm')
#dataset = Dataset(features, labels, 'naval')
#dataset = Dataset(features, labels, 'power')
#dataset = Dataset(features, labels, 'protein')
#dataset = Dataset(features, labels, 'wine')
dataset = Dataset(features, labels, 'yacht')
#dataset = Dataset(features, labels, 'Year')

#dataset = Dataset(features, labels, 'banana')

# We create the train and test sets with 90% and 10% of the data




"""
Set number of epochs the models should be trained for
"""
n_epochs = 600
def run_evaluation(config_path, create_hyperparameters, model_instance, metrics):
    with open(config_path) as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
        config['n_features'] = dataset.features.shape[1]
        config['timestamp'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config['dataset_name'] = dataset.name
        hyperparams = create_hyperparameters(**config)

    model = model_instance(device, hyperparams).to(device)
    optimizer = optim.SGD(model.parameters(), lr=hyperparams.learning_rate)
    evaluationparams = EvaluationParameters(n_splits=10,
                                            scaler=False,
                                            normalize=True,
                                            n_epochs=n_epochs,
                                            poly_features=False,
                                            learning_rate = hyperparams.learning_rate,
                                            optimizer=optimizer)

    results, _ =  evaluate(model, dataset, metrics, evaluationparams, config, save=True)

    print('###################################################')
    print(f"RESULTS {model.__class__.__name__} MODEL")
    print('###################################################')
    results.print()
    print()


"""
Evaluate all models
"""
config_horseshoeBNN = os.path.join(root, 'configs/horseshoeBNN_config.yaml')
config_linearHorseshoe = os.path.join(root, 'configs/linear_horseshoe.yaml')
config_gaussianBNN = os.path.join(root, 'configs/bnn_config.yaml')
config_GMMBNN = os.path.join(root, 'configs/gmm_bnn_config.yaml')
config_linearGaussian = os.path.join(root, 'configs/linearBNN_config.yaml')
config_rehy_horseshoeBNN = os.path.join(root, 'configs/hy_horseshoeBNN_config.yaml')

#regression
metrics = [AllMetrics.mae, AllMetrics.rmse, AllMetrics.logprob]

#classification
#metrics = [AllMetrics.zero_one_loss, AllMetrics.global_f1_score, AllMetrics.f1_score]
#metrics = [AllMetrics.accuracy_s, AllMetrics.calibration_ece, AllMetrics.calibration_nll]



#run_evaluation(config_horseshoeBNN, HorseshoeHyperparameters, HorseshoeBNN, metrics)
#run_evaluation(config_horseshoeBNN, HorseshoeHyperparameters, HYHorseshoeBNN1, metrics)
run_evaluation(config_rehy_horseshoeBNN, HorseshoeHyperparameters, ReHYHorseshoeBNN2, metrics)
##run_evaluation(config_linearHorseshoe, LinearHorseshoeHyperparameters, LinearHorseshoe, metrics)
#run_evaluation(config_gaussianBNN, BNNRegressionHyperparameters, GaussianBNN, metrics)
#run_evaluation(config_GMMBNN, BNNRegressionHyperparameters, GMMBNN, metrics)
##run_evaluation(config_linearGaussian, LinearBNNHyperparameters, LinearGaussian, metrics)

