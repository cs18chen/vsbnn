"""
File: metrics.py
Author: Hiske Overweg, Anna-Lena Popkes
Description: All metric classes
"""
import torch
import numpy as np
from abc import ABCMeta, abstractmethod
from horseshoe_bnn.distributions import PredictiveDistribution, BinarySampleDistribution
from horseshoe_bnn.aggregation_result import AveragedResult, CalibrationResult, HistogramResult, UncertaintyResult
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score
import sklearn
from .util.metrics import accuracy, nll, brier, calibration
from torchmetrics.functional.classification import calibration_error
def calibration_test(p, y, nbins=10):
    '''
    Returns ece:  Expected Calibration Error
            conf: confindence levels (as many as nbins)
            accu: accuracy for a certain confidence level
                  We are interested in the plot confidence vs accuracy
            bin_sizes: how many points lie within a certain confidence level
    '''
    edges = np.linspace(0, 1, nbins + 1)
    accu = np.zeros(nbins)
    conf = np.zeros(nbins)
    bin_sizes = np.zeros(nbins)
    # Multiclass problems are treated by considering the max
    if p.ndim > 1 and p.shape[1] != 1:
        pred = np.argmax(p, axis=1)
        p = np.max(p, axis=1)
    else:
        # the treatment for binary classification
        pred = np.ones(p.size)
    #
    y = y.flatten()
    p = p.flatten()
    for i in range(nbins):
        idx_in_bin = (p > edges[i]) & (p <= edges[i + 1])
        bin_sizes[i] = max(sum(idx_in_bin), 1)
        accu[i] = np.sum(y[idx_in_bin] == pred[idx_in_bin]) / bin_sizes[i]
        conf[i] = (edges[i + 1] + edges[i]) / 2
    ece = np.sum(np.abs(accu - conf) * bin_sizes) / np.sum(bin_sizes)
    return ece, conf, accu

class Metric(metaclass=ABCMeta):

    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def compute():
        pass

    @property
    def name(self):
        return self.metric_name

    @staticmethod
    def aggregate(results_of_folds):
        """
        Conputes the mean and standard deviation given the results across
        several folds of crossvalidation.

        Args:
            results_of_folds: list containing the metric values for each fold

        Returns:
            an instance of the AveragedResult class

        Raises:
            TypeError: if the input is not a list
            ValueError: if the input is an empty list
        """
        if not isinstance(results_of_folds, list):
            raise TypeError("The results from cross-validation should be stored in a list!")

        if len(results_of_folds) == 0:
            raise ValueError("The results list is empty!")

        mean = np.mean(results_of_folds)
        std = np.std(results_of_folds)
        return AveragedResult(mean, std)

    def print(self, aggregationResult):
        """
        Prints metric name and the computed metric values.

        Args:
            aggregationResult: instance of the AggregationResult class
        """
        print(f'{self.metric_name}')
        aggregationResult.print()


class MeanAbsoluteError(Metric):
    def __init__(self):
        self.metric_name = 'Mean Absolute Error'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes the mean absolute error between given ground_truth and predictions.

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            mean absolute error (float)

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        predicted_labels = predictiveDistribution.get_all_means()
        return np.mean(np.abs(ground_truth - predicted_labels))


class RootMeanSquaredError(Metric):

    def __init__(self):
        self.metric_name = 'Root Mean Squared Error'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes the root mean squared error between given ground_truth and predictions.

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            root mean squared error (float)

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        predicted_labels = predictiveDistribution.get_all_means()
        return np.sqrt(((ground_truth - predicted_labels) ** 2).mean())




class PredictiveLogLikelihood(Metric):

    def __init__(self):
        self.metric_name = 'Predictive Log-Likelihood'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes the predictive log likelihood between given ground_truth and predictions.

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            predictive log likelihood (float)

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        loglike = 0
        for i, label in enumerate(ground_truth):
            distr = predictiveDistribution.distributions[i]
            loglike += distr.logprob(label) / ground_truth.shape[0]

        return loglike


class CalibrationPlot(Metric):

    def __init__(self):
        self.metric_name = 'Calibration plot'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes the calibration plot values given ground_truth values and a
        predictive distribution.

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            dictionary containing the extracted variances and absolute errors

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        predicted_variance = predictiveDistribution.get_all_variances()
        error = np.abs(ground_truth - predictiveDistribution.get_all_means())
        return {'var': predicted_variance, 'err': error}

    @staticmethod
    def aggregate(results_of_folds):
        """
        Extracts the variances and absolute errors given the results across several
        folds of cross-validation.

        Args:
            results_of_folds: list, containing the results across the folds

        Returns:
            instance of the CalibrationResult class
        """
        all_variances = []
        all_errors = []
        for result in results_of_folds:
            all_variances.extend(result['var'])
            all_errors.extend(result['err'])
        idx = np.argsort(all_variances)[::-1]
        return CalibrationResult(np.array(all_variances)[idx], np.array(all_errors)[idx])

    def print(self, aggregationResult):
        axis_labels = ['var_index', '|y_test - y_pred|']
        aggregationResult.print(axis_labels, self.metric_name)

class Histogram(Metric):

    def __init__(self):
        self.metric_name = 'Prediction Histogram'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes the ratio of mean prediction and standard deviation for all data points

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            numpy array containing the extracted values of mean prediction / standard deviation

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        predicted_mean = predictiveDistribution.get_all_means()
        predicted_std = np.sqrt(predictiveDistribution.get_all_variances())
        ratio = predicted_mean / predicted_std
        return ratio

    @staticmethod
    def aggregate(results_of_folds):
        """
        Extracts the variances and absolute errors given the results across several
        folds of cross-validation.

        Args:
            results_of_folds: list, containing the results across the folds

        Returns:
            instance of the CalibrationResult class
        """
        all_ratios = []
        for result in results_of_folds:
            all_ratios.extend(result)
        return HistogramResult(all_ratios)

    def print(self, aggregationResult):
        axis_labels = ['mean / standard deviation', 'number of occurrences']
        aggregationResult.print(axis_labels, self.metric_name)




class Uncertainty(Metric):

    def __init__(self):
        self.metric_name = 'Prediction Uncertainty'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes the ratio of mean prediction and standard deviation for all data points

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            numpy array containing the extracted values of mean prediction / standard deviation

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        predicted_mean = predictiveDistribution.get_all_means()
        predicted_std = np.sqrt(predictiveDistribution.get_all_variances())
        return predicted_mean, predicted_std

    @staticmethod
    def aggregate(results_of_folds):
        """
        Extracts the variances and absolute errors given the results across several
        folds of cross-validation.

        Args:
            results_of_folds: list, containing the results across the folds

        Returns:
            instance of the CalibrationResult class
        """
        all_mean= []
        all_std = []
        for result in results_of_folds:
            all_mean.extend(result)
            all_std.extend(result)
        return UncertaintyResult(all_mean, all_std)

    def print(self, aggregationResult):
        axis_labels = ['mean / standard deviation', 'number of occurrences']
        aggregationResult.print(axis_labels, self.metric_name)

class ZeroOneLoss(Metric):

    def __init__(self):
        self.metric_name = 'Zero One Loss'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        """
        Computes zero one loss of a ground_truth array and corresponding predictions.

        Args:
            ground_truth: numpy array of ground truth values
            predictiveDistribution: instance of the PredictiveDistribution class

        Returns:
            zero one loss: float, fraction of missclassifications

        Raises:
            TypeError: if given ground truth labels are not a numpy array
            TypeError: if given predictive distribution is not a valid instance of the
                        PredictiveDistribution class
        """
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("The ground truth labels should be stored in a numpy array")

        if not isinstance(predictiveDistribution, PredictiveDistribution):
            raise TypeError("Please insert a valid predictive distribution instance")

        predictions = predictiveDistribution.get_all_point_estimates()
        fraction = 1 - np.sum(ground_truth == predictions) / len(predictions)

        return fraction

class F1Score(Metric):
    def __init__(self):
        self.metric_name = 'F1 score'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        class_predictions = np.round(predictiveDistribution.get_all_means())
        classes_present = len(np.unique(ground_truth))

        score = f1(ground_truth, class_predictions, average=None)

        score = np.sum(score)/classes_present
        return score


class Accuracy_score(Metric):
    def __init__(self):
        self.metric_name = 'accuracy_score'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        class_predictions = np.round(predictiveDistribution.get_all_means())
        # count number of classes occuring in ground truth
        classes_present = len(np.unique(ground_truth))

        score = accuracy_score(ground_truth, class_predictions)





        return score


class GlobalF1Score(Metric):
    def __init__(self):
        self.metric_name = 'global F1 score'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        class_predictions = np.round(predictiveDistribution.get_all_means())
        score = f1(ground_truth, class_predictions, average='micro')
        return score




class Calibration_accuracy(Metric):
    def __init__(self):
        self.metric_name = 'accuracy'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        class_predictions = np.round(predictiveDistribution.get_all_means())
        # count number of classes occuring in ground truth
        classes_present = len(np.unique(ground_truth))

        accuracy_MAP = accuracy(class_predictions, ground_truth)

        return accuracy_MAP


class Calibration_ece(Metric):
    def __init__(self):
        self.metric_name = 'ece'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        class_predictions = np.round(predictiveDistribution.get_all_means())
        # count number of classes occuring in ground truth
        classes_present = len(np.unique(ground_truth))

        ece_map, conf, accu = calibration_test(class_predictions, ground_truth)
        return ece_map


class Calibration_nll(Metric):
    def __init__(self):
        self.metric_name = 'NLL'

    @staticmethod
    def compute(ground_truth, predictiveDistribution):
        class_predictions = np.round(predictiveDistribution.get_all_means())
        # count number of classes occuring in ground truth
        classes_present = len(np.unique(ground_truth))
        return nll_MAP


class AllMetrics:
    """
    Helper class to instantiate all metrics
    """
    mae = MeanAbsoluteError()
    rmse = RootMeanSquaredError()
    logprob = PredictiveLogLikelihood()
    calibration_plot = CalibrationPlot()
    histogram = Histogram()
    zero_one_loss = ZeroOneLoss()
    f1_score = F1Score()
    accuracy_s=Accuracy_score()
    global_f1_score = GlobalF1Score()
    calibration_accuracy= Calibration_accuracy()
    calibration_ece= Calibration_ece()
    calibration_nll= Calibration_nll()



