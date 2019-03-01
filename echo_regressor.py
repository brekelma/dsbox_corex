import copy
import git
import os
import re
import sys
import typing

import pandas as pd
import numpy as np

from collections import defaultdict, OrderedDict
from common_primitives import utils

from scipy import linalg
from sklearn.linear_model.base import LinearModel, RegressorMixin
from sklearn.utils import check_consistent_length, check_array, check_X_y

from echo_regression import EchoRegression

import d3m.container as container
import d3m.metadata.base as mbase
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import SupervisedLearnerPrimitiveBase

import config as cfg_
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import string

Input = container.DataFrame
Output = container.DataFrame #typing.Union[container.DataFrame, None]

class EchoRegressor_Params(params.Params):
    fitted_: typing.Union[bool, None]
    model_: typing.Union[EchoRegression, None]
    #latent_factors_: typing.Union[pd.DataFrame, None]
    #max_iter_: typing.Union[int, None]

# Set hyperparameters according to https://gitlab.com/datadrivendiscovery/d3m#hyper-parameters
class EchoRegressor_Hyperparams(hyperparams.Hyperparams):
    # regularization strength
    alpha = Uniform(
        lower = 0, 
        upper = 10, 
        default = 1, 
        q = .1, 
        description = 'regularization strength', 
        semantic_types=["http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # 
    diagonal = UniformBool(
    	default = True,
        description = 'assume diagonal covariance, leading to sparsity in data basis (instead of covariance eigenbasis)', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class EchoRegressor(SupervisedLearnerPrimitiveBase[Input, Output, EchoRegressor_Params, EchoRegressor_Hyperparams]):  #(Primitive):
    """
    Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "18e63b10-c5b7-34bc-a670-f2c831d6b4bf",
        "version": "1.0.0",
        "name": "EchoRegressor",
        "description": "Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.",
        #"python_path": "d3m.primitives.dsbox.echo.EchoRegressor",
        "python_path": "d3m.primitives.regression.echo.EchoRegressor",
        "original_python_path": "EchoRegressor.echo.EchoRegressor",
        "source": {
            "name": "ISI",
            "contact": "mailto:brekelma@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_corex" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": ["LINEAR_REGRESSION"],
        "primitive_family": "REGRESSION",
        "hyperparams_to_tune": ["alpha"]
    })

    def __init__(self, *, hyperparams : EchoRegressor_Hyperparams) -> None: 
        super(EchoRegressor, self).__init__(hyperparams = hyperparams)

    # instantiate data and create model and bag of words
    def set_training_data(self, *, inputs: Input, outputs: Output) -> CallResult:
        self.training_data = inputs
        self.labels = outputs
        self.fitted = False
         
    # assumes input as data-frame and do prediction on the 'text' labeled columns
    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult:
        # if already fitted, do nothing
        if self.fitted:
            return CallResult(None, True, 1)
        
        self.model = EchoRegression(alpha = self.hyperparams['alpha'], assume_diagonal = self.hyperparams['diagonal'])

        self.model.fit(self.training_data, self.labels)

       	return CallResult(None, True, 1)


	def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult:
		return CallResult(self.model.produce(inputs), True, 1)


	def get_params(self) -> SearchParams:
		return EchoRegressor_Params(fitted_ = self.fitted, model_= self.model)

	"""
	Sets all the search parameters from a Params object
	:param is_classifier: True for discrete-class output. False for numeric output.
	:type: boolean
	:type: Double
	"""
	def set_params(self, *, params: SearchParams) -> base.CallResult[None]:
		self.fitted = params['fitted_']
		self.model = params['model_']
		return CallResult(None)


