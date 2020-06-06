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
#from sklearn.linear_model import LinearModel #, RegressorMixin
from sklearn.utils import check_consistent_length, check_array, check_X_y

from dsbox_corex.echo_regression.echo_regression import EchoRegression

import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import string

from d3m.metadata.base import PrimitiveMetadata, DataMetadata, ALL_ELEMENTS
import common_primitives.utils as common_utils
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple, NamedTuple
import typing

import dsbox_corex._config as cfg_

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



Input = container.DataFrame
Output = container.DataFrame #typing.Union[container.DataFrame, None]

class EchoRegressor_Params(params.Params):
    fitted_: typing.Union[bool, None]
    model_: typing.Union[EchoRegression, None]
    output_columns_: typing.Union[Any, None]
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
        default = False,
        description = 'assume diagonal covariance, leading to sparsity in data basis (instead of covariance eigenbasis)', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class EchoLinearRegression(SupervisedLearnerPrimitiveBase[Input, Output, EchoRegressor_Params, EchoRegressor_Hyperparams]):  #(Primitive):
    """
    Least squares regression with information capacity constraint from echo noise. Minimizes the objective function::
    E(y - y_hat)^2 + alpha * I(X,y)
    where, X_bar = X + S * echo noise, y_hat = X_bar w + w_0,
    so that I(X,y) <= -log det S,
    with w the learned weights / coefficients.
    The objective simplifies and has an analytic solution.
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "18e63b10-c5b7-34bc-a670-f2c831d6b4bf",
        "version": "1.0.0",
        "name": "EchoLinearRegression",
        "description": "Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.",
        #"python_path": "d3m.primitives.dsbox.echo.EchoRegressor",
        "python_path": "d3m.primitives.regression.echo_linear.DSBOX",
        "original_python_path": "echo_regressor.EchoLinearRegression",
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
        super().__init__(hyperparams = hyperparams)

    # instantiate data and create model and bag of words
    def set_training_data(self, *, inputs: Input, outputs: Output) -> None:
        self.training_data = inputs
        self.labels = outputs
        self._output_columns = outputs.columns
        self.fitted = False
         
    # assumes input as data-frame and do prediction on the 'text' labeled columns
    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult[None]:
        # if already fitted, do nothing
        if self.fitted:
            return CallResult(None, True, 1)
        
        self.model = EchoRegression(alpha = self.hyperparams['alpha'], assume_diagonal = self.hyperparams['diagonal'])

        self.model.fit(self.training_data, self.labels)

        return CallResult(None, True, 1)


    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        try:
            self._output_columns = self._output_columns
        except:
            self._output_columns = ['output']*len(list(output))
        preds = self.model.produce(inputs.values)
 
        output = d3m_DataFrame(preds,  columns = self._output_columns, source = self, generate_metadata = True) #
        output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)
        #output.metadata = self._add_target_semantic_types(metadata=output.metadata, target_names=self._output_columns, source=self)



        self._training_indices = [c for c in inputs.columns if isinstance(c, str) and 'index' in c.lower()]
        outputs = common_utils.combine_columns(return_result='new', #self.hyperparams['return_result'],
                                               add_index_columns=True,#self.hyperparams['add_index_columns'],
                                               inputs=inputs, columns_list=[output], source=self, column_indices=self._training_indices)
        return CallResult(outputs, True, 1)
        


    def get_params(self) -> EchoRegressor_Params:
        return EchoRegressor_Params(fitted_ = self.fitted, model_= self.model, output_columns_ = self._output_columns)

        """
        Sets all the search parameters from a Params object
        :param is_classifier: True for discrete-class output. False for numeric output.
        :type: boolean
        :type: Double
        """
    def set_params(self, *, params: EchoRegressor_Params) -> CallResult[None]:
        self.fitted = params['fitted_']
        self.model = params['model_']
        self._output_columns = params['output_columns_']
        
        return CallResult(None, True, 1)



    def _add_target_semantic_types(cls, metadata: DataMetadata,
                            source: typing.Any,  target_names: List = None,) -> DataMetadata:
        for column_index in range(metadata.query((ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  source=source)
            metadata = metadata.add_semantic_type((ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  source=source)
            if target_names:
                metadata = metadata.update((ALL_ELEMENTS, column_index), {
                    'name': target_names[column_index],
                }, source=source)
        return metadata
