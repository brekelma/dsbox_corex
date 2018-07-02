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
from corextext.corex_topic import Corex
from scipy import sparse as sp

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import d3m.container as container
import d3m.metadata.base as mbase
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

import config as cfg_
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd

Input = container.DataFrame
Output = container.DataFrame

class CorexText_Params(params.Params):
    model_: typing.Union[Corex, None]
    bow_: typing.Union[TfidfVectorizer, None]

# Set hyperparameters according to https://gitlab.com/datadrivendiscovery/d3m#hyper-parameters
class CorexText_Hyperparams(hyperparams.Hyperparams):
    # number of Corex latent factors
    n_hidden = Uniform(
        lower = 0, 
        upper = 100, 
        default = 10, 
        q = 1, 
        description = 'number of topics', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # max_df @ http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    max_df = Uniform(
        lower = 0.0, 
        upper = 1.00, 
        default = .9, 
        q = .05, 
        description = 'max percent document frequency of analysed terms', 
        semantic_types=["http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # min_df @ http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    min_df = Uniform(
        lower = 0.0, 
        upper = 1.00, 
        default = .02, 
        q = .01, 
        description = 'min percent document frequency of analysed terms', 
        semantic_types=["http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # max_features @ http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    max_features = Union(
        OrderedDict([
            ('none', Enumeration([None], default = None)), 
            ('int_mf', UniformInt(
                lower = 1000, 
                upper = 50000, 
                default = 50000, 
                upper_inclusive=True, 
                description = 'max number of terms to use'))]), 
        default = 'none', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # chunking is only used for text datasets
    chunking = Uniform(
        lower = 0, 
        upper = 2000, 
        default = 0, 
        q = 100, 
        description = 'number of tfidf-filtered terms to include as a document, 0 => no chunking.  last chunk may be > param value to avoid small documents', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # read-in documents
    read_docs = UniformBool(
        default=False, 
        description = 'corex parameter regarding what dataset format we are using', 
        semantic_types=["http://schema.org/Bool", 'https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class CorexText(UnsupervisedLearnerPrimitiveBase[Input, Output, CorexText_Params, CorexText_Hyperparams]):  #(Primitive):
    """
    Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "18e63b10-c5b7-34bc-a670-f2c831d6b4bf",
        "version": "1.0.0",
        "name": "CorexText",
        "description": "Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.",
        "python_path": "d3m.primitives.dsbox.CorexText",
        "original_python_path": "corextext.corex_text.CorexText",
        "source": {
            "name": "ISI",
            "contact": "mailto:sstan@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_corex" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM", "LATENT_DIRICHLET_ALLOCATION"],
        "primitive_family": "FEATURE_CONSTRUCTION",
        "hyperparams_to_tune": ["n_hidden", "chunking", "max_df", "min_df", "max_features"]
    })

    def __init__(self, *, hyperparams : CorexText_Hyperparams) -> None: 
        super().__init__(hyperparams = hyperparams)

    # instantiate data and create model and bag of words
    def set_training_data(self, *, inputs : Input) -> None:
        self.training_data = inputs
        self.fitted = False
         
    # assumes input as data-frame and do prediction on the 'text' labeled columns
    def fit(self, *, timeout : float = None, iterations : int = None) -> None:
        # if already fitted, do nothing
        if self.fitted:
            return CallResult(None, True, 1)

        text_attributes = utils.list_columns_with_semantic_types(metadata=self.training_data.metadata,semantic_types=["http://schema.org/Text"])
        all_attributes = utils.list_columns_with_semantic_types(metadata=self.training_data.metadata,semantic_types=["https://metadata.datadrivendiscovery.org/types/Attribute"])
        self.text_columns = list(set(all_attributes).intersection(text_attributes))

        # if no text columns are present don't do anything
        if len(self.text_columns) == 0:
            self.fitted = False
            return CallResult(None, True, 1)

        # instantiate a corex model and a bag of words model
        self.model = Corex(n_hidden= self.hyperparams['n_hidden'], max_iter = iterations, seed = self.random_seed)
        self.bow = TfidfVectorizer(decode_error='ignore', max_df = self.hyperparams['max_df'], min_df = self.hyperparams['min_df'], max_features = self.hyperparams['max_features'])

        # set the number of iterations (for wrapper and underlying Corex model)
        if iterations is not None:
            self.max_iter = iterations
        else:
            self.max_iter = 250
        self.model.max_iter = self.max_iter

        # concatenate the columns row-wise
        concat_cols = None
        for column_index in self.text_columns:
            if concat_cols is not None:
                concat_cols = concat_cols.str.cat(self.training_data.ix[:,column_index])
            else:
                concat_cols = copy.deepcopy(self.training_data.ix[:,column_index])

        bow = self.bow.fit_transform(concat_cols.ravel())

        self.latent_factors = self.model.fit_transform(bow)
        self.fitted = True

        return CallResult(None, True, 1)


    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: 
        # if model was not fitted for any reason (e.g. no text columns present), then just return the input data unchanged
        if not self.fitted:
            return CallResult(inputs, True, 1)

        if iterations is not None:
            self.max_iter = iterations
        else:
            self.max_iter = 250
        self.model.max_iter = self.max_iter

        # concatenate the columns row-wise
        concat_cols = None
        for column_index in self.text_columns:
            if concat_cols is not None:
                concat_cols = concat_cols.str.cat(inputs.ix[:,column_index])
            else:
                concat_cols = copy.deepcopy(inputs.ix[:,column_index])
        bow = self.bow.transform(concat_cols.ravel())
        self.latent_factors = self.model.transform(bow).astype(float)

        # remove the selected columns from input and add the latent factors given by corex
        out_df = d3m_DataFrame(inputs)

        # create metadata for the corex columns
        corex_df = d3m_DataFrame(self.latent_factors)
        for column_index in range(corex_df.shape[1]):
            col_dict = dict(corex_df.metadata.query((mbase.ALL_ELEMENTS, column_index)))
            col_dict['structural_type'] = type(1.0)
            # FIXME: assume we apply corex only once per template, otherwise column names might duplicate
            col_dict['name'] = 'corex_' + str(out_df.shape[1] + column_index)
            col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')

            corex_df.metadata = corex_df.metadata.update((mbase.ALL_ELEMENTS, column_index), col_dict)

        # concatenate is --VERY-- slow without this next line
        corex_df.index = out_df.index.copy()

        out_df = utils.append_columns(out_df, corex_df)

        # remove the initial text column from the df, if we do this before CorEx we can get an empty dataset error
        adjust = 0
        for column_index in self.text_columns:
            out_df = utils.remove_column(out_df, column_index - adjust)
            adjust = adjust + 1

        # TO DO : Incorporate timeout, max_iter
        # return CallResult(d3m_DataFrame(self.latent_factors))
        return CallResult(out_df, True, 1)

    def get_params(self) -> CorexText_Params:
        if not self.fitted:
            raise ValueError("Fit not performed")
        return CorexText_Params(model = self.model_, bow = self.bow_)

    def set_params(self, *, params: CorexText_Params) -> None:
        self.model_ = params['model_']
        self.bow_ = params['bow_']

    def _annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'CorexText'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = 'UnsupervisedLearning'
        self._annotation.ml_algorithm = ['Dimension Reduction']
        self._annotation.tags = ['feature_extraction', 'text']
        return self._annotation
