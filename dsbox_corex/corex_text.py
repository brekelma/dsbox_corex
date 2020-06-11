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

try:
    from dsbox_corex.corextext.corex_topic import Corex
except:
    from corextext.corex_topic import Corex

#from corextext.corex_topic import Corex
from scipy import sparse as sp

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import d3m.container as container
from d3m.metadata.base import ALL_ELEMENTS
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import PrimitiveMetadata, DataMetadata
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase


import dsbox_corex._config as cfg_

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import string

Input = container.DataFrame
Output = container.DataFrame #typing.Union[container.DataFrame, None]

class CorexText_Params(params.Params):
    fitted_: typing.Union[bool, None]
    model_: typing.Union[Corex, None]
    bow_: typing.Union[TfidfVectorizer, None]
    do_nothing_: typing.Union[bool, None]
    text_columns_: typing.Union[typing.List[int], None]
    latent_factors_: typing.Union[pd.DataFrame, None]
    max_iter_: typing.Union[int, None]

# Set hyperparameters according to https://gitlab.com/datadrivendiscovery/d3m#hyper-parameters
class CorexText_Hyperparams(hyperparams.Hyperparams):
    # number of Corex latent factors
    n_hidden = UniformInt(
        lower = 1, 
        upper = 301, 
        default = 30, 
        description = 'number of topics', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # 
    threshold = Uniform(
        lower = 0,
        upper = 10000, 
        default = 0, 
        q = 1, 
        description = 'threshold for number of columns in the tfidf matrix below which we don`t call CorEx', 
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # 
    n_grams = UniformInt(
        lower = 1,
        upper = 10,
        default = 1, 
        description = 'n_grams parameter to use before feeding in text to TfidfVectorizer', 
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


class CorexText(UnsupervisedLearnerPrimitiveBase[Input, Output, CorexText_Params, CorexText_Hyperparams]):  #(Primitive):
    """
    Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "0c64ffd6-cb9e-49f0-b7cb-abd70a5a8261",
        "version": "1.0.0",
        "name": "CorexText",
        "description": "Learns latent factors / topics which explain the most multivariate information in bag of words representations of documents. Returns learned topic scores for each document. Also supports hierarchical models and 'anchoring' to encourage topics to concentrate around desired words.",
        #"python_path": "d3m.primitives.dsbox.corex_text.CorexText",
        "python_path": "d3m.primitives.feature_construction.corex_text.DSBOX",
        "original_python_path": "corextext.corex_text.CorexText",
        "source": {
            "name": "ISI",
            "contact": "mailto:sstan@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_corex" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM", "LATENT_DIRICHLET_ALLOCATION"],
        "primitive_family": "FEATURE_CONSTRUCTION",
        "hyperparams_to_tune": ["n_hidden", "threshold", "n_grams", "max_df", "min_df"]
    })

    def __init__(self, *, hyperparams : CorexText_Hyperparams) -> None: 
        super(CorexText, self).__init__(hyperparams = hyperparams)

    # instantiate data and create model and bag of words
    def set_training_data(self, *, inputs : Input) -> None:
        self.training_data = inputs
        self.fitted = False
         
    # assumes input as data-frame and do prediction on the 'text' labeled columns
    def fit(self, *, timeout : float = None, iterations : int = None) -> None:
        # if already fitted, do nothing
        if self.fitted:
            return CallResult(None, True, 1)

        self.training_data = self._process_files(self.training_data)

        text_attributes = DataMetadata.list_columns_with_semantic_types(self=self.training_data.metadata,\
            semantic_types=["http://schema.org/Text"])
        all_attributes = DataMetadata.list_columns_with_semantic_types(self=self.training_data.metadata,\
            semantic_types=["https://metadata.datadrivendiscovery.org/types/Attribute"])
        categorical_attributes = DataMetadata.list_columns_with_semantic_types(self=self.training_data.metadata,\
            semantic_types=["https://metadata.datadrivendiscovery.org/types/CategoricalData"])

        # want text columns that are attributes
        self.text_columns = set(all_attributes).intersection(text_attributes)

        # but, don't want to edit categorical columns
        self.text_columns = set(self.text_columns) - set(categorical_attributes)

        # and, we want the text columns as a list 
        self.text_columns = list(self.text_columns)

        # if no text columns are present don't do anything
        self.do_nothing = False
        if len(self.text_columns) == 0:
            self.fitted = True

            self.model = None
            self.bow = None
            self.do_nothing = True
            self.text_columns = None
            self.latent_factors = None
            self.max_iter = None
            
            return CallResult(None, True, 1)

        # instantiate a corex model and a bag of words model
        self.model = Corex(n_hidden = self.hyperparams['n_hidden'], max_iter = iterations, seed = self.random_seed)
        self.bow = TfidfVectorizer(decode_error='ignore', max_df = self.hyperparams['max_df'], min_df = self.hyperparams['min_df'])

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
                concat_cols = concat_cols.str.cat(self.training_data.iloc[:,column_index], sep = " ")
            else:
                concat_cols = copy.deepcopy(self.training_data.iloc[:,column_index])

        try:
            bow = self.bow.fit_transform(map(self._get_ngrams, concat_cols.ravel()))
        except ValueError:
            self.bow = TfidfVectorizer(decode_error='ignore', max_df = self.hyperparams['max_df'], min_df = 0)
            bow = self.bow.fit_transform(map(self._get_ngrams, concat_cols.ravel()))

            print("[WARNING] Setting min_df to 0 to avoid ValueError")

        # choose between CorEx and the TfIdf matrix
        if bow.shape[1] > self.hyperparams['threshold']:
            # use CorEx
            self.latent_factors = self.model.fit_transform(bow)
        else:
            # just use the bag of words representation
            self.latent_factors = pd.DataFrame(bow.todense())

        self.fitted = True

        return CallResult(None, True, 1)

    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: 
        # if corex didn't run for any reason, just return the given dataset
        if self.do_nothing:
            return CallResult(inputs, True, 1)

        inputs = self._process_files(inputs)

        if iterations is not None:
            self.max_iter = iterations
        else:
            self.max_iter = 250
        self.model.max_iter = self.max_iter

        # concatenate the columns row-wise
        concat_cols = None
        for column_index in self.text_columns:
            if concat_cols is not None:
                concat_cols = concat_cols.str.cat(inputs.iloc[:,column_index], sep = " ")
            else:
                concat_cols = copy.deepcopy(inputs.iloc[:,column_index])
        bow = self.bow.transform(map(self._get_ngrams, concat_cols.ravel()))

        # choose between CorEx and the TfIdf matrix
        if bow.shape[1] > self.hyperparams['threshold']:
            # use CorEx
            self.latent_factors = self.model.transform(bow).astype(float)
        else:
            # just use the bag of words representation
            self.latent_factors = pd.DataFrame(bow.todense())
        # make the columns corex adds distinguishable from other columns
        

        # remove the selected columns from input and add the latent factors given by corex
        out_df = d3m_DataFrame(inputs, generate_metadata = True)
        
        self.latent_factors.columns = [str(out_df.shape[-1] + i) for i in range(self.latent_factors.shape[-1])]

        # create metadata for the corex columns
        corex_df = d3m_DataFrame(self.latent_factors, generate_metadata = True)
        for column_index in range(corex_df.shape[1]):
            col_dict = dict(corex_df.metadata.query((ALL_ELEMENTS, column_index)))
            col_dict['structural_type'] = type(1.0)
            # FIXME: assume we apply corex only once per template, otherwise column names might duplicate
            col_dict['name'] = 'corex_' + str(out_df.shape[1] + column_index)
            col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')

            corex_df.metadata = corex_df.metadata.update((ALL_ELEMENTS, column_index), col_dict)
       
        
        # concatenate is --VERY-- slow without this next line
        corex_df.index = out_df.index.copy()

        out_df = utils.append_columns(out_df, corex_df)

        # remove the initial text columns from the df, if we do this before CorEx we can get an empty dataset error
        out_df = utils.remove_columns(out_df, self.text_columns)

        # TO DO : Incorporate timeout, max_iter
        # return CallResult(d3m_DataFrame(self.latent_factors))
        return CallResult(out_df, True, 1)

    #def fit_multi_produce(self, ):

    def _get_ngrams(self, text : str = None) -> str:
        punctuation_table = str.maketrans(dict.fromkeys(string.punctuation))
        try:
            words = text.translate(punctuation_table).lower().rsplit(" ")
        except:
            words = text.str.translate(punctuation_table).lower().rsplit(" ")

        new_text = ""
        for i in range(len(words)):
            new_text += "".join(words[i : i+int(self.hyperparams['n_grams'])]) + " "

        return new_text

    # remove the FileName columns from the data frame and replace them with text
    def _process_files(self, inputs: Input):
        fn_attributes = DataMetadata.list_columns_with_semantic_types(self=inputs.metadata, \
            semantic_types=["https://metadata.datadrivendiscovery.org/types/FileName"])
        all_attributes = DataMetadata.list_columns_with_semantic_types(self=inputs.metadata, \
            semantic_types=["https://metadata.datadrivendiscovery.org/types/Attribute"])
        fn_columns = list(set(all_attributes).intersection(fn_attributes))

        # if no file name columns are detected, default to regular behavior
        if len(fn_columns) == 0:
            return inputs

        # create an empty DataFrame of the required size
        processed_cols = pd.DataFrame("", index = copy.deepcopy(inputs.index), \
            columns = ['text_files_' + str(i) for i in range(len(fn_columns))])

        # for column_index in range(len(fn_columns)):
        for column_index in fn_columns:
            curr_column = copy.deepcopy(inputs.iloc[:, column_index])

            file_loc = inputs.metadata.query((ALL_ELEMENTS, column_index))['location_base_uris']
            file_loc = file_loc[0]  # take the first elem of the tuple
            file_loc = file_loc[7:] # get rid of 'file://' prefix

            for row_index in range(curr_column.shape[0]):
                text_file = curr_column.iloc[row_index]
                file_path = file_loc + text_file

                with open(file_path, 'rb') as file:
                    doc = file.read()
                doc = "".join(map(chr, doc))
                doc_tokens = re.compile(r"(?u)\b\w\w+\b").findall(doc) # list of strings

                processed_cols.iloc[row_index, fn_columns.index(column_index)] = " ".join(doc_tokens)

        # construct metadata for the newly generated columns
        processed_cols = d3m_DataFrame(processed_cols, generate_metadata = True)

        for column_index in range(processed_cols.shape[1]):
            col_dict = dict(processed_cols.metadata.query((ALL_ELEMENTS, column_index)))
            col_dict['structural_type'] = type("text")
            # FIXME: assume we apply corex only once per template, otherwise column names might duplicate
            col_dict['name'] = 'processed_file_' + str(inputs.shape[1] + column_index)
            col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')

            processed_cols.metadata = processed_cols.metadata.update((ALL_ELEMENTS, column_index), col_dict)

        # concatenate the input with the newly created columns
        updated_inputs = utils.append_columns(inputs, processed_cols)

        # remove the initial FileName columns from the df, if we do this before concatenating we might get an empty dataset error
        updated_inputs = utils.remove_columns(updated_inputs, fn_columns)

        return updated_inputs


    def get_params(self) -> CorexText_Params:
        return CorexText_Params(
            fitted_ = self.fitted,
            model_ = self.model, 
            bow_ = self.bow, 
            do_nothing_ = self.do_nothing,
            text_columns_ = self.text_columns,
            latent_factors_ = self.latent_factors,
            max_iter_ = self.max_iter
            )

    def set_params(self, *, params: CorexText_Params) -> None:
        self.fitted = params['fitted_']
        self.model = params['model_']
        self.bow = params['bow_']
        self.do_nothing = params['do_nothing_']
        self.text_columns = params['text_columns_']
        self.latent_factors = params['latent_factors_']
        self.max_iter = params['max_iter_']

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
