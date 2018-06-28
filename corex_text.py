import git
import os
import re
import sys
import typing

import pandas as pd
import numpy as np

from corextext.corex_topic import Corex
from collections import defaultdict, OrderedDict
from scipy import sparse as sp

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

import config as cfg_
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Input = container.DataFrame
Output = container.DataFrame

class CorexText_Params(params.Params):
    model_: typing.Union[Corex, None]
    bow_: typing.Union[TfidfVectorizer, None]

# Set hyperparameters according to https://gitlab.com/datadrivendiscovery/d3m#hyper-parameters
class CorexText_Hyperparams(hyperparams.Hyperparams):
    # number of Corex latent factors
    n_hidden = Uniform(lower = 0, upper = 100, default = 10, q = 1, description = 'number of topics', semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # max_df @ http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    max_df = Uniform(lower = 0.0, upper = 1.00, default = .9, q = .05, description = 'max percent document frequency of analysed terms', semantic_types=["http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # min_df @ http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    min_df = Uniform(lower = 0.0, upper = 1.00, default = .02, q = .01, description = 'min percent document frequency of analysed terms', semantic_types=["http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # max_features @ http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    max_features = Union(OrderedDict([('none', Enumeration([None], default = None)), ('int_mf', UniformInt(lower = 1000, upper = 50000, default = 50000, upper_inclusive=True, description = 'max number of terms to use'))]), default = 'none', semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    chunking = Uniform(lower = 0, upper = 2000, default = 0, q = 100, description = 'number of tfidf-filtered terms to include as a document, 0 => no chunking.  last chunk may be > param value to avoid small documents', semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter'])


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
         
    def fit(self, *, timeout : float = None, iterations : int = None) -> None:
        # don't fit twice
        if self.fitted:
            return

        print(self.hyperparams['max_df'], type(self.hyperparams['max_df']))
        print(self.hyperparams['min_df'], type(self.hyperparams['min_df']))


        self.model = Corex(n_hidden= self.hyperparams['n_hidden'], max_iter = iterations, seed = self.random_seed)
        self.bow = TfidfVectorizer(decode_error='ignore', max_df = self.hyperparams['max_df'], min_df = self.hyperparams['min_df'], max_features = self.hyperparams['max_features'])

        # set the number of iterations (for wrapper and underlying Corex model)
        if iterations is not None:
            self.max_iter = iterations
        else:
            self.max_iter = 250
        self.model.max_iter = self.max_iter

        column_name = training_data.columns[0]
        if self.hyperparams['chunking'] == 0:
            bow = self.bow.fit_transform(self.training_data[column_name].ravel())       
        else:
            inp, self.chunks = self._chunk(self.training_data[column_name].ravel())
            bow = self.bow.fit_transform(inp)
        
        self.latent_factors = self.model.fit_transform(bow)
        if self.hyperparams['chunking'] > 0:
            self.latent_factors = self._unchunk(self.latent_factors, self.chunks)

        self.fitted = True

        return


    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: 
        if iterations is not None:
            self.max_iter = iterations
            self.model.max_iter = self.max_iter
        else:
            self.max_iter = 250
            self.model.max_iter = self.max_iter

        column_name = inputs.columns[0]
        if self.hyperparams['chunking'] == 0:
            bow = self.bow.transform(inputs[column_name].ravel())
        else:
            inp, self.chunks = self._chunk(inputs[column_name].ravel())
            bow = self.bow.transform(inp)

        self.latent_factors = self.model.transform(bow).astype(float)
        if self.hyperparams['chunking'] > 0:
            self.latent_factors = self._unchunk(self.latent_factors, self.chunks)

        # TO DO : Incorporate timeout, max_iter
        return CallResult(d3m_DataFrame(self.latent_factors))
    
    def _chunk(self, inputs : np.ndarray = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        # chunk each string in the string list

        chunked_docs = []
        chunk_list = []
        overall_j = 0
        for i in range(inputs.shape[0]):
            doc_tokens = inputs[i]
            
            j = 0
            
            while (j+2)*self.hyperparams['chunking'] <= len(doc_tokens):
                new_chunked_str = " ".join(doc_tokens[j*self.hyperparams['chunking']:(j+1)*self.hyperparams['chunking']])
                chunked_docs.append(new_chunked_str)
                j = j + 1

            new_chunked_str = " ".join(doc_tokens[j*self.hyperparams['chunking']:])
            chunked_docs.append(new_chunked_str)
            overall_j += (j+1)
            chunk_list.append(overall_j)

        # all docs in 1 array, list indicating changepoints of documents
        return np.array(chunked_docs), np.array(chunk_list) 

    def _unchunk(self, transformed : np.ndarray,  chunk_array : np.ndarray):
        # transformed is samples x topics
        j = 0
        return_val = None
        # hacky?
        
        chunk_array = np.append(chunk_array, np.array([transformed.shape[0]-1]), axis = 0)
        
        temp = np.zeros((transformed.shape[1],)) 
        for i in range(transformed.shape[0]):
            if i < chunk_array[j] and i < transformed.shape[0]-1:
                temp = temp + transformed[i,:] 
            else:    
                divisor = (chunk_array[j] - chunk_array[j-1]+1) if j > 0 else chunk_array[j]
                
                temp = temp / float(divisor)
                temp = temp[np.newaxis, :] # 1 x features
                
                if return_val is None:
                    return_val = temp
                else:
                    return_val = np.concatenate([return_val, temp], axis = 0)

                j = j+1
                temp = np.zeros((transformed.shape[1],)) 

        return return_val

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

