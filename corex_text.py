from sklearn import preprocessing
#import primitive
import sys
import os
#sys.path.append('corex_topic/')
from corextext.corextext.corex_topic import Corex
#import corex_topic.corex_topic as corex_text
from collections import defaultdict, OrderedDict
from scipy import sparse as sp
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
import git

#from d3m import utils
import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params
from d3m.metadata.base import PrimitiveMetadata

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
#from d3m.primitive_interfaces.params import Params
from d3m.metadata.hyperparams import Uniform, UniformInt, Union, Enumeration

from typing import NamedTuple, Optional, Sequence, Any, Tuple
import typing

import config as cfg_
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Input = container.DataFrame
# Output = container.ndarray
Output = container.DataFrame

class CorexText_Params(params.Params):
    model: typing.Union[Corex, None]
    bow: typing.Union[TfidfVectorizer, None]
    get_text: bool 
    data_path: str
    #fitted: bool = False
    #training_inputs: Input = None
    # add support for resuming training / storing model information


# Set hyperparameters according to https://gitlab.com/datadrivendiscovery/d3m#hyper-parameters
class CorexText_Hyperparams(hyperparams.Hyperparams):
    n_hidden = Uniform(lower = 0, upper = 100, default = 10, q = 1, description = 'number of topics', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    max_df = Uniform(lower = .10, upper = 1.01, default = .9, q = .05, description = 'max percent document frequency of analysed terms', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    min_df = Union(OrderedDict([('int df' , Uniform(lower = 1, upper = 20, default = 2, q = 1, description = 'min integer document frequency of analysed terms')),
            ('pct df' , Uniform(lower = 0, upper = .10, default = .02, q = .01, description = 'min percent document frequency of analysed terms'))]), 
            default = 'pct df', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    chunking = Uniform(lower = 0, upper = 2000, default = 0, q = 100, 
        description = 'number of tfidf-filtered terms to include as a document, 0 => no chunking.  last chunk may be > param value to avoid small documents', 
        semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    max_features = Union(OrderedDict([('none', Enumeration([None], default = None)), 
        ('int mf', Uniform(lower = 1000, upper = 50001, default = 50000, q = 1000, description = 'max number of terms to use'))]), 
        default = 'none', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])


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
            "contact": "mailto:brekelma@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_corex" ]
            },
          "installation": [ cfg_.INSTALLATION ]
                #{'type': 'PIP', 
                # 'package_uri': 'git+https://github.com/brekelma/dsbox_corex.git@7381c3ed2d41a8dbe96bbf267a915a0ec48ee397#egg=dsbox-corex'#'+ str(git.Repo(search_parent_directories = True).head.object.hexsha) + '#egg=dsbox-corex'#@'+str(utils.current_git_commit(os.path.dirname(__file__)))+'#egg=dsbox-corex'
                #}]
          ,
          "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM", "LATENT_DIRICHLET_ALLOCATION"],
          "primitive_family": "FEATURE_CONSTRUCTION",
          "hyperparams_to_tune": ["n_hidden", "chunking", "max_df", "min_df"]
        })
        #"preconditions": [],
    #      "effects": [],


    def __init__(self, *, hyperparams : CorexText_Hyperparams) -> None: #, random_seed : int =  0, docker_containers: typing.Dict[str, DockerContainer] = None)

        super().__init__(hyperparams = hyperparams)#, random_seed = random_seed, docker_containers = docker_containers)
        
         
    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult[None]: #X : Sequence[Input]): 
        #self.columns = list(X)
        #X_ = X[self.columns].values # useless if only desired columns are passed
        if self.fitted:
            return

        if not hasattr(self, 'model') or self.model is None:
            self.model = Corex(n_hidden= self.hyperparams['n_hidden'], max_iter = iterations, seed = self.random_seed)#, **kwargs)

        if not hasattr(self, 'training_inputs'):
            raise ValueError("Missing training data.")

        if not hasattr(self, 'get_text'):
            raise ValueError("Missing get_text parameter")
        else:
            if not self.get_text or self.hyperparams['chunking'] > 0:
                self.bow = TfidfVectorizer(input = 'content', decode_error='ignore', max_df = self.hyperparams['max_df'], min_df = self.hyperparams['min_df'], max_features = self.hyperparams['max_features'])
            else:
                self.bow = TfidfVectorizer(input = 'filename', max_df = self.hyperparams['max_df'], min_df = self.hyperparams['min_df'], max_features = self.hyperparams['max_features'])


        if iterations is not None:
            self.max_iter = iterations
            self.model.max_iter = self.max_iter
        else:
            self.max_iter = 250
            self.model.max_iter = self.max_iter

        if self.hyperparams['chunking'] == 0:
            bow = self.bow.fit_transform(self.training_inputs.values.ravel()) if not self.get_text else self.bow.fit_transform(self._get_raw_inputs())        
        else:
            inp, self.chunks = self._read_and_chunk(self.training_inputs.values.ravel(), read = self.get_text)
            bow = self.bow.fit_transform(inp)
            
        
        self.latent_factors = self.model.fit_transform(bow)

        if self.hyperparams['chunking'] > 0:
                self.latent_factors = self._unchunk(self.latent_factors, self.chunks)

        self.fitted = True
        return CallResult(None, True, self.max_iter)


    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: # TAKES IN DF with index column
        #self.columns = list(X)
        #X_ = X[self.columns].values # useless if only desired columns are passed
        if iterations is not None:
            self.max_iter = iterations
            self.model.max_iter = self.max_iter
        else:
            self.max_iter = 250
            self.model.max_iter = self.max_iter

        if not self.fitted:
            if self.hyperparams['chunking'] == 0:
                bow = self.bow.fit_transform(inputs.values.ravel()) if not self.get_text else self.bow.fit_transform(self._get_raw_inputs(inputs = inputs, data_path = self.data_path))
            else:
                inp, self.chunks = self._read_and_chunk(inputs.values.ravel(), data_path = self.data_path, read = self.get_text)
                bow = self.bow.fit_transform(inp)
            self.latent_factors = self.model.fit_transform(bow).astype(float)
            
            self.fitted = True
        else:
            if self.hyperparams['chunking'] == 0:
                bow = self.bow.transform(inputs.values.ravel()) if not self.get_text else self.bow.transform(self._get_raw_inputs(inputs = inputs, data_path = self.data_path))
            else:
                inp, self.chunks = self._read_and_chunk(inputs.values.ravel(), data_path = self.data_path, read = self.get_text)
                bow = self.bow.transform(inp)
                #print('bow shape ', bow.shape)

            self.latent_factors = self.model.transform(bow).astype(float)
            
        if self.hyperparams['chunking'] > 0:
            self.latent_factors = self._unchunk(self.latent_factors, self.chunks)

        
        # TO DO : Incorporate timeout, max_iter
        return CallResult(self.latent_factors, True, self.max_iter)

    def _fit_transform(self, inputs : Input, timeout : float = None, iterations : int = None) -> Sequence[Output]: # TAKES IN DF with index column
        #self.columns = list(X)
        #X_ = X[self.columns].values # useless if only desired columns are passed


        if iterations is not None:
            self.max_iter = iterations
            self.model.max_iter = self.max_iter

        if self.hyperparams['chunking'] == 0:
            bow = self.bow.fit_transform(inputs.values.ravel()) if not self.get_text else self.bow.fit_transform(self._get_raw_inputs(inputs = inputs))
        else:
            inp, self.chunks = self._read_and_chunk(inputs.values.ravel(), read = self.get_text)
            bow = self.bow.fit_transform(inp)  
        self.latent_factors = self.model.fit_transform(bow)
        
        if self.hyperparams['chunking'] > 0:
                self.latent_factors = self._unchunk(self.latent_factors, self.chunks)

        self.fitted = True
        return self.latent_factors

    def _get_raw_inputs(self, inputs : Input = None, data_path = None) -> np.ndarray:
        print_ = True
        raw_inputs = self.training_inputs.values if inputs is None else inputs.values
        inp = self.training_inputs.values if inputs is None else inputs.values
        if data_path is not None:
            for idx, val in np.ndenumerate(inp):
                raw_inputs[idx] = os.path.join(data_path, val)
        elif self.data_path is not None:
            for idx, val in np.ndenumerate(inp):
                raw_inputs[idx] = os.path.join(self.data_path, val)
        else:
            warn('Data_path param not passed.')
        
        return raw_inputs.ravel()
    
    def _read_and_chunk(self, inputs : Input = None, data_path : str = None, read : bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # read data into documents / text
        # chunk_array = np.zeros((inputs.shape[0],))
        chunked_docs = []
        chunk_list = []
        overall_j = 0
        for i in range(inputs.shape[0]):
            if read:
                if data_path is None:
                    file_path = os.path.join(self.data_path, inputs[i])
                else:
                    file_path = os.path.join(data_path, inputs[i])
                with open(file_path, 'rb') as fn:
                    doc = fn.read()
                doc = "".join(map(chr, doc))
                doc_tokens = re.compile(r"(?u)\b\w\w+\b").findall(doc) # list of strings
            else:
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
            #print('row ', i, '/', transformed.shape[0], ' chunk: ', j, ' chunk_array[j]: ', chunk_array[j])
            if i < chunk_array[j] and i < transformed.shape[0]-1:   
                #temp = np.maximum(temp, transformed[i,:])
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
                #print(i, return_val.shape)
        return return_val

    def set_training_data(self, *, inputs : Input, outputs : Output) -> None:
        self.training_inputs = inputs
        self.fitted = False

    def get_params(self) -> CorexText_Params:
        return CorexText_Params(model = self.model, bow = self.bow, get_text = self.get_text, data_path = self.data_path)
                                #fitted = self.fitted, training_inputs = self.training_inputs)

    def set_params(self, *, params: CorexText_Params) -> None:
        self.model = params['model']
        self.bow = params['bow']
        self.get_text = params['get_text']
        self.data_path = params['data_path']
        #self.fitted = params.fitted
        #self.training_inputs = params.training_inputs

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

    def _get_feature_names(self):
        return ['CorexText_'+ str(i) for i in range(self.hyperparams['n_hidden'])]
