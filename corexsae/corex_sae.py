import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Concatenate, Input, Dense, merge
from keras.layers import Activation, BatchNormalization, Lambda, Reshape
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.engine.topology import Layer
from keras.utils import to_categorical

import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params
from d3m.metadata.base import PrimitiveMetadata

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
#from d3m.primitive_interfaces.params import Params
from d3m.metadata.hyperparams import Uniform, UniformInt, Union, Enumeration

from typing import NamedTuple, Optional, Sequence, Any
import typing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Input = container.ndarray #container.DataFrame
Output = container.ndarray #container.DataFrame

class CorexSAE_Params(params.Params):
    model: typing.Union[Model, None]
    max_discrete_labels: int
    # add support for resuming training / storing model information


class CorexSAE_Hyperparams(hyperparams.Hyperparams):
    label_beta = Uniform(lower = 0, upper = 1000, default = 1, q = .01, description = 'Lagrange multiplier for beta : 1 tradeoff btwn label relevance : compression.')
    epochs = Uniform(lower = 1, upper = 1000, default = 100, description = 'number of epochs to train')
    #n_hidden = Uniform(lower = 0, upper = 100, default = 10, q = 1, description = 'number of topics')
    #max_df = Uniform(lower = .10, upper = 1.01, default = .9, q = .05, description = 'max percent document frequency of analysed terms')
    #min_df = Union(OrderedDict([('int df' , Uniform(lower = 1, upper = 20, default = 2, q = 1, description = 'min integer document frequency of analysed terms')),
    #        ('pct df' , Uniform(lower = 0, upper = .10, default = .01, q = .01, description = 'min percent document frequency of analysed terms'))]), 
    #        default = 'int df')
    #max_features = Union(OrderedDict([('none', Enumeration([None], default = None)), 
    #            ('int mf', Uniform(lower = 1000, upper = 50001, default = 50000, q = 1000, description = 'max number of terms to use'))]),
    #            default = 'none')


class CorexSAE(SupervisedLearnerPrimitiveBase[Input, Output, CorexSAE_Params, CorexSAE_Hyperparams]):

    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "6c95166f-434a-435d-a3d7-bce8d7238061",
        "version": "1.0.0",
        "name": "CorexSupervised",
        "description": "Autoencoder implementation of Corex / Information Bottleneck",
        "python_path": "d3m.primitives.dsbox.CorexSupervised",
        "original_python_path": "corexsae.corex_sae.CorexSAE",
        "source": {
            "name": "ISI",
            "contact": "mailto:brekelma@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_corex" ]
            },
        # git+https://github.com/brekelma/corex_continuous#egg=corex_continuous
        "installation": [
            {'type': 'PIP', 'package_uri': 'git+https://github.com/brekelma/dsbox_corex.git@8672da14a7f2e00ea488da460ad68ef0799a9532#egg=dsbox-corex'}
            ],
      "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM"],
      "primitive_family": "CLASSIFICATION", #"FEATURE_CONSTRUCTION",
      "hyperparams_to_tune": ["label_beta", "epochs"]
    })



    def __init__(self, *, hyperparams : CorexSAE_Hyperparams) -> None: #, random_seed : int =  0, docker_containers: typing.Dict[str, DockerContainer] = None
        super().__init__(hyperparams = hyperparams) # random_seed = random_seed, docker_containers = docker_containers)

    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult[None]:

        # create keras architecture
        self._latent_dims = [100, 100, 20]
        self._decoder_dims = list(reversed(self.latent_dims[:-1]))
        
        # TRAINING ARGS... what to do?
        self._activation = 'softplus'
        self._optimizer = Adam(.001)
        self._batch = 100
        self._epochs = None # HYPERPARAM?
        self._noise = 'add'
        self._anneal_sched = None
        
        if iterations is not None:
            self.Hyperparams["epochs"] = iterations

        x = Input(shape = (self.training_inputs.shape[-1],))
        t = x

        for i in range(len(self.latent_dims[:-1])):
            t = Dense(self.latent_dims[i], activation = self.activation)(t)
        
        if self._noise == 'add' or self._noise == 'vae':
            final_enc_act = 'linear'
            sample_function = vae_sample
        else:
            #final_enc_act = 'softplus'
            final_enc_act = 'linear'
            sample_function = ido_sample

        z_mean = Dense(self.latent_dims[:-1], activation = final_enc_act, name = 'z_mean')(t)
        z_noise = Dense(self.latent_dims[:-1], activation = final_enc_act, name = 'z_noise')(t)
        z_act = Lambda(vae_sample, output_shape = (self.latent_dims[:-1],))([z_mean, z_var])

        t = z_act
        for i in range(len(self._decoder_dims)):
            t = Dense(self._decoder_dims[i], activation = self.activation)(t) 
        
        label_act = 'softmax' if self._label_unique > 1 else 'linear'
        y_pred = Dense(self._label_unique, activation = 'softmax', name = 'y_pred')

        if self._input_types:
            pass
        else:
            print("Purely Supervised Bottleneck")
            # no reconstruction layers
        
        outputs = []
        loss_functions = []
        loss_weights = []
        

        beta = Beta(name = 'beta', beta = self.Hyperparams["label_beta"])(x)

        outputs.append(y_pred)
        if label_act == 'softmax':
            loss_functions.append(objectives.categorical_crossentropy)
        else: 
            loss_functions.append(objectives.mean_squared_error)#mse
        loss_weights.append(beta)

        self.model = Model(inputs = x, outputs = outputs)
        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)


        # anneal? 
        if self._anneal_sched:
            raise NotImplementedError
        else:
            self.model.fit(self.training_inputs, [self.training_outputs]*len(outputs), 
                shuffle = True, epochs = self.Hyperparams["epochs"], batch_size = self._batch_size) # validation_data = [] early stopping?

        #Lambda(ido_sample)
        #Lambda(vae_sample, output_shape = (d,))([z_mean, z_var])

        return CallResult(None, True, self.Hyperparams["epochs"])

    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: # TAKES IN DF with index column
        return CallResult(self.model.predict(inputs), True, 0)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        self.training_inputs = inputs
        self.training_outputs = to_categorical(outputs, num_classes = np.unique(outputs).shape[0])
        self.fitted = False
        

        # DATA PROFILING? softmax categorical (encoded) X or labels Y 
        # binary data? np.logical_and(self.training_inputs >= 0, self.training_inputs )
        self._input_types = []
        self._label_unique = np.unique(outputs).shape[0]
        self._label_unique = 1 if self._label_unique > self.max_discrete_labels else self._label_unique


    def get_params(self) -> CorexSAE_Params:
        return CorexSAE_Params()#args)

    def set_params(self, *, params: CorexSAE_Params) -> None:
        self.max_discrete_labels = params["max_discrete_labels"]
        pass


def vae_sample(args):
    z_mean, z_noise = args
    std = 1.0
    K.random_normal(shape=(z_mean._keras_shape[1],),
                                  mean=0.,
                                  stddev=epsilon_std)
    return z_mean + K.exp(z_noise / 2) * epsilon
    #return z_mean + z_noise * epsilon

def ido_sample(args):
    z_mean, z_noise = args
    std = 1.0
    K.random_normal(shape=(z_mean._keras_shape[1],),
                                  mean=0.,
                                  stddev=epsilon_std)
    
    return K.exp(K.log(z_mean) + K.exp(z_noise / 2) * epsilon)
    #return K.exp(K.log(z_mean) + z_noise * epsilon)

class Beta(Layer):

    def __init__(self, shape = 1, beta = None, trainable = False, **kwargs):
        self.shape = shape
        self.trainable = trainable
        #self.n_dims = n_dims
        if beta is not None:
          self.set_beta(beta)
        else:
          self.set_beta(1.0)
        #self.output_dim = output_dim
        super(Beta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.dim = input_shape[1]
        if self.trainable:
            self.betas = self.add_weight(name='beta', 
                                          shape = (self.shape,),
                                          initializer= Constant(value = self.beta),
                                          trainable= True)
        else:
            self.betas = self.add_weight(name='beta', 
                          shape = (self.shape,),
                          initializer= Constant(value = self.beta),
                          trainable= False)

        super(Beta, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
          return K.repeat_elements(K.expand_dims(self.betas,1), 1, -1)

    #not used externally
    def set_beta(self, beta):
        self.beta = beta

    def compute_output_shape(self, input_shape):
        return (1, 1)















