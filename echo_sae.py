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

import git
from d3m import utils
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
import config as cfg_
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Input = container.ndarray #container.DataFrame
Output = container.ndarray #container.DataFrame

class EchoSAE_Params(params.Params):
    model: typing.Union[Model, None]
    #max_discrete_labels: int
    # add support for resuming training / storing model information


class EchoSAE_Hyperparams(hyperparams.Hyperparams):
    label_beta = Uniform(lower = 0, upper = 1000, default = 1, q = .01, 
    	description = 'Lagrange multiplier for beta : 1 tradeoff btwn label relevance : compression.', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    epochs = Uniform(lower = 1, upper = 1000, default = 100, description = 'number of epochs to train', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])


    #n_hidden = Uniform(lower = 0, upper = 100, default = 10, q = 1, description = 'number of topics')
    #max_df = Uniform(lower = .10, upper = 1.01, default = .9, q = .05, description = 'max percent document frequency of analysed terms')
    #min_df = Union(OrderedDict([('int df' , Uniform(lower = 1, upper = 20, default = 2, q = 1, description = 'min integer document frequency of analysed terms')),
    #        ('pct df' , Uniform(lower = 0, upper = .10, default = .01, q = .01, description = 'min percent document frequency of analysed terms'))]), 
    #        default = 'int df')
    #max_features = Union(OrderedDict([('none', Enumeration([None], default = None)), 
    #            ('int mf', Uniform(lower = 1000, upper = 50001, default = 50000, q = 1000, description = 'max number of terms to use'))]),
    #            default = 'none')


class EchoClassification(SupervisedLearnerPrimitiveBase[Input, Output, EchoSAE_Params, EchoSAE_Hyperparams]):

    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "6c95166f-434a-435d-a3d7-bce8d7238061",
        "version": "1.0.0",
        "name": "EchoClassification",
        "description": "Autoencoder implementation of Information Bottleneck using Echo Noise",
        "python_path": "d3m.primitives.classification.echo.EchoClassification",
        "original_python_path": "echo_sae.EchoClassification",
        "source": {
            "name": "ISI",
            "contact": "mailto:brekelma@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_corex" ]
            },
        # git+https://github.com/brekelma/corex_continuous#egg=corex_continuous
        "installation": [ cfg_.INSTALLATION ]
            #{'type': 'PIP', 
             #'package_uri': 'git+https://github.com/brekelma/dsbox_corex.git@7381c3ed2d41a8dbe96bbf267a915a0ec48ee397#egg=dsbox-corex'#'+ str(git.Repo(search_parent_directories = True).head.object.hexsha) + '#egg=dsbox-corex'
            #}
            #]
            ,
      "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM"],
      "primitive_family": "CLASSIFICATION",
      "hyperparams_to_tune": ["label_beta", "epochs"]
    })



    def __init__(self, *, hyperparams : CorexSAE_Hyperparams) -> None: #, random_seed : int =  0, docker_containers: typing.Dict[str, DockerContainer] = None
        super().__init__(hyperparams = hyperparams) # random_seed = random_seed, docker_containers = docker_containers)

    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult[None]:

        # create keras architecture
        # TODO : Architecture / layers as input hyperparameter
        self._latent_dims = [100, 100, 20]
        self._decoder_dims = list(reversed(self.latent_dims[:-1]))
        
        # TRAINING ARGS... what to do?
        self._activation = 'softplus'
        self._lr = 0.0005
        self._optimizer = Adam(lr)
        self._batch = 100
        self._epochs = None # HYPERPARAM?
        self._noise = 'echo'
        self._anneal_sched = None
        
        if iterations is not None:
            self.hyperparams["epochs"] = iterations

        x = Input(shape = (self.training_inputs.shape[-1],))
        t = x

        for i in range(len(self.latent_dims[:-1])):
            t = Dense(self.latent_dims[i], activation = self.activation)(t)
        
        if self._noise == 'add' or self._noise == 'vae':
            final_enc_act = 'linear'
            sample_function = vae_sample
            latent_loss = gaussian_kl_prior
        elif self.:
            #final_enc_act = 'softplus'
            final_enc_act = 'linear'
            sample_function = ido_sample
            latent_loss = gaussian_kl_prior 
        elif self._noise == 'echo':
            final_enc_act = tanh64
            sample_function = echo_sample
            latent_loss = echo_loss 
        else:
            final_enc_act = tanh64
            sample_function = echo_sample
            latent_loss = echo_loss


        z_mean = Dense(self.latent_dims[:-1], activation = final_enc_act, name = 'z_mean')(t)
        z_noise = Dense(self.latent_dims[:-1], activation = final_enc_act, name = 'z_noise')(t)
        z_act = Lambda(echo_sample, output_shape = (self.latent_dims[:-1],))([z_mean, z_var])


        t = z_act
        for i in range(len(self._decoder_dims)):
            t = Dense(self._decoder_dims[i], activation = self.activation)(t) 
        
        # CLASSIFICATION ONLY here
        label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
        y_pred = Dense(self._label_unique, activation = 'softmax', name = 'y_pred')

        if self._input_types:
            pass
        else:
            print("Purely Supervised Bottleneck")
            # no reconstruction layers
        
        outputs = []
        loss_functions = []
        loss_weights = []
        

        #beta = Beta(name = 'beta', beta = self.hyperparams["label_beta"])(x)

        outputs.append(y_pred)
        if label_act == 'softmax':
            loss_functions.append(objectives.categorical_crossentropy)
        else: 
            loss_functions.append(objectives.mean_squared_error)#mse
        loss_weights.append(self.hyperparams["label_beta"])

        outputs.append([z_mean, z_noise])
        loss_functions.append(latent_loss_function)
        loss_weights.append(1)


        self.model = Model(inputs = x, outputs = outputs)
        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)


        # anneal? 
        if self._anneal_sched:
            raise NotImplementedError
        else:
            self.model.fit(self.training_inputs, [self.training_outputs]*len(outputs), 
                shuffle = True, epochs = self.hyperparams["epochs"], batch_size = self._batch_size) # validation_data = [] early stopping?

        #Lambda(ido_sample)
        #Lambda(vae_sample, output_shape = (d,))([z_mean, z_var])

        return CallResult(None, True, self.hyperparams["epochs"])

    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: # TAKES IN DF with index column
        return CallResult(self.model.predict(inputs), True, 0)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        self.training_inputs = inputs
        self._label_unique = np.unique(outputs).shape[0]
        self.training_outputs = to_categorical(outputs, num_classes = np.unique(outputs).shape[0])
        self.fitted = False
        

        # DATA PROFILING? softmax categorical (encoded) X or labels Y 
        # binary data? np.logical_and(self.training_inputs >= 0, self.training_inputs )
        
        # CHECK unique values for determining discrete / continuous
        #self._input_types = []
        #self._label_unique = np.unique(outputs).shape[0]
        #self._label_unique = 1 if self._label_unique > self.max_discrete_labels else self._label_unique


    def get_params(self) -> CorexSAE_Params:
        return CorexSAE_Params(model = self.model, max_discrete_labels = self.max_discrete_labels)#args)

    def set_params(self, *, params: CorexSAE_Params) -> None:
        self.max_discrete_labels = params["max_discrete_labels"]
        self.model = params['model']
        pass



def tanh64(x):
        y = 64
        return (K.exp(1.0/y*x)-K.exp(-1.0/y*x))/(K.exp(1.0/y*x)+K.exp(-1.0/y*x)+K.epsilon())

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

def gaussian_prior_kl(inputs):
    [mu1, logvar1] = inputs
    mu2 = K.variable(0.0)
    logvar2 = K.variable(0.0)
    return gaussian_kl([mu1, logvar1, mu2, logvar2])

def echo_loss(inputs, d_max = 50, clip= 0.85, binary_input = True, calc_log = True, plus_sx = True, neg_log = False, fx_clip = None):
    if isinstance(inputs, list):
        #cap_param = inputs[0]                                                                                                                                                                             
        cap_param = inputs[-1]
        # -1 for stat, 0 for addl                                                                                                                                                                          
    else:
        cap_param = inputs

    capacities = -K.log(K.abs(clip*cap_param)+K.epsilon()) if not calc_log else -(tf.log(clip) + (cap_param if plus_sx else -cap_param))
    return capacities



def echo_sample(inputs, clip = 0.85, per_sample = True, init = -5., d_max = 100, batch = 100, multiplicative = False, replace = True, fx_clip = None, plus_sx = True, nomc = False, pop = True,
                neg_log = False, noise = 'additive', trainable = True, periodic = False, return_noise = False, fx_act = None, sx_act = None, encoder = True, calc_log = True, echomc = False, fullmc = False):

    if isinstance(inputs, list):
      z_mean = inputs[0]
      z_scale_echo = inputs[-1]
      # TURN INTO LOG INPUTS?                                                                                                                                                                              
    else:
      z_mean = inputs
    
    try:
      shp = K.int_shape(z_mean)
    except:
      shp = z_mean.shape

    if not encoder or shp[-1] == 1:
      orig_shape = shp
                                                                                                                                                                                            

    if fx_clip is not None:
      z_mean = K.clip(z_mean, -fx_clip, fx_clip)

    if not calc_log:
      #if not neg_log:                                                                                                                                                                                     
      cap_param = clip*z_scale_echo #if per_sample else clip*K.sigmoid(z_scale_echo)                                                                                                                       
      cap_param = tf.where(tf.abs(cap_param) < K.epsilon(), K.epsilon()*tf.sign(cap_param), cap_param)
    else:
      cap_param = tf.log(clip) + (-1*z_scale_echo if not plus_sx else z_scale_echo)
                                                                                            
    inds = permute_neighbor_indices(batch, d_max, replace = replace, pop = pop)

    if echomc:
      z_mean = z_mean - K.mean(z_mean, axis = 0, keepdims = True)

                                                                                                                                                               
    c_z_stack = tf.stack([cap_param for k in range(d_max)])  # no phase                                                                                                                                    
                                                                                                                
    f_z_stack = tf.stack([z_mean for k in range(d_max)])  # no phase                                                                                                                                       

    stack_dmax = tf.gather_nd(c_z_stack, inds)
    stack_zmean = tf.gather_nd(f_z_stack, inds)


    ax = 1
    #stack_dmax= tf.where(tf.abs(stack_dmax) < K.epsilon(), K.epsilon()*tf.sign(stack_dmax), stack_dmax)                                                                                                   
    # cum product over dmax axis. exclusive => element 0 =                                                                                                                                                 
    if calc_log:
      noise_sx_product = tf.cumsum(stack_dmax, axis = ax, exclusive = True)
    else:
      noise_sx_product = tf.cumprod(stack_dmax, aaxis = ax, exclusive = True)# if not noise == 'rescaled' else False)
        noise_sx_product = tf.exp(noise_sx_product) if calc_log else noise_sx_product
    noise_times_sample = tf.multiply(stack_zmean, noise_sx_product)

    noise_tensor = tf.reduce_sum(noise_times_sample, axis = ax)

    if return_noise and nomc:
      return noise_tensor
                                                                                                     
    if not nomc:
      noise_tensor = noise_tensor - K.mean(noise_tensor, axis=0, keepdims = True) # 0 mean noise : ends up being 1 x m                                                                                     
                                                                                                                                                         

                                                                                                                                                                                                
    sx = cap_param if not calc_log else tf.exp(cap_param)
    full_sx_noise = tf.multiply(sx, noise_tensor)  # batch x Am                                                                                                                                            
    if fullmc:
      full_sx_noise = full_sx_noise - K.mean(full_sx_noise, axis = 0, keepdims = True)
    noisy_encoder = z_mean + full_sx_noise # tf.multiply(sx, noise_tensor)  # batch x Am                                                                                                                   
                                                                                                                                          
    return noisy_encoder #if not return_noise else noise_tensor 

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















