import os
import copy
import numpy as np
import pandas as pd
#CUDA_VISIBLE_DEVICES = ""
import tensorflow as tf
from keras import backend as K
import keras.objectives
import keras
import keras.layers
from keras.layers import Concatenate, Dense, Input, merge
from keras.layers import Activation, BatchNormalization, Lambda, Reshape
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler
#from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.engine.topology import Layer
from keras.utils import to_categorical
import tempfile
from sklearn import preprocessing

import git
from d3m import utils
from d3m.container import DataFrame as d3m_DataFrame
import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata import base as metadata_base
import common_primitives.utils as common_utils
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple, NamedTuple
import typing

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
#from d3m.primitive_interfaces.params import Params
from d3m.metadata.hyperparams import Uniform, UniformInt, Union, Enumeration, UniformBool


import config as cfg_

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Input = container.DataFrame
Output = container.DataFrame

class EchoSAEc_Params(params.Params):
    model: typing.Union[keras.models.Model, None]
    model_weights: typing.Union[Any,None]
    fitted: typing.Union[bool, None] #bool
    label_encode: typing.Union[preprocessing.LabelEncoder, None]
    output_columns: typing.Union[list, pd.Index, None]
    #max_discrete_labels: int
    # add support for resuming training / storing model information

class EchoSAEr_Params(params.Params):
    model: typing.Union[keras.models.Model, None]
    model_weights: typing.Union[Any,None]
    fitted: typing.Union[bool, None] #bool
    output_columns: typing.Union[list, pd.Index, None]

class EchoSAE_Hyperparams(hyperparams.Hyperparams):
    n_hidden = Uniform(lower = 1, upper = 50, default = 10, q = 1, description = 'number of hidden factors learned', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    beta = Uniform(lower = 0, upper = 1000, default = .1, q = .01, 
    	description = 'Lagrange multiplier for beta (applied to regularizer I(X:Z)): defining tradeoff btwn label relevance : compression.', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    epochs = Uniform(lower = 1, upper = 1000, default = 100, description = 'number of epochs to train', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    convolutional = UniformBool(default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="whether to use a convolutional architecture"
    )


    error_on_no_input = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False."
    )
    gpus = Uniform(lower = 0, upper = 5, q = 1, 
                   default = 0,
                   semantic_types = ['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
                   description = 'GPUs to Use'
    )
                

def build_convolutional_encoder(n_hidden, sq_dim = None, architecture = 'alemi', encoder_layers = [], decoder_layers = []):
    x = keras.layers.Input(shape = self.training_inputs.shape[1:])
    t = x
    if sq_dim is None:
        sq_dim = int(np.sqrt(self.training_inputs.shape[-1]))
    if not len(self.training_inputs.shape) > 2:
        reshp = keras.layers.Reshape(inp_shape, input_shape = (sq_dim, sq_dim))(x)
  
    if architecture == 'alemi':
        el = [32, 32, 64, 64, 256, n_hidden]
        dl = [64, 64, 64, 32, 32, 32, 1]
    else:
        el = encoder_layers if encoder_layers else [32, 32, 64, 64, 256, n_hidden]
        dl = decoder_layers if decoder_layers else [64, 64, 64, 32, 32, 32, 1]


    if architecture == 'alemi':
        # works for 28 by 28 only
        h = keras.layers.Conv2D(el[0], activation = 'relu', kernel_size = 5, strides = 1, padding = 'same')(reshp)
        h = keras.layers.Conv2D(el[1], activation = 'relu', kernel_size = 5, strides = 2, padding = 'same')(h)
        h = keras.layers.Conv2D(el[2], activation = 'relu', kernel_size = 5, strides = 1, padding = 'same')(h)
        h = keras.layers.Conv2D(el[3], activation = 'relu', kernel_size = 5, strides = 2, padding = 'same')(h)
        h = keras.layers.Conv2D(el[4], activation = 'relu', kernel_size = 7, strides = 2, padding = 'valid')(h)

    #for i in range(len(self._latent_dims[:-1])):
    #    t = Dense(self._latent_dims[i], activation = self._activation)(t)
   
    # if self._noise == 'add' or self._noise == 'vae':
    #     z_mean_act = 'linear'
    #     z_var_act = 'linear'
    #     sample_function = vae_sample
    #     latent_loss = gaussian_kl_prior
    # elif self._noise == 'ido' or self._noise == 'mult':
    #     #final_enc_act = 'softplus'                                                                                                                                           
    #     z_mean_act = 'linear'
    #     z_var_act = 'linear'
    #     sample_function = ido_sample
    #     latent_loss = gaussian_kl_prior
    # elif self._noise == 'echo':
    #     z_mean_act = tanh64
    #     z_var_act = tf.math.log_sigmoid
    #     sample_function = echo_sample
    #     latent_loss = echo_loss
    #else:
    z_mean_act = tanh64
    z_var_act = tf.math.log_sigmoid
    sample_function = echo_sample
    latent_loss = echo_loss

    fx = Dense(el[-1], activation = z_mean_act)(h)
    sx = Dense(el[-1], activation = z_var_act)(h)
    z_act = Lambda(sample_function, name = 'latent_act')([fx,sx])
    return keras.models.Model(inputs = x, outputs = z_act)


class EchoClassification(SupervisedLearnerPrimitiveBase[Input, Output, EchoSAEc_Params, EchoSAE_Hyperparams]):
    """
    Keras NN implementing the information bottleneck method with Echo Noise to calculate I(X:Z), where Z also trained to maximize I(X:Y) for label Y.  Control tradeoff using 'label_beta param'
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "6c95166f-434a-435d-a3d7-bce8d7238061",
        "version": "1.0.0",
        "name": "Echo",
        "description": "Autoencoder implementation of Information Bottleneck using Echo Noise",
        "python_path": "d3m.primitives.classification.corex_supervised.EchoIB",
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
        "hyperparams_to_tune": ["n_hidden", "beta", "epochs"]
    })



    def __init__(self, *, hyperparams : EchoSAE_Hyperparams) -> None: #, random_seed : int =  0, docker_containers: typing.Dict[str, DockerContainer] = None
        super().__init__(hyperparams = hyperparams) # random_seed = random_seed, docker_containers = docker_containers)

    def _extra_params(self, latent_dims = None, activation = None, lr = None, batch = None, epochs = None, noise = None):
        self._latent_dims = [200, 200, self.hyperparams['n_hidden']]
        self._decoder_dims = list(reversed(self._latent_dims[:-1]))
        
        # TRAINING ARGS... what to do?
        self._activation = 'softplus'
        self._lr = 0.0005
        self._optimizer = Adam(self._lr)
        self._batch = 10
        self._epochs = None # HYPERPARAM?
        self._noise = 'echo'
        self._anneal_sched = None
        try:
            self.label_encode = self.label_encode
        except:
            self.label_encode = None
        try:
            self.output_columns = self.output_columns#['Hall of Fame']
        except:
            pass

    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult[None]:
        make_keras_pickleable()
        # create keras architecture
        # TODO : Architecture / layers as input hyperparameter
        
        self._extra_params()

        if iterations is not None:
            self.hyperparams["epochs"] = iterations
            
        if self.hyperparams['convolutional']:
            encoder = build_convolutional_encoder(self.hyperparams['n_hidden'])
            z_act = encoder.outputs[0]
        else:
            x = keras.layers.Input(shape = (self.training_inputs.shape[-1],))
            t = x

            for i in range(len(self._latent_dims[:-1])):
                t = Dense(self._latent_dims[i], activation = self._activation)(t)
        
            if self._noise == 'add' or self._noise == 'vae':
                z_mean_act = 'linear'
                z_var_act = 'linear'
                sample_function = vae_sample
                latent_loss = gaussian_kl_prior
            elif self._noise == 'ido' or self._noise == 'mult':
                #final_enc_act = 'softplus'
                z_mean_act = 'linear'
                z_var_act = 'linear'
                sample_function = ido_sample
                latent_loss = gaussian_kl_prior 
            elif self._noise == 'echo':
                z_mean_act = tanh64
                z_var_act = tf.math.log_sigmoid
                sample_function = echo_sample
                latent_loss = echo_loss 
            else:
                z_mean_act = tanh64
                z_var_act = tf.math.log_sigmoid
                sample_function = echo_sample
                latent_loss = echo_loss


            z_mean = Dense(self._latent_dims[-1], activation = z_mean_act, name = 'z_mean')(t)
            z_noise = Dense(self._latent_dims[-1], activation = z_var_act, name = 'z_noise', bias_initializer = 'ones')(t)
            z_act = Lambda(echo_sample, arguments = {'batch': self._batch, 'd_max': self._batch, 'nomc': True, 'calc_log': True, 'plus_sx': True}, output_shape = (self._latent_dims[-1],), name = 'z_act')([z_mean, z_noise])

        t = z_act
        for i in range(len(self._decoder_dims)):
            t = Dense(self._decoder_dims[i], name = 'decoder_'+str(i), activation = self._activation)(t) 
        
        # CLASSIFICATION ONLY here
        #print("LABEL UNIQUE ", self._label_unique)
        label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
        #if self._label_unique == 2:
        #    y_pred = Dense(1, activation = 'sigmoid', name = 'y_pred')(t)
        #else:
        y_pred = Dense(self._label_unique, activation = label_act, name = 'y_pred')(t)

        #if self._input_types:
        #    pass
        #else:
        #    print("Purely Supervised Bottleneck")
            # no reconstruction layers
        
        outputs = []
        loss_functions = []
        loss_weights = []
        

        #beta = Beta(name = 'beta', beta = self.hyperparams["beta"])(x)

        outputs.append(y_pred)
        if label_act == 'softmax':
            loss_functions.append(keras.objectives.categorical_crossentropy)
        elif label_act == 'sigmoid':
            loss_functions.append(keras.objectives.binary_crossentropy)
        else: 
            loss_functions.append(keras.objectives.mean_squared_error)#mse
        loss_weights.append(1)

        loss_tensor = Lambda(latent_loss)([z_mean,z_noise])
        outputs.append(loss_tensor)
        loss_functions.append(dim_sum)
        loss_weights.append(self.hyperparams["beta"])


        self.model = keras.models.Model(inputs = x, outputs = outputs)
        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)


        # anneal? 
        if self._anneal_sched:
            raise NotImplementedError
        else:
            self.model.fit_generator(generator(self.training_inputs, self.training_outputs, target_len = len(outputs), batch = self._batch),
                                     steps_per_epoch=int(self.training_inputs.values.shape[0]/self._batch), epochs = self.hyperparams["epochs"])
            #self.model.fit(self.training_inputs, [self.training_outputs]*len(outputs), 
            #    shuffle = True, epochs = self.hyperparams["epochs"], batch_size = self._batch) # validation_data = [] early stopping?

        #Lambda(ido_sample)
        #Lambda(vae_sample, output_shape = (d,))([z_mean, z_var])
        self.fitted = True
        
        return CallResult(None, True, self.hyperparams["epochs"])

    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: # TAKES IN DF with index column
        self._extra_params()
        inp = self.model.input
        outputs = [layer.output for layer in self.model.layers if 'z_mean' in layer.name or 'z_noise' in layer.name]
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
        dec_inp = [layer.input for layer in self.model.layers if 'decoder_0' in layer.name][0]
        print()
        print("DEC INPUT ", dec_inp)
        print()
        preds = [layer.output for layer in self.model.layers if 'y_pred' in layer.name]
        pred_function = K.function([dec_inp, K.learning_phase()], [preds[0]])
        
        predictions = []
        for i in range(0, inputs.shape[0], self._batch):
            data = inputs.values[i:i+self._batch]
            z_stats = [func([data, 1.])[0] for func in functors]
            z_act = echo_sample(z_stats).eval(session=K.get_session())
            y_pred= pred_function([z_act, 1.])[0]#.eval(session=K.get_session())
            if i == 0:
                print("Y PRED ", y_pred)
            #if self._label_unique > 2:
            y_pred = np.argmax(y_pred, axis = -1)
            predictions.extend([y_pred[yp] for yp in range(y_pred.shape[0])])
            #int_df = d3m_DataFrame(y_pred,index = inputs.index[i:i+self._batch], columns = self.output_columns)
            #if i == 0:
            #    predictions = int_df
            #else:
            #    predictions.append(int_df)
            #    print("Predictions SHAPE ", predictions.shape)
        predictions = np.array(predictions)
        predictions = self.label_encode.inverse_transform(predictions)
        output = d3m_DataFrame(predictions, columns = self.output_columns, generate_metadata = True, source = self)
        #output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)
        #output.metadata = self._add_target_semantic_types(metadata=output.metadata, target_names=self.output_columns, source=self)

        print("INPUT COLUMNS ", inputs.columns)
        print(inputs.columns)
        self._training_indices = [c for c in inputs.columns if isinstance(c, str) and 'index' in c.lower()]
        print("Traning indices ", self._training_indices)
        outputs = common_utils.combine_columns(return_result='new', #self.hyperparams['return_result'],
                                               add_index_columns=True,#self.hyperparams['add_index_columns'],
                                               inputs=inputs, columns_list=[output], source=self, column_indices=self._training_indices)
        print("outputs index ", outputs.index)
        print("OUTPUT COLUMNS ", outputs.columns)
        #predictions = d3m_DataFrame(predictions, index = inputs.index.copy())# columns = self.output_columns)


        return CallResult(outputs, True, 0)
        #return CallResult(d3m_DataFrame(self.model.predict(inputs)), True, 0)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        self.training_inputs = inputs
        
        self.output_columns = outputs.columns

        self._label_unique = np.unique(outputs.values).shape[0]
        self.label_encode = preprocessing.LabelEncoder()
        self.training_outputs = to_categorical(self.label_encode.fit_transform(outputs), num_classes = np.unique(outputs.values).shape[0])
        #self.training_outputs = to_categorical(outputs, num_classes = np.unique(outputs.values).shape[0])
        self.fitted = False
        

        # DATA PROFILING? softmax categorical (encoded) X or labels Y 
        # binary data? np.logical_and(self.training_inputs >= 0, self.training_inputs )
        
        # CHECK unique values for determining discrete / continuous
        #self._input_types = []
        #self._label_unique = np.unique(outputs).shape[0]
        #self._label_unique = 1 if self._label_unique > self.max_discrete_labels else self._label_unique


    def get_params(self) -> EchoSAEc_Params:
        return EchoSAEc_Params(model = self.model, model_weights = self.model.get_weights(), fitted = self.fitted, label_encode = self.label_encode, output_columns = self.output_columns)#max_discrete_labels = self.max_discrete_labels)#args)

    def set_params(self, *, params: EchoSAEc_Params) -> None:
        #self.max_discrete_labels = params["max_discrete_labels"]
        self._extra_params()
        self.model = params['model']
        self.model.set_weights(params['model_weights'])
        self.fitted = params['fitted']
        self.label_encode = params['label_encode']
        self.output_columns = params['output_columns']

    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                            source: typing.Any,  target_names: List = None,) -> metadata_base.DataMetadata:
        for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  source=source)
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  source=source)
            if target_names:
                metadata = metadata.update((metadata_base.ALL_ELEMENTS, column_index), {
                    'name': target_names[column_index],
                }, source=source)
        return metadata


class EchoRegression(SupervisedLearnerPrimitiveBase[Input, Output, EchoSAEr_Params, EchoSAE_Hyperparams]):
    """
    Keras NN implementing the information bottleneck method with Echo Noise to calculate I(X:Z), where Z also trained to maximize I(X:Y) for label Y.  Control tradeoff using 'label_beta param'
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "6c95166f-434a-435d-a3d7-bce8d7238061",
        "version": "1.0.0",
        "name": "Echo",
        "description": "Autoencoder implementation of Information Bottleneck using Echo Noise",
        "python_path": "d3m.primitives.regression.corex_supervised.EchoIB",
        "original_python_path": "echo_sae.EchoRegression",
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
      "hyperparams_to_tune": ["n_hidden", "beta", "epochs"]
    })



    def __init__(self, *, hyperparams : EchoSAE_Hyperparams) -> None: #, random_seed : int =  0, docker_containers: typing.Dict[str, DockerContainer] = None
        super().__init__(hyperparams = hyperparams) # random_seed = random_seed, docker_containers = docker_containers)

    def _extra_params(self, latent_dims = None, activation = None, lr = None, batch = None, epochs = None, noise = None):
        self._latent_dims = [200, 200, self.hyperparams['n_hidden']]
        self._decoder_dims = list(reversed(self._latent_dims[:-1]))
        
        # TRAINING ARGS... what to do?
        self._activation = 'softplus'
        self._lr = 0.0005
        self._optimizer = Adam(self._lr)
        self._batch = 10
        self._epochs = None # HYPERPARAM?
        self._noise = 'echo'
        self._anneal_sched = None

        #self.output_columns = ['Hall of Fame']


    def fit(self, *, timeout : float = None, iterations : int = None) -> CallResult[None]:
        make_keras_pickleable()
        # create keras architecture
        # TODO : Architecture / layers as input hyperparameter
        
        self._extra_params()

        if iterations is not None:
            self.hyperparams["epochs"] = iterations


        #if self.hyperparams['convolutional']:
        #    encoder = build_convolutional_encoder(self.hyperparams['n_hidden'])
        #    z_act = encoder.outputs[0]
         
        x = keras.layers.Input(shape = (self.training_inputs.shape[-1],))
        t = x

        for i in range(len(self._latent_dims[:-1])):
            t = Dense(self._latent_dims[i], activation = self._activation)(t)
        
        if self._noise == 'add' or self._noise == 'vae':
            z_mean_act = 'linear'
            z_var_act = 'linear'
            sample_function = vae_sample
            latent_loss = gaussian_kl_prior
        elif self._noise == 'ido' or self._noise == 'mult':
            #final_enc_act = 'softplus'
            z_mean_act = 'linear'
            z_var_act = 'linear'
            sample_function = ido_sample
            latent_loss = gaussian_kl_prior 
        elif self._noise == 'echo':
            z_mean_act = tanh64
            z_var_act = tf.math.log_sigmoid
            sample_function = echo_sample
            latent_loss = echo_loss 
        else:
            z_mean_act = tanh64
            z_var_act = tf.math.log_sigmoid
            sample_function = echo_sample
            latent_loss = echo_loss


        z_mean = Dense(self._latent_dims[-1], activation = z_mean_act, name = 'z_mean')(t)
        z_noise = Dense(self._latent_dims[-1], activation = z_var_act, name = 'z_noise', bias_initializer = 'ones')(t)
        z_act = Lambda(echo_sample, arguments = {'batch': self._batch, 'd_max': self._batch, 'nomc': True, 'calc_log': True, 'plus_sx': True}, output_shape = (self._latent_dims[-1],), name = 'z_act')([z_mean, z_noise])

        t = z_act
        for i in range(len(self._decoder_dims)):
            t = Dense(self._decoder_dims[i], name = 'decoder_'+str(i), activation = self._activation)(t) 
        
        # CLASSIFICATION ONLY here
        
        label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
        if self._label_unique == 2:
            y_pred = Dense(1, activation = 'sigmoid', name = 'y_pred')(t)
        else:
            y_pred = Dense(self._label_unique, activation = label_act, name = 'y_pred')(t)

        #if self._input_types:
        #    pass
        #else:
        #    print("Purely Supervised Bottleneck")
            # no reconstruction layers
        
        outputs = []
        loss_functions = []
        loss_weights = []
        

        #beta = Beta(name = 'beta', beta = self.hyperparams["label_beta"])(x)

        outputs.append(y_pred)
        if label_act == 'softmax':
            loss_functions.append(keras.objectives.categorical_crossentropy)
        else: 
            loss_functions.append(keras.objectives.mean_squared_error)#mse
        loss_weights.append(1)

        loss_tensor = Lambda(latent_loss)([z_mean,z_noise])
        outputs.append(loss_tensor)
        loss_functions.append(dim_sum)
        loss_weights.append(self.hyperparams["beta"])


        self.model = keras.models.Model(inputs = x, outputs = outputs)
        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)

        self.model_weights = self.model.get_weights()
        # anneal? 
        if self._anneal_sched:
            raise NotImplementedError
        else:
            self.model.fit_generator(generator(self.training_inputs, self.training_outputs, target_len = len(outputs), batch = self._batch),
                                     steps_per_epoch=int(self.training_inputs.values.shape[0]/self._batch), epochs = self.hyperparams["epochs"])
            #self.model.fit(self.training_inputs, [self.training_outputs]*len(outputs), 
            #    shuffle = True, epochs = self.hyperparams["epochs"], batch_size = self._batch) # validation_data = [] early stopping?

        #Lambda(ido_sample)
        #Lambda(vae_sample, output_shape = (d,))([z_mean, z_var])
        self.fitted = True

        return CallResult(None, True, self.hyperparams["epochs"])

    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: # TAKES IN DF with index column
        self._extra_params()
        inp = self.model.input
        outputs = [layer.output for layer in self.model.layers if 'z_mean' in layer.name or 'z_noise' in layer.name]
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
        dec_inp = [layer.input for layer in self.model.layers if 'decoder_0' in layer.name][0]
      
        preds = [layer.output for layer in self.model.layers if 'y_pred' in layer.name]
        pred_function = K.function([dec_inp, K.learning_phase()], [preds[0]])
        
        predictions = []
        for i in range(0, inputs.shape[0], self._batch):
            data = inputs.values[i:i+self._batch]
            z_stats = [func([data, 1.])[0] for func in functors]
            z_act = echo_sample(z_stats).eval(session=K.get_session())
            y_pred= pred_function([z_act, 1.])[0]#.eval(session=K.get_session())
            
            
            y_pred = np.argmax(y_pred, axis = -1)
            predictions.extend([y_pred[yp] for yp in range(y_pred.shape[0])])
            #int_df = d3m_DataFrame(y_pred,index = inputs.index[i:i+self._batch], columns = self.output_columns)
            #if i == 0:
            #    predictions = int_df
            #else:
            #    predictions.append(int_df)
            #    print("Predictions SHAPE ", predictions.shape)
        predictions = np.array(predictions)

        predictions = d3m_DataFrame(predictions, index = inputs.index.copy(), columns = self.output_columns, generate_metadata = True)# columns = self.output_columns)






        predictions = np.array(predictions)
        #predictions = self.label_encode.inverse_transform(predictions)
        output = d3m_DataFrame(predictions, columns = self.output_columns, generate_metadata = True, source = self)
        #output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)                                                                                                       
        #output.metadata = self._add_target_semantic_types(metadata=output.metadata, target_names=self.output_columns, source=self)                                                                           

        self._training_indices = [c for c in inputs.columns if isinstance(c, str) and 'index' in c.lower()]

        outputs = common_utils.combine_columns(return_result='new', #self.hyperparams['return_result'],                                                                                                      
                                               add_index_columns=True,#self.hyperparams['add_index_columns'],                                                                                                
                                               inputs=inputs, columns_list=[output], source=self, column_indices=self._training_indices)

        return CallResult(outputs, True, 0)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        self.training_inputs = inputs
        self.output_columns = outputs.columns
        self._label_unique = np.unique(outputs.values).shape[0]
        self.training_outputs = to_categorical(outputs, num_classes = np.unique(outputs.values).shape[0])
        self.fitted = False
        

        # DATA PROFILING? softmax categorical (encoded) X or labels Y 
        # binary data? np.logical_and(self.training_inputs >= 0, self.training_inputs )
        
        # CHECK unique values for determining discrete / continuous
        #self._input_types = []
        #self._label_unique = np.unique(outputs).shape[0]
        #self._label_unique = 1 if self._label_unique > self.max_discrete_labels else self._label_unique


    def get_params(self) -> EchoSAEr_Params:
        return EchoSAEr_Params(model = self.model, model_weights = self.model.get_weights(), fitted = self.fitted, output_columns = self.output_columns)#max_discrete_labels = self.max_discrete_labels)#args)

    def set_params(self, *, params: EchoSAEr_Params) -> None:
        #self.max_discrete_labels = params["max_discrete_labels"]
        self._extra_params()
        self.model = params['model']
        self.model.set_weights(params['model_weights'])
        self.fitted = params['fitted']
        self.output_columns = params['output_columns']
    
    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                            source: typing.Any,  target_names: List = None,) -> metadata_base.DataMetadata:
        for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  source=source)
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  source=source)
            if target_names:
                metadata = metadata.update((metadata_base.ALL_ELEMENTS, column_index), {
                    'name': target_names[column_index],
                }, source=source)
        return metadata





def make_keras_pickleable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name, custom_objects = {'tanh64': tanh64, 'log_sigmoid': tf.math.log_sigmoid, 'dim_sum': dim_sum, 'echo_loss': echo_loss, 'tf': tf, 'permute_neighbor_indices': permute_neighbor_indices})
        self.__dict__ = model.__dict__


    #cls = Sequential
    #cls.__getstate__ = __getstate__
    #cls.__setstate__ = __setstate__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def generator(data, labels = None, target_len = 1, batch = 100, mode = 'train', unsupervised = False):
        samples_per_epoch = data.shape[0]
        number_of_batches = int(samples_per_epoch/batch)
        counter=0
        
        while 1:
            x_batch = np.array(data.values[batch*counter:batch*(counter+1)]).astype('float32')
            y_batch = x_batch if unsupervised or labels is None else np.array(labels[batch*counter:batch*(counter+1)]).astype('float32')
            counter += 1
            yield x_batch, [y_batch]*target_len

            #restart counter to yeild data in the next epoch as well                                                                                                          
            if counter >= number_of_batches:
                counter = 0
                sd = np.random.randint(data.shape[0])
                data.sample(frac = 1, random_state = sd)
                np.random.seed(sd)
                np.random.shuffle(labels)
                #labels.sample(frac =1, random_state = sd)
                

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


def dim_sum(true, tensor, keepdims = False):
    #print('DIMSUM TRUE ', _true)
    #print('DIMSUM Tensor ', tensor)
    return K.sum(tensor, axis = -1, keepdims = keepdims)

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


def permute_neighbor_indices(batch_size, d_max=-1, replace = True, pop = True):
      """Produce an index tensor that gives a permuted matrix of other samples in batch, per sample.
      Parameters
      ----------
      batch_size : int
          Number of samples in the batch.
      d_max : int
          The number of blocks, or the number of samples to generate per sample.
      """
      if d_max < 0:
          d_max = batch_size + d_max
      inds = []
      if not replace:
        for i in range(batch_size):
          sub_batch = list(range(batch_size))
          if pop:
            sub_batch.pop(i)
          shuffle(sub_batch)
          inds.append(list(enumerate(sub_batch[:d_max])))
        return inds
      else:
        for i in range(batch_size):
            inds.append(list(enumerate(np.random.choice(batch_size, size = d_max, replace = True))))
        return inds

def echo_sample(inputs, clip = None, per_sample = True, init = -5., d_max = 100, batch = 100, multiplicative = False, replace = True, fx_clip = None, plus_sx = True, nomc = True, pop = True, neg_log = False, noise = 'additive', trainable = True, periodic = False, return_noise = False, fx_act = None, sx_act = None, encoder = True, calc_log = True, echomc = False, fullmc = False):

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
                                                                                                                                                                                            
    try:
        batch = min(batch, z_mean.shape[0])
    except:
        pass


    if fx_clip is not None:
      z_mean = K.clip(z_mean, -fx_clip, fx_clip)
    else:
        fx_clip = 1

    if clip is None:
        clip = (2**(-23)/fx_clip)**(1/batch)

    if not calc_log:
      #if not neg_log:                                                                                                                                                                                     
      cap_param = clip*z_scale_echo #if per_sample else clip*K.sigmoid(z_scale_echo)                                                                                                                       
      cap_param = tf.where(tf.abs(cap_param) < K.epsilon(), K.epsilon()*tf.sign(cap_param), cap_param)
    else:
      cap_param = K.log(clip) + (-1*z_scale_echo if not plus_sx else z_scale_echo)
                                                                                            
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















