import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import copy
import numpy as np
import pandas as pd
#CUDA_VISIBLE_DEVICES = ""

#import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras.backend import get_session
#from keras.backend import get_session
#import tensorflow.keras.objectives as keras_objectives
import tensorflow.keras as keras
import tensorflow.keras.layers as keras_layers
from tensorflow.keras.layers import Concatenate, Dense, Input # merge
from tensorflow.keras.layers import Activation, BatchNormalization, Lambda, Reshape
from tensorflow.keras.callbacks import Callback, TensorBoard, LearningRateScheduler
import tensorflow.keras.callbacks
#from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.utils import to_categorical
import tempfile
from sklearn import preprocessing

import git
#from common_primitives import utils
#from d3m import utils
from d3m.container import DataFrame as d3m_DataFrame
import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.base import ALL_ELEMENTS, DataMetadata
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

class EchoIB_Params(params.Params):
    model: typing.Union[keras.models.Model, None]
    model_weights: typing.Union[Any,None]
    fitted: typing.Union[bool, None] #bool
    label_encode: typing.Union[preprocessing.LabelEncoder, None]
    output_columns: typing.Union[list, pd.Index, None]
    #max_discrete_labels: int
    # add support for resuming training / storing model information

class EchoIB_Hyperparams(hyperparams.Hyperparams):
    n_hidden = UniformInt(lower = 1, upper = 401, default = 200, description = 'number of hidden factors learned', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    beta = Uniform(lower = 0, upper = 1000, default = .1, q = .01, 
        description = 'Lagrange multiplier for beta (applied to regularizer I(X:Z)): defining tradeoff btwn label relevance : compression.', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    epochs = UniformInt(lower = 1, upper = 10000, default = 100, description = 'number of epochs to train', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    batch = UniformInt(lower = 10, upper = 1000, default = 50, description = 'batch_size', semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    
    activation = Enumeration(values = ['relu', 'tanh', 'elu'], default = 'elu', 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="activation to use for intermediate activations"
    )

    convolutional = UniformBool(default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="whether to use a convolutional architecture"
    )

    task = Enumeration(values = ['CLASSIFICATION', 'REGRESSION'], default = 'REGRESSION', 
                       semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                       description='task type')

    use_as_modeling = UniformBool(default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="whether to return constructed features AND predictions (else, used for modeling i.e. only predictions"
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


class ZeroAnneal(Callback):
    def __init__(self, lw = 1, index = 0, epochs = 10, scaled = 0.0):#1):
        self.lw = tf.constant(lw, dtype = tf.float32)
        self.zero_epochs = epochs
        self.ind = index
        self.replace = scaled*lw
        
    def on_epoch_begin(self, epoch, logs={}):
        if epoch < self.zero_epochs:
            tf.assign(self.model.loss_weights[self.ind], tf.constant(self.replace, dtype = tf.float32)).eval(session=get_session())
        else:
            tf.assign(self.model.loss_weights[self.ind],self.lw).eval(session=get_session())


                

def build_convolutional_encoder(n_hidden, sq_dim = None, architecture = 'alemi', encoder_layers = [], decoder_layers = []):
    # for 28 x 28 image shapes
    x = keras_layers.Input(shape = (self.training_inputs.shape[1:],))
    t = x
    if sq_dim is None:
        sq_dim = int(np.sqrt(self.training_inputs.shape[-1]))
    if not len(self.training_inputs.shape) > 2:
        reshp = keras_layers.Reshape(inp_shape, input_shape = (sq_dim, sq_dim))(x)
  
    if architecture == 'alemi':
        el = [32, 32, 64, 64, 256, n_hidden]
        dl = [64, 64, 64, 32, 32, 32, 1]
    else:
        el = encoder_layers if encoder_layers else [32, 32, 64, 64, 256, n_hidden]
        dl = decoder_layers if decoder_layers else [64, 64, 64, 32, 32, 32, 1]


    if architecture == 'alemi':
        # works for 28 by 28 only
        h = keras_layers.Conv2D(el[0], activation = 'relu', kernel_size = 5, strides = 1, padding = 'same')(reshp)
        h = keras_layers.Conv2D(el[1], activation = 'relu', kernel_size = 5, strides = 2, padding = 'same')(h)
        h = keras_layers.Conv2D(el[2], activation = 'relu', kernel_size = 5, strides = 1, padding = 'same')(h)
        h = keras_layers.Conv2D(el[3], activation = 'relu', kernel_size = 5, strides = 2, padding = 'same')(h)
        h = keras_layers.Conv2D(el[4], activation = 'relu', kernel_size = 7, strides = 2, padding = 'valid')(h)
    #else:
    #    for i in range(len(self._latent_dims[:-1])):
    #        t = Dense(self._latent_dims[i], activation = self._activation)(t)
   
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
    z_act = Lambda(sample_function, name = 'latent_act', arguments = self._echo_args)([fx,sx])
    return keras.models.Model(inputs = x, outputs = z_act)


class EchoIB(SupervisedLearnerPrimitiveBase[Input, Output, EchoIB_Params, EchoIB_Hyperparams]):
    """
    Keras NN implementing the information bottleneck method with Echo Noise to calculate I(X:Z), where Z also trained to maximize I(X:Y) for label Y.  Control tradeoff using 'label_beta param'
    """
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "393f9de8-a5b9-4d92-aaff-8808d563b6c4",
        "version": "1.0.0",
        "name": "Echo",
        "description": "Autoencoder implementation of Information Bottleneck using Echo Noise: https://arxiv.org/abs/1904.07199.  Can be used for feature construction with the task of classification or regression.  Image featurization and collaborative filtering in prep.  Returns embedding of size n_hidden, alongside predictions (which can be used with downstream modeling primitive).  Beta hyperparam controls regularization: Loss = task_loss - beta * I(X:Z)",
        "python_path": "d3m.primitives.feature_construction.echo_ib.DSBOX",
        "original_python_path": "echo_ib.EchoIB",
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
      "algorithm_types": ["STOCHASTIC_NEURAL_NETWORK"],#"EXPECTATION_MAXIMIZATION_ALGORITHM"],
      "primitive_family": "FEATURE_CONSTRUCTION",
        "hyperparams_to_tune": ["n_hidden", "beta", "epochs"]
    })



    def __init__(self, *, hyperparams : EchoIB_Hyperparams) -> None: #, random_seed : int =  0, docker_containers: typing.Dict[str, DockerContainer] = None
        super().__init__(hyperparams = hyperparams) # random_seed = random_seed, docker_containers = docker_containers)

    def _extra_params(self, latent_dims = None, activation = None, lr = None, batch = None, epochs = None, noise = None):
        self._latent_dims = [200, 200, self.hyperparams['n_hidden']]
        self._decoder_dims = list(reversed(self._latent_dims[:-1]))
        
        # TRAINING ARGS... what to do?
        self._activation = self.hyperparams['activation'] #'tanh' #'softplus'
        self._lr = 0.0001
        self._optimizer = SGD(self._lr) #Adam(self._lr)
        self._batch = int(self.hyperparams['batch']) #20
        self._epochs = None # HYPERPARAM?
        self._noise = 'echo'
        self._kl_warmup = 1 # .1 * kl reg for first _ epochs
        self._anneal_sched = None # not supported
        self._echo_args = {'batch': self._batch, 'd_max': self._batch, 'nomc': True, 'calc_log': True, 'plus_sx': True}

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
            x = keras_layers.Input(shape = (self.training_inputs.shape[-1],))
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

            #z_var_act = log_sigmoid_64

            z_mean = Dense(self._latent_dims[-1], activation = z_mean_act, name = 'z_mean')(t)
            z_noise = Dense(self._latent_dims[-1], activation = z_var_act, name = 'z_noise', bias_initializer = 'ones')(t)
            z_act = Lambda(echo_sample, arguments = self._echo_args, output_shape = (self._latent_dims[-1],), name = 'z_act')([z_mean, z_noise])

        t = z_act
        for i in range(len(self._decoder_dims)):
            t = Dense(self._decoder_dims[i], name = 'decoder_'+str(i), activation = self._activation)(t) 
        

        if 'classification' in self.hyperparams['task'].lower():
            label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
            y_pred = Dense(self._label_unique, activation = label_act, name = 'y_pred')(t)
        elif 'regression' in self.hyperparams['task'].lower():
            label_act = 'linear'
            y_pred = Dense(self.training_outputs.shape[-1], activation = label_act, name = 'y_pred')(t)
        else:
            raise NotImplementedError("TASK TYPE SHOULD BE CLASSIFICATION OR REGRESSION")

        # TO DO : Add reconstruction layers and additional representation as in https://arxiv.org/abs/1912.00646

        
        outputs = []
        loss_functions = []
        loss_weights = []
        

        outputs.append(y_pred)
        if label_act == 'softmax':
            loss_functions.append(K.categorical_crossentropy)
        elif label_act == 'sigmoid':
            loss_functions.append(K.binary_crossentropy)
        else: 
            loss_functions.append(tf.keras.losses.mean_squared_error)#mse
        loss_weights.append(1)

        loss_tensor = Lambda(latent_loss)([z_mean,z_noise])
        outputs.append(loss_tensor)
        loss_functions.append(dim_sum)
        loss_weights.append(tf.Variable(self.hyperparams["beta"], dtype = tf.float32, trainable = False))

        if self._kl_warmup is not None and self._kl_warmup > 0:
            my_callbacks = [ZeroAnneal(lw = self.hyperparams['beta'], index = -1, epochs = self._kl_warmup)] 
        else:
            my_callbacks = []
        my_callbacks.append(keras.callbacks.TerminateOnNaN())
        self.model = keras.models.Model(inputs = x, outputs = outputs)
        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)
        get_session().run(tf.global_variables_initializer())

        # anneal? 
        if self._anneal_sched:
            raise NotImplementedError
        else:

            self.model.fit_generator(generator(self.training_inputs, self.training_outputs, target_len = len(outputs), batch = self._batch),
                                     verbose = 1, #callbacks = my_callbacks,
                                     steps_per_epoch=int(self.training_inputs.shape[0]/self._batch), epochs = int(self.hyperparams["epochs"]))

        self.fitted = True
        
        return CallResult(None, True, self.hyperparams["epochs"])

    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]: # TAKES IN DF with index column
        self._extra_params()
        
        modeling = self.hyperparams['use_as_modeling']
        inp = self.model.input
        outputs = [layer.output for layer in self.model.layers if 'z_mean' in layer.name or 'z_noise' in layer.name]
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
        dec_inp = [layer.input for layer in self.model.layers if 'decoder_0' in layer.name][0]
        
        preds = [layer.output for layer in self.model.layers if 'y_pred' in layer.name]
        pred_function = K.function([dec_inp, K.learning_phase()], [preds[0]])

        inps = inputs.remove_columns([inputs.columns.get_loc('d3mIndex')])
        predictions = []
        features = []
        for i in range(0, inps.shape[0], self._batch):
            data = inps.values[i:i+self._batch]
            z_stats = [func([data, 1.])[0] for func in functors]
        
            _echo_args = copy.copy(self._echo_args)
            _echo_args['batch'] = data.shape[0]
            _echo_args['d_max'] = data.shape[0]
            
            z_act = echo_sample(z_stats, **_echo_args).eval(session=get_session())
            
            y_pred= pred_function([z_act, 1.])[0]#.eval(session=K.get_session())
            features.extend([z_act[yp] for yp in range(z_act.shape[0])])
        
            
            y_pred = np.argmax(y_pred, axis = -1)
            predictions.extend([y_pred[yp] for yp in range(y_pred.shape[0])])
            
                
            
        predictions = np.array(predictions)
        if self.label_encode is not None:
            predictions = self.label_encode.inverse_transform(predictions)
        
        
        if modeling:
            output = d3m_DataFrame(predictions, columns = self.output_columns, generate_metadata = True, source = self)
        else:
            out_df = d3m_DataFrame(inputs, generate_metadata = True)

            # create metadata for the corex columns                                                                                                                                        
            features = np.array(features)

            if len(predictions.shape)<len(features.shape): predictions = np.expand_dims(predictions,axis=-1)
            
            constructed = np.concatenate([features, predictions], axis = -1)
            corex_df = d3m_DataFrame(constructed, generate_metadata = True)

            for column_index in range(corex_df.shape[1]):
                col_dict = dict(corex_df.metadata.query((ALL_ELEMENTS, column_index)))
                col_dict['structural_type'] = type(1.0)
                # FIXME: assume we apply corex only once per template, otherwise column names might duplicate                                                                                    
                col_dict['name'] = str(out_df.shape[1] + column_index)  #'echoib_'+('pred_' if column_index < self.hyperparams['n_hidden'] else 'feature_') + 
                col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')
                
                corex_df.metadata = corex_df.metadata.update((ALL_ELEMENTS, column_index), col_dict)


            # concatenate is --VERY-- slow without this next line                                                                                                                                
            corex_df.index = out_df.index.copy()
            
            outputs = common_utils.append_columns(out_df, corex_df)


        if modeling:
            self._training_indices = [c for c in inputs.columns if isinstance(c, str) and 'index' in c.lower()]
            #print("Traning indices ", self._training_indices)
            outputs = common_utils.combine_columns(return_result='new', #self.hyperparams['return_result'],
                                                   add_index_columns=True,#self.hyperparams['add_index_columns'],
                                                   inputs=inputs, columns_list=[output], source=self, column_indices=self._training_indices)

        #predictions = d3m_DataFrame(predictions, index = inputs.index.copy())# columns = self.output_columns)

        return CallResult(outputs, True, 1)
        #return CallResult(d3m_DataFrame(self.model.predict(inputs)), True, 0)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        inps = inputs.remove_columns([inputs.columns.get_loc('d3mIndex')])

        self.training_inputs = inps.values
        
        
        self.output_columns = outputs.columns

        if 'classification' in self.hyperparams['task'].lower():
            self._label_unique = np.unique(outputs.values).shape[0]
            self.label_encode = preprocessing.LabelEncoder()
            self.training_outputs = to_categorical(self.label_encode.fit_transform(outputs.values), num_classes = np.unique(outputs.values).shape[0])
        else:
            self.training_outputs = outputs.values
        
        #self.training_outputs = to_categorical(outputs, num_classes = np.unique(outputs.values).shape[0])
        self.fitted = False
        

        # DATA PROFILING? softmax categorical (encoded) X or labels Y 
        # binary data? np.logical_and(self.training_inputs >= 0, self.training_inputs )
        
        # CHECK unique values for determining discrete / continuous
        #self._input_types = []
        #self._label_unique = np.unique(outputs).shape[0]
        #self._label_unique = 1 if self._label_unique > self.max_discrete_labels else self._label_unique


    def get_params(self) -> EchoIB_Params:
        return EchoIB_Params(model = self.model, model_weights = self.model.get_weights(), fitted = self.fitted, label_encode = self.label_encode, output_columns = self.output_columns)#max_discrete_labels = self.max_discrete_labels)#args)

    def set_params(self, *, params: EchoIB_Params) -> None:
        #self.max_discrete_labels = params["max_discrete_labels"]
        self._extra_params()
        self.model = params['model']
        self.model.set_weights(params['model_weights'])
        self.fitted = params['fitted']
        self.label_encode = params['label_encode']
        self.output_columns = params['output_columns']

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
            
                
            x_batch = np.array(data[batch*counter:batch*(counter+1)]).astype('float32')
            y_batch = x_batch if unsupervised or labels is None else np.array(labels[batch*counter:batch*(counter+1)]).astype('float32')#.ravel()
            counter += 1
             
            yield (x_batch, [y_batch]*target_len)

            #restart counter to yeild data in the next epoch as well                                                                                                          
            if counter >= number_of_batches:
                counter = 0
                sd = np.random.randint(data.shape[0])
                np.random.seed(sd)
                np.random.shuffle(data)
                #data.sample(frac = 1, random_state = sd)
                np.random.seed(sd)
                np.random.shuffle(labels)
                #labels.sample(frac =1, random_state = sd)
                

def tanh64(x):
    y = 64
    return (K.exp(1.0/y*x)-K.exp(-1.0/y*x))/(K.exp(1.0/y*x)+K.exp(-1.0/y*x)+K.epsilon())

def log_sigmoid_64(x):
    y = 1.0/64
    return K.log(1.0/(1.0+K.exp(-y*x)))

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

def echo_loss(inputs, d_max = 100, clip= 0.85, binary_input = True, calc_log = True, plus_sx = True, neg_log = False, fx_clip = None):
    if isinstance(inputs, list):
        #cap_param = inputs[0]                                                                                                                                                                             
        cap_param = inputs[-1]
        # -1 for stat, 0 for addl                                                                                                                                                                          
    else:
        cap_param = inputs

    capacities = -K.log(K.abs(clip*cap_param)+K.epsilon()) if not calc_log else -(tf.log(clip) + (cap_param if plus_sx else -cap_param))
    return capacities


def random_indices(n, d):
    return tf.random.uniform((n * d,), minval=0, maxval=n, dtype=tf.int32)

def gather_nd_reshape(t, indices, final_shape):
    h = tf.gather_nd(t, indices)
    return K.reshape(h, final_shape)


def permute_neighbor_indices(batch_size, d_max=-1, replace = False, pop = True):
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
            # pop = False includes training sample for echo 
            # (i.e. dmax = batch instead of dmax = batch - 1)
            sub_batch.pop(i)
          np.random.shuffle(sub_batch)
          inds.append(list(enumerate(sub_batch[:d_max])))
        return inds
      else:
        for i in range(batch_size):
            inds.append(list(enumerate(np.random.choice(batch_size, size = d_max, replace = True))))
        return inds



def echo_sample(inputs, clip=None, d_max=100, batch=100, multiplicative=False, echo_mc = False,
                replace=True, fx_clip=None, plus_sx=True, calc_log=True, return_noise=False, **kwargs):
    # kwargs unused

    # inputs should be specified as list:
    #   [ f(X), s(X) ] with s(X) in log space if calc_log = True 
    # plus_sx =
    #   True if logsigmoid activation for s(X)
    #   False for softplus (equivalent)
    if d_max is None:
        d_max = batch
        
    if isinstance(inputs, list):
        fx = inputs[0]
        sx = inputs[-1]
    else:
        fx = inputs

    # TO DO : CALC_LOG currently determines both whether to do log space calculations AND whether sx is a log
 
    try:
        fx_shape = fx.get_shape()
        sx_shape = sx.get_shape()
    except:
        fx_shape = fx.shape
        sx_shape = sx.shape

    if clip is None:
    # clip is multiplied times s(x) to ensure that last sampled term:
    #   (clip^d_max)*f(x) < machine precision 
        max_fx = fx_clip if fx_clip is not None else 1.0
        clip = (2**(-23)/max_fx)**(1.0/d_max)
    
    # fx_clip can be used to restrict magnitude of f(x), not used in paper
    # defaults to no clipping and M = 1 (e.g. with tanh activation for f(x))
    if fx_clip is not None: 
        fx = K.clip(fx, -fx_clip, fx_clip)
    

    if not calc_log:
        sx = tf.multiply(clip,sx)
        sx = tf.where(tf.abs(sx) < K.epsilon(), K.epsilon()*tf.sign(sx), sx)
    #raise ValueError('calc_log=False is not supported; sx has to be log_sigmoid')
    else:
        # plus_sx based on activation for sx = s(x):
        #   True for log_sigmoid
        #   False for softplus
        sx = tf.log(clip) + (-1*sx if not plus_sx else sx)
    
    if echo_mc is not None and echo_mc:    
      # use mean centered fx for noise
      fx = fx - K.mean(fx, axis = 0, keepdims = True)

    #batch_size = K.shape(fx)[0]
    z_dim = K.int_shape(fx)[-1]
    
    if replace: # replace doesn't set batch size (using permute_neighbor_indices does)
        sx = K.batch_flatten(sx) if len(sx_shape) > 2 else sx 
        fx = K.batch_flatten(fx) if len(fx_shape) > 2 else fx 
        inds = K.reshape(random_indices(batch, d_max), (-1, 1))
        select_sx = gather_nd_reshape(sx, inds, (-1, d_max, z_dim))
        select_fx = gather_nd_reshape(fx, inds, (-1, d_max, z_dim))

        if len(sx_shape)>2:
          select_sx = K.expand_dims(K.expand_dims(select_sx, 2), 2)
          sx = K.expand_dims(K.expand_dims(sx, 1),1)
        if len(fx_shape)>2:
          select_fx = K.expand_dims(K.expand_dims(select_fx, 2), 2)
          fx = K.expand_dims(K.expand_dims(fx, 1),1)

    else:
        # batch x batch x z_dim 
        # for all i, stack_sx[i, :, :] = sx
        repeat = tf.multiply(tf.ones_like(tf.expand_dims(fx, 0)), tf.ones_like(tf.expand_dims(fx, 1)))
        stack_fx = tf.multiply(fx, repeat)
        stack_sx = tf.multiply(sx, repeat)

        # select a set of dmax examples from original fx / sx for each batch entry
        inds = permute_neighbor_indices(batch, d_max, replace = replace)
        
        # note that permute_neighbor_indices sets the batch_size dimension != None
        # this necessitates the use of fit_generator, e.g. in training to avoid 'remainder' batches if data_size % batch > 0
        
        select_sx = tf.gather_nd(stack_sx, inds)
        select_fx = tf.gather_nd(stack_fx, inds)
      
    if calc_log:
        sx_echoes = tf.cumsum(select_sx, axis = 1, exclusive = True)
    else:
        sx_echoes = tf.cumprod(select_sx, axis = 1, exclusive = True)
    
    # calculates S(x0)S(x1)...S(x_l)*f(x_(l+1))
    sx_echoes = tf.exp(sx_echoes) if calc_log else sx_echoes 
    fx_sx_echoes = tf.multiply(select_fx, sx_echoes) 
    
    # performs the sum over dmax terms to calculate noise
    noise = tf.reduce_sum(fx_sx_echoes, axis = 1) 
    
    
    #if multiplicative:
        # unused in paper, not extensively tested
        #sx = sx if not calc_log else tf.exp(sx) 
        #output = tf.multiply(fx, tf.multiply(sx, noise))
        
    #else:
    sx = sx if not calc_log else tf.exp(sx) 
    output = fx + tf.multiply(sx, noise) 

    return output if not return_noise else noise













