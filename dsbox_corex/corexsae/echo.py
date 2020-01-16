# This exact code is not tested, as it is taken from a wider system and edited for clarity, but it should work up to import errors and pruned function arguments.
# See .json supplementary for architectural details

import tensorflow as tf
import keras.backend as K
import numpy as np

# RECOMMENDED PARAMETERS:
# echo inputs : [f(X), s(X)] with f(X) having tanh_act, s(X) with activation = tf.math.log_sigmoid (or softplus)
# set d_max = batch
# clip ensures noise truncates :  choose 2^(-23/batch) for this tanh_act (magnitude <= 1)
# plus_sx = True if s(X) = log_sigmoid else False, calc_log should be True unless you feed an actual sigmoid s(X)


# you can think about parametrizing an "Echo Block" similarly to a VAE:
# fx = Dense(latent_dim, activation = tanh_act)(h)
# sx = Dense(latent_dim, activation = tf.math.log_sigmoid)(h)
# echo_act = Lambda(echo_sample, arguments = arg_dict)([fx,sx])

def tanh_act(x, y = 64):
	# tanh with extended linear range based on y
	return (K.exp(1.0/y*x)-K.exp(-1.0/y*x))/(K.exp(1.0/y*x)+K.exp(-1.0/y*x)+K.epsilon())


def permute_neighbor_indices(batch_size, d_max=-1, replace = False):
      """Produce an index tensor that gives a permuted matrix of other samples in batch, per sample.  
		 Also adapted to handle sampling with replacement using option replace = True
      
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
          sub_batch.pop(i)
          shuffle(sub_batch)
          inds.append(list(enumerate(sub_batch[:d_max])))
        return inds
      else:
        for i in range(batch_size):
            inds.append(list(enumerate(np.random.choice(batch_size, size = d_max, replace = True))))
        return inds

def echo_sample(inputs, clip = None,  d_max = 100, batch = 100, multiplicative = False, replace = False, fx_clip = None, plus_sx = True, return_noise = False, noisemc = False, calc_log = True):

	# inputs should be specified as list : [ f(X), s(X) ] with s(X) in log space if calc_log = True 
	# plus_sx = True if logsigmoid activation for s(X), False for softplus (equivalent)
    if isinstance(inputs, list):
      z_mean = inputs[0]
      z_scale_echo = inputs[-1]
    else:
      z_mean = inputs
      
    if clip is None:
	# fx_clip can be used to restrict magnitude of f(x) ('mean')
	# defaults to 1 magnitude (e.g. with tanh activation for f(x))
	# clip is multiplied times s(x) to ensure that last sampled term: (clip^d_max)*f(x) < machine precision 
	max_fx = fx_clip if fx_clip is not None else 1.0
	clip = (2**(-23)/max_fx)**(1.0/d_max)
	
    # clipping can also be used to limit magnitude of f(x), not used in paper
    if fx_clip is not None: 
      z_mean = K.clip(z_mean, -fx_clip, fx_clip)
      
    if not calc_log:
      cap_param = clip*z_scale_echo if per_sample else clip*K.sigmoid(z_scale_echo)
      # necesssary clip for cumprod gradients
      cap_param = tf.where(tf.abs(cap_param) < K.epsilon(), K.epsilon()*tf.sign(cap_param), cap_param)
    else:
	# plus_sx based on activation for z_scale_echo = s(x) : true for log_sigmoid , false for softplus
      cap_param = tf.log(clip) + (-1*z_scale_echo if not plus_sx else z_scale_echo)

    inds = permute_neighbor_indices(batch, d_max, replace = replace)

    
    c_z_stack = tf.stack([cap_param for k in range(d_max)])
      
    f_z_stack = tf.stack([z_mean for k in range(d_max)]) 

    stack_dmax = tf.gather_nd(c_z_stack, inds)
    stack_zmean = tf.gather_nd(f_z_stack, inds)

    if calc_log:
      noise_sx_product = tf.cumsum(stack_dmax, axis = 1, exclusive = True)
      
    else:
      noise_sx_product = tf.cumprod(stack_dmax, axis = 1, exclusive = True)
    
    noise_sx_product = tf.exp(noise_sx_product) if calc_log else noise_sx_product
    # calculates S(x0)S(x1)...S(x_l)*f(x_(l+1))
    noise_times_sample = tf.multiply(stack_zmean, noise_sx_product)
   
    # performs the sum over dmax terms to calculate noise
    noise_tensor = tf.reduce_sum(noise_times_sample, axis = 1)

    
    if noisemc:
	noise_tensor -= tf.reduce_mean(noise_tensor, axis=0) # 0 mean noise : ends up being 1 x m
    

    
    if noise == 'multiplicative':
      # unused in paper
      noisy_encoder = tf.multiply(z_mean, tf.multiply(cap_param, noise_tensor))
      #noisy_encoder = tf.exp(z_mean + tf.multiply(cap_param, noise_tensor))
    else:
      sx = cap_param if not calc_log else tf.exp(cap_param) 
      noisy_encoder = z_mean + tf.multiply(sx, noise_tensor) 

    return noisy_encoder if not return_noise else noise_tensor




def echo_loss(inputs, d_max = 100, clip= 0.85, binary_input = True, calc_log = True, plus_sx = True):
    # input is either [ f(X), s(X) ] or just s(X)
    if isinstance(inputs, list):
        cap_param = inputs[-1]
    else:
        cap_param = inputs

    capacities = -K.log(K.abs(clip*cap_param)+K.epsilon()) if not calc_log else -(tf.log(clip) + (cap_param if plus_sx else -cap_param)) 
    return capacities


# example fit generator

