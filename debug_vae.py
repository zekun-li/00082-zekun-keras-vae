'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np             
import matplotlib.pyplot as plt
from scipy.stats import norm   
import keras
from keras.layers import Input, Dense, Lambda ,Conv2D,MaxPooling2D,Flatten, Reshape, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import metrics 
from keras.datasets import mnist

def VAE(original_dim = (224,224,3), latent_dim = 2048, epsilon_std = 1.0, lr = 0.0001, is_sum = True): 
    img = Input(shape=original_dim) 
    en_kernel_size = (4,4)
    de_kernel_size = (3,3)
    # Block 1
    x = Conv2D(64, en_kernel_size, activation='relu', padding='same', name='block1_conv1')(img)
    x = Conv2D(64, en_kernel_size, activation='relu', padding='same', name='block1_conv2',strides = 2)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, en_kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, en_kernel_size, activation='relu', padding='same', name='block2_conv2',strides = 2)(x)
    #x = MaxPooling2D((2, 2), stridess=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, en_kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, en_kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, en_kernel_size, activation='relu', padding='same', name='block3_conv3',strides = 2)(x)
    #x = MaxPooling2D((2, 2), stridess=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, en_kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, en_kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, en_kernel_size, activation='relu', padding='same', name='block4_conv3',strides = 2)(x)
    #x = MaxPooling2D((2, 2), stridess=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, en_kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, en_kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, en_kernel_size, activation='relu', padding='same', name='block5_conv3',strides = 2)(x)
    #x = MaxPooling2D((2, 2), stridess=(2, 2), name='block5_pool')(x)                           
                                                                           
    flat = Flatten()(x)                                                                       
    z_mean = Dense(latent_dim)(flat) 
    z_log_var = Dense(latent_dim)(flat) 

    def sampling(args):               
        z_mean, z_log_var = args      
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,  
                                  stddev=epsilon_std)           
        return z_mean + K.exp(z_log_var / 2) * epsilon     
                                                           
    # note that "output_shape" isn't necessary with the TensorFlow backend 
    z = Lambda(sampling, output_shape=(latent_dim,),name='sampling')([z_mean, z_log_var])  
    
    # we instantiate these layers separately so as to reuse them later     
    decoder_h = Dense(7*7*512, activation='relu',name='decode_dense1')(z)  
    if K.image_data_format() == 'channels_last':
      reshape_h  = Reshape((7,7,512),name='decode_reshape')(decoder_h) 
    else:
      reshape_h  = Reshape((512,7,7),name='decode_reshape')(decoder_h) 
    up1 = UpSampling2D(size=(2, 2),name='decode_upsample_1')(reshape_h)                       
    decode_conv2_1 = Conv2D(256, de_kernel_size,padding='same', activation='relu',name='decode_conv1')(up1) 

    up2 = UpSampling2D(size=(2, 2),name='decode_upsample_2')(decode_conv2_1)                  
    decode_conv2_2 = Conv2D(128, de_kernel_size, padding='same',activation='relu',name='decode_conv2')(up2) 

    up3 = UpSampling2D(size=(2, 2),name='decode_upsample_3')(decode_conv2_2)             
    decode_conv2_3 = Conv2D(64, de_kernel_size, padding='same',activation='relu',name='decode_conv3')(up3) 
                                                                                                   

    up4 = UpSampling2D(size=(2, 2),name='decode_upsample_4')(decode_conv2_3)                       
    decode_conv2_4 = Conv2D(32, de_kernel_size,padding='same', activation='relu',name='decode_conv4')(up4) 
    
    up5 = UpSampling2D(size=(2, 2),name='decode_upsample_5')(decode_conv2_4)
    decode_conv2_5 = Conv2D(3, de_kernel_size,padding='same', activation='tanh',name = 'decode_conv5')(up5)

                                                                                                   
    # instantiate VAE model                                                                        
    vae = Model(img, decode_conv2_5)  
    '''
    def vae_loss(img, decode_conv2_5):
        if is_sum == True:
            l2_loss = K.sum(K.square(img- decode_conv2_5), axis = (1,2,3))
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        else:
            l2_loss = K.mean(K.square(img- decode_conv2_5), axis = (1,2,3))
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return  K.mean(l2_loss + kl_loss)
        #xent_loss: 0.03 vs 130930.2969
    '''
    
    if is_sum == True:
        l2_loss = K.sum(K.square(img- decode_conv2_5), axis = (1,2,3))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    else:
        l2_loss = K.mean(K.square(img- decode_conv2_5), axis = (1,2,3))
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss =  K.mean(l2_loss + kl_loss)
    #vae_loss = K.mean(kl_loss)
    
    '''
    def vae_loss(img, decode_conv2_5):
        if is_sum == True:
            l2_loss = K.sum(K.square(img- decode_conv2_5), axis = None)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=None)
        else:
            l2_loss = K.mean(K.square(img- decode_conv2_5), axis = None)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=None)
        return  l2_loss + kl_loss
    '''
        
    # Compute VAE loss                  
    vae.add_loss(vae_loss)      
    rmsprop = keras.optimizers.RMSprop(lr=lr, rho=0.9,  decay=0.0 )
    vae.compile(optimizer=rmsprop, loss = None)                                   
    #vae.summary()
    
    return vae