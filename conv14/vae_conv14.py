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
from keras.layers import Concatenate

def VAE(original_dim = (224,224,3), latent_dim = 2048, epsilon_std = 1.0, lr = 0.0001, is_sum = True): 
    img = Input(shape=original_dim)                                        
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',strides = 2)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',strides = 2)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', strides =2)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', strides = 2)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)                           

    z_mean = Conv2D(512,(1,1), activation='linear',padding='same',name='z_mean')(x)
    z_log_var = Conv2D(512,(1,1),activation='linear',padding = 'same',name='z_log_var')(x)
    z_mean_var = Concatenate(axis=-1)([z_mean,z_log_var])
    
    def sampling(args):                                     
        z_mean, z_log_var = args                
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],K.shape(z_mean)[1],14,14), mean=0.,   
                                  stddev=epsilon_std)   
        return z_mean + K.exp(z_log_var / 2) * epsilon   
    
    z = Lambda(sampling, output_shape = (512,14,14,),name='sampling')([z_mean,z_log_var])

                                                                           
    '''
    flat = Flatten()(x)                                                                       
                                                                                              
    z_mean = Dense(latent_dim)(flat) 
    z_log_var = Dense(latent_dim)(flat) 
    z_mean_var = Concatenate(axis = -1)([z_mean,z_log_var])

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
    '''

    #up1 = UpSampling2D(size=(2, 2),name='decode_upsample_1')(z)                       
    #decode_conv2_1 = Conv2D(256, (3, 3),padding='same', activation='relu',name='decode_conv1')(up1) 

    up2 = UpSampling2D(size=(2, 2),name='decode_upsample_2')(z)                  
    decode_conv2_2 = Conv2D(128, (3, 3), padding='same',activation='relu',name='decode_conv2')(up2) 

    up3 = UpSampling2D(size=(2, 2),name='decode_upsample_3')(decode_conv2_2)             
    decode_conv2_3 = Conv2D(64, (3, 3), padding='same',activation='relu',name='decode_conv3')(up3) 
                                                                                                   

    up4 = UpSampling2D(size=(2, 2),name='decode_upsample_4')(decode_conv2_3)                       
    decode_conv2_4 = Conv2D(32, (3, 3),padding='same', activation='relu',name='decode_conv4')(up4) 
    
    up5 = UpSampling2D(size=(2, 2),name='decode_upsample_5')(decode_conv2_4)
    decode_conv2_5 = Conv2D(3, (3, 3),padding='same', activation='tanh',name = 'decode_conv5')(up5)

                                                                                                   
    # instantiate VAE model                                                                        
    #vae = Model(img, decode_conv2_5)  
    vae = Model(img, [decode_conv2_5,z_mean_var])     

    def l2_loss(img, decode_x):
        if is_sum == True:               
            l2_loss = K.sum(K.square(img- decode_x), axis = (1,2,3))                       
        else:              
            l2_loss = K.mean(K.square(img- decode_x), axis = (1,2,3))                                                                                                                   
        return  l2_loss
    '''
    def kl_loss(img,pred):
        dim = pred.shape[-1]
        z_mean = pred[:,0:dim/2]
        z_logvar = pred[:,dim/2:]
        if is_sum == True:
            kl_loss = - 0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1) 
        else:
            kl_loss = - 0.5 * K.mean(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1)        
        return kl_loss
    '''
    def kl_loss(img,pred):
        dim = pred.shape[-1]
        z_mean = pred[:,:,:,:dim/2]
        z_logvar = pred[:,:,:,dim/2:]
        if is_sum == True:
            kl_loss = - 0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=(1,2,3)) 
        else:
            kl_loss = - 0.5 * K.mean(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=(1,2,3))        
        return kl_loss
    # Compute VAE loss                  
    #vae.add_loss(vae_loss)      
    rmsprop = keras.optimizers.RMSprop(lr=lr, rho=0.9,  decay=0.0)
    vae.compile(optimizer=rmsprop,loss=[l2_loss, kl_loss])                                   
    #vae.summary()
    
    return vae
