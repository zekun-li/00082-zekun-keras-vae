import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' 
import keras
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.models import load_model
from vae_conv14 import VAE
from keras.callbacks import ModelCheckpoint,CSVLogger, TensorBoard
from keras.callbacks import Callback
from keras import metrics
import keras.backend as K
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import h5py

K.set_image_data_format('channels_first')

is_sum = False # this assignment does not matteer, with be reset in main funciton

def main(args):
    #test_with_generator(args)
    test_with_saved_data(args)


def l2_loss(img, decode_x):
    global is_sum
    if is_sum == True:               
        l2_loss = K.sum(K.square(img- decode_x), axis = (1,2,3))                       
    else:              
        l2_loss = K.mean(K.square(img- decode_x), axis = (1,2,3))                                                                                                                   
        return  l2_loss

def kl_loss(img,pred):
    global is_sum
    dim = pred.shape[-1]
    z_mean = pred[:,:,:,:dim/2]
    z_logvar = pred[:,:,:,dim/2:]
    if is_sum == True:
        kl_loss = - 0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=(1,2,3)) 
    else:
        kl_loss = - 0.5 * K.mean(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=(1,2,3))        
        return kl_loss


#####################################

def test_with_saved_data(args):
    global is_sum
    mean_img_file = args.mean_img_file
    is_sum = args.issum
    if_xscale = args.ifxscale
    #if_localconv = args.iflocalconv
    #bottleneck_filter_size = args.bottleneck_filter_size
    b_size = args.bsize
    saved_weights = args.saved_weights #'weights/01_lr0.0001_conv14_nomeanfile_sumloss_xscale-best-98-2659.69.hdf5'
    #lr_rate = args.lr
    val_steps_per_epoch = args.val_steps_per_epoch
    #ith = args.ith

    #data_path = '/lfs2/tmp/anh-train/
    data_path = os.environ['TMPDIR']+'/'
    #data_path = '../debug_data/'
    nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    #############################################################################################
    #model = VAE(original_dim = (3,224,224),  epsilon_std = 1.0, lr = lr_rate, is_sum = is_sum, bottleneck_filter_size = bottleneck_filter_size, if_localconv = if_localconv) #weights=None for random initialization
    # load weights
    if saved_weights is not None:
        assert os.path.isfile(saved_weights)
        #model.load_weights(saved_weights)
        model = load_model(saved_weights, custom_objects={'l2_loss': l2_loss, 'kl_loss':kl_loss})

    model.summary()


    valid_x = np.load('../valid_X.npy')
    predict_x = model.predict(valid_x)[0]
    valid_x = valid_x * 255. + 128.
    predict_x = predict_x * 255 + 128.
    import cv2 
    i = 0   
    for vx, px in zip(valid_x, predict_x):
        cv2.imwrite('out_visual/in_'+str(i)+'.jpg', vx.transpose(1,2,0))
        cv2.imwrite('out_visual/out_'+str(i)+'.jpg', px.transpose(1,2,0))
        i = i+1


def test_with_generator(args):
    global is_sum
    sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
    from vae_conv_face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen

    ######### params #############
    #mean_img_file = 'model/keras_mean_img.npy' 
    print (args)
    mean_img_file = args.mean_img_file
    is_sum = args.issum
    if_xscale = args.ifxscale
    #if_localconv = args.iflocalconv
    #bottleneck_filter_size = args.bottleneck_filter_size
    b_size = args.bsize
    saved_weights = args.saved_weights #'weights/01_lr0.0001_conv14_nomeanfile_sumloss_xscale-best-98-2659.69.hdf5'
    #lr_rate = args.lr
    val_steps_per_epoch = args.val_steps_per_epoch
    #ith = args.ith

    #data_path = '/lfs2/tmp/anh-train/
    data_path = os.environ['TMPDIR']+'/'
    #data_path = '../debug_data/'
    nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    #############################################################################################
    #model = VAE(original_dim = (3,224,224),  epsilon_std = 1.0, lr = lr_rate, is_sum = is_sum, bottleneck_filter_size = bottleneck_filter_size, if_localconv = if_localconv) #weights=None for random initialization
    # load weights
    if saved_weights is not None:
        assert os.path.isfile(saved_weights)
        #model.load_weights(saved_weights)
        model = load_model(saved_weights, custom_objects={'l2_loss': l2_loss, 'kl_loss':kl_loss})

    model.summary()
    # generators
    if mean_img_file is not None:
        val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file , if_xscale = if_xscale)
    else:
        val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, if_xscale = if_xscale)


    valid_X  = []
    valid_X1_1,valid_X1_2 = [], []
    for step in range(val_steps_per_epoch):
        batch_x, batch_x1 = val_datagen[step]
        valid_X.append( batch_x)
        batch_x1_1, batch_x1_2 = batch_x1
        #valid_X1.append( batch_x1)
        valid_X1_1.append(batch_x1_1)
        valid_X1_2.append(batch_x1_2)

    valid_X = np.concatenate(valid_X, axis = 0)
    valid_X1_1 = np.concatenate(valid_X1_1, axis = 0)
    valid_X1_2 = np.concatenate(valid_X1_2, axis = 0)
    print (valid_X.shape)
    print (valid_X1_1.shape)
    print (valid_X1_2.shape)

    valid_x = valid_X[0:10]
    predict_x = model.predict(valid_x)[0]
    valid_x = valid_x * 255. + 128.
    predict_x = predict_x * 255 + 128.
    import cv2 
    i = 0   
    for vx, px in zip(valid_x, predict_x):
        cv2.imwrite('in_'+str(i)+'.jpg', vx.transpose(1,2,0))
        cv2.imwrite('out_'+str(i)+'.jpg', px.transpose(1,2,0))
        i = i+1


    
    #H = model.fit_generator(generator = train_datagen, steps_per_epoch = train_steps_per_epoch, epochs = 100, validation_data = (valid_X, [valid_X1_1,valid_X1_2]), callbacks =mycallbacks )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean-img-file', help = 'path point to mean img file (.npy)', default = None)
    parser.add_argument('--issum', help='whether use sum or mean for loss function', default = False, action = 'store_true')
    parser.add_argument('--ifxscale', help='whether scale input to [-1,1] (if not, then [-0.5, 0.5])', default = False, action = 'store_true')
    #parser.add_argument('--iflocalconv', help='whether use localconv for bottleneck layer', default = False, action = 'store_true')
    #parser.add_argument('--bottleneck-filter-size', help = 'bottleneck filter size', default = 1, type = int)
    parser.add_argument('--bsize', help = 'batch size ', default = 32, type = int)
    parser.add_argument('--saved-weights', help = 'path point to checkpoint file', default = None)
    #parser.add_argument('--lr', help = 'learning rate', default = 0.0001, type = float)
    parser.add_argument('--train-steps-per-epoch', help = 'the steps per epoch for training ', default = 100, type = int)
    parser.add_argument('--val-steps-per-epoch', help = 'the steps per epoch for validation ', default = 100, type = int)
    #parser.add_argument('--ith', help = 'the ith training ', default = 0, type = int)
    args = parser.parse_args()
    main(args)
