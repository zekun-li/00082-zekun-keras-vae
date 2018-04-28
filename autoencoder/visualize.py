import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
#from vae_face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from autoencoder import Autoencoder
from keras.callbacks import ModelCheckpoint,CSVLogger, TensorBoard
from keras.callbacks import Callback
from keras import metrics
import keras.backend as K
import tensorflow as tf
import numpy as np

K.set_image_data_format('channels_first')
######### params #############
#mean_img_file = 'model/keras_mean_img.npy' 
mean_img_file = None
#nb_classes = 62955
is_sum = False
if_xscale = True
b_size = 32
saved_weights = 'weights/02_lr0.0001_latentdim4096_nomeanfile_meanloss_xscale-best-92-0.01.hdf5'
lr_rate = 0.0001  
train_steps_per_epoch=500
val_steps_per_epoch = 500
#latent_dim = 2048
latent_dim = 4096


#data_path = '/lfs2/tmp/anh-train/
data_path = os.environ['TMPDIR']+'/'
#data_path = 'debug_data/'
nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

prefix = '02_lr'+ str(lr_rate) + '_latentdim' + str(latent_dim)
if mean_img_file is not None:
    prefix +='_usemeanfile'
else:
    prefix +='_nomeanfile'

if is_sum == True:
    prefix +='_sumloss'
else:
    prefix += '_meanloss'

if if_xscale == True:
    prefix += '_xscale'
else:
    prefix += '_noxscale'

csv_logger = CSVLogger('logs/'+prefix + '_train.log')     
tensor_board = TensorBoard( log_dir= 'logs/'+prefix + '_tb') 
model_save_path = 'weights/'+prefix + '-{epoch:02d}-{val_loss:.2f}.hdf5' 
model_save_best = 'weights/'+prefix + '-best-{epoch:02d}-{val_loss:.2f}.hdf5'  

#####################################


#############################################################################################
#with tf.device('/cpu:0'):
model = Autoencoder(original_dim = (3,224,224), latent_dim = latent_dim, epsilon_std = 1.0, lr = lr_rate, is_sum = is_sum) #weights=None for random initialization

# load weights
if saved_weights is not None:
    assert os.path.isfile(saved_weights)
    model.load_weights(saved_weights)


model.summary()
# generators
'''
if mean_img_file is not None:
    train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file, latent_dim = latent_dim , if_xscale = if_xscale)
    val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file ,latent_dim = latent_dim, if_xscale = if_xscale)
else:
    train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), source = data_path, latent_dim = latent_dim, if_xscale = if_xscale)
    val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, latent_dim = latent_dim, if_xscale = if_xscale)

# callbacks
#check_point = MultiGPUCheckpointCallback(filepath = model_save_path,base_model = model,  save_best_only=False,period = 10)   
#check_point_best = MultiGPUCheckpointCallback(filepath = model_save_best, base_model = model, save_best_only=True)
check_point = ModelCheckpoint(filepath = model_save_path,  save_best_only=False,period = 10)   
check_point_best = ModelCheckpoint(filepath = model_save_best, save_best_only=True)
'''
'''
valid_X , valid_X1 = [],[]
for step in range(val_steps_per_epoch):
    batch_x, batch_x1 = val_datagen[step]
    valid_X.append( batch_x)
    valid_X1.append( batch_x1)
valid_X = np.concatenate(valid_X, axis = 0)
valid_X1 = np.concatenate(valid_X1, axis = 0)
print (valid_X.shape)
print (valid_X1.shape)

mycallbacks = [csv_logger, check_point, check_point_best]
#mycallbacks = [check_point]

#H = multi_model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = (valid_X, valid_Y),  callbacks =mycallbacks)
H = model.fit_generator(generator = train_datagen, steps_per_epoch = train_steps_per_epoch, epochs = 600, validation_data = (valid_X, valid_X1), callbacks =mycallbacks )
#multi_model.evaluate(x = valid_X, y = valid_Y)
'''

'''

valid_X  = []
for step in range(val_steps_per_epoch):
    batch_x, _ = val_datagen[step]
    valid_X.append( batch_x)

valid_X = np.concatenate(valid_X, axis = 0)
print (valid_X.shape)

mycallbacks = [csv_logger, tensor_board, check_point_best]
#mycallbacks = [check_point]
'''
valid_X = np.load('../valid_X.npy')
valid_x = valid_X
#H = multi_model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = (valid_X, valid_Y),  callbacks =mycallbacks)
#H = model.fit_generator(generator = train_datagen, steps_per_epoch = train_steps_per_epoch, epochs = 100, validation_data = (valid_X, [valid_X]), callbacks =mycallbacks )
#multi_model.evaluate(x = valid_X, y = valid_Y)

predict_x = model.predict(valid_x) #[0]
valid_x = valid_x * 255. + 128.
predict_x = predict_x * 255 + 128.
import cv2 
i = 0   
for vx, px in zip(valid_x, predict_x):
    cv2.imwrite('out_visual/in_'+str(i)+'.jpg', vx.transpose(1,2,0))
    cv2.imwrite('out_visual/out_'+str(i)+'.jpg', px.transpose(1,2,0))
    i = i+1

