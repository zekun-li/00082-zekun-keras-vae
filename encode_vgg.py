import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 
from debug_vae_face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from vae import VAE
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.callbacks import Callback
from keras import metrics
import keras.backend as K
import tensorflow as tf
import numpy as np

K.set_image_data_format('channels_first')
######### params #############
#mean_img_file = 'model/keras_mean_img.npy' 
mean_img_file = None
is_sum = True
if_xscale = True
b_size = 32
#saved_weights = 'weights/12_lr0.0001_latentdim2048_nomeanfile_sumloss_xscale-best-25-3097.78.hdf5'
saved_weights = 'weights/12_lr0.0001_latentdim2048_nomeanfile_sumloss_xscale-best-94-2743.67.hdf5'
lr_rate = 0.0001  
#train_steps_per_epoch=10
val_steps_per_epoch = 10
latent_dim = 2048

#data_path = os.environ['TMPDIR']+'/'
data_path = 'debug_data/'
nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))


model = VAE(original_dim = (3,224,224), latent_dim = latent_dim, epsilon_std = 1.0, lr = lr_rate, is_sum = is_sum) #weights=None for random initialization

# load weights
if saved_weights is not None:
    print ('loading weights', saved_weights)
    assert os.path.isfile(saved_weights)
    model.load_weights(saved_weights)
    
model.summary()

encoder = Model(inputs = model.input, outputs = model.get_layer('dense_1').output)
encoder.summary()

'''
z = model.get_layer('decode_dense1')
decoder = Model(inputs = model.get_layer('decode_dense1').input, outputs = model.output)
decoder.summary()
'''


# generators
if mean_img_file is not None:
    #train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file, latent_dim = latent_dim , if_xscale = if_xscale)
    val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file ,latent_dim = latent_dim, if_xscale = if_xscale)
else:
    #train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), source = data_path, latent_dim = latent_dim, if_xscale = if_xscale)
    val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, latent_dim = latent_dim, if_xscale = if_xscale)

valid_X  = []
valid_X1_1,valid_X1_2 = [], []
valid_Y = []
for step in range(val_steps_per_epoch):
    batch_x, batch_x1, batch_y = val_datagen[step]
    valid_X.append( batch_x)
    valid_Y.append( batch_y)
    batch_x1_1, batch_x1_2 = batch_x1
    #valid_X1.append( batch_x1)
    valid_X1_1.append(batch_x1_1)
    valid_X1_2.append(batch_x1_2)

valid_X = np.concatenate(valid_X, axis = 0)
valid_Y = np.concatenate(valid_Y, axis = 0)
valid_X1_1 = np.concatenate(valid_X1_1, axis = 0)
valid_X1_2 = np.concatenate(valid_X1_2, axis = 0)
print (valid_X.shape)
print (valid_X1_1.shape)
print (valid_X1_2.shape)
print (valid_Y.shape)

'''
#H = multi_model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = (valid_X, valid_Y),  callbacks =mycallbacks)
H = model.fit_generator(generator = train_datagen, steps_per_epoch = train_steps_per_epoch, epochs = 100, validation_data = (valid_X, [valid_X1_1,valid_X1_2]), callbacks =mycallbacks )
#multi_model.evaluate(x = valid_X, y = valid_Y)

'''
predict_X = encoder.predict(valid_X)
print (predict_X.shape)
from sklearn.manifold import TSNE as TSNE

np.save('predict_X.npy',predict_X)
np.save('valid_Y.npy',valid_Y)

X_embedded = TSNE(n_components = 2).fit_transform(predict_X)
import matplotlib.pyplot as plt
plt.switch_backend('agg')

plt.figure(figsize=(6, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=valid_Y)
plt.colorbar()
#plt.show()
plt.savefig('save_fig1.jpg')
