import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 
from vae_conv_face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from vae_conv14 import VAE
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
is_sum = True
if_xscale = True
b_size = 32
saved_weights = 'weights/02_lr0.0001_conv14_nomeanfile_sumloss_xscale-best-93-2524.28.hdf5'
lr_rate = 0.00001  
train_steps_per_epoch=500
val_steps_per_epoch = 500
#latent_dim = 2048
latent_dim = 4096


#data_path = '/lfs2/tmp/anh-train/
data_path = os.environ['TMPDIR']+'/'
#data_path = '../debug_data/'
nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

#prefix = '03_lr'+ str(lr_rate) + '_latentdim' + str(latent_dim)
prefix = '03_lr'+ str(lr_rate) + '_conv14'
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

# customized callback class
class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

#############################################################################################
#with tf.device('/cpu:0'):
model = VAE(original_dim = (3,224,224), latent_dim = latent_dim, epsilon_std = 1.0, lr = lr_rate, is_sum = is_sum) #weights=None for random initialization

# load weights
if saved_weights is not None:
    assert os.path.isfile(saved_weights)
    model.load_weights(saved_weights)

model.summary()
# generators
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
mycallbacks = [csv_logger, check_point, check_point_best, tensor_board]
mycallbacks = [csv_logger,  check_point_best, tensor_board]
#mycallbacks = [check_point]

#H = multi_model.fit_generator(generator = train_datagen, steps_per_epoch = 1000, epochs = 600, validation_data = (valid_X, valid_Y),  callbacks =mycallbacks)
H = model.fit_generator(generator = train_datagen, steps_per_epoch = train_steps_per_epoch, epochs = 100, validation_data = (valid_X, [valid_X1_1,valid_X1_2]), callbacks =mycallbacks )
#multi_model.evaluate(x = valid_X, y = valid_Y)

