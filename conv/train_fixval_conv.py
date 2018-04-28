import sys
import os
sys.path.insert(0, '/nfs/isicvlnas01/users/iacopo/codes/Aug_Layer_v2/')
os.environ['KERAS_BACKEND'] = 'tensorflow' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 
from vae_conv_face_aug_datagen_prefetch_mp_queue import  FaceAugDataGen
import keras
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.models import Sequential
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

def main(args):
    ######### params #############
    #mean_img_file = 'model/keras_mean_img.npy' 
    print (args)
    mean_img_file = args.mean_img_file
    is_sum = args.issum
    if_xscale = args.ifxscale
    if_localconv = args.iflocalconv
    bottleneck_filter_size = args.bottleneck_filter_size
    b_size = args.bsize
    saved_weights = args.saved_weights #'weights/01_lr0.0001_conv14_nomeanfile_sumloss_xscale-best-98-2659.69.hdf5'
    lr_rate = args.lr
    train_steps_per_epoch= args.train_steps_per_epoch
    val_steps_per_epoch = args.val_steps_per_epoch
    ith = args.ith

    #data_path = '/lfs2/tmp/anh-train/'
    #data_path = os.environ['TMPDIR']+'/'
    data_path = '../debug_data/'
    nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    prefix = str(ith) + '_lr'+ str(lr_rate) + '_conv14'
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

    if if_localconv == True:
        prefix +='_localconv'
    else:
        prefix += '_conv'

    if type(bottleneck_filter_size) is not tuple:
        bottleneck_filter_size = (bottleneck_filter_size, bottleneck_filter_size)

    prefix = prefix + '_bfs' + str(bottleneck_filter_size[0]) + 'x' + str(bottleneck_filter_size[1])

    csv_logger = CSVLogger('logs/'+prefix + '_train.log')     
    tensor_board = TensorBoard( log_dir= 'logs/'+prefix + '_tb') 
    model_save_path = 'weights/'+prefix + '-{epoch:02d}-{val_loss:.2f}.hdf5' 
    model_save_best = 'weights/'+prefix + '-best-{epoch:02d}-{val_loss:.2f}.hdf5'  
    #optimizer_save_path = 'weights/' +prefix + '-optimizer.pkl'

    #############################################################################################
    #with tf.device('/cpu:0'):
    model = VAE(original_dim = (3,224,224),  epsilon_std = 1.0, lr = lr_rate, is_sum = is_sum, bottleneck_filter_size = bottleneck_filter_size, if_localconv = if_localconv) #weights=None for random initialization

    # load weights
    if saved_weights is not None:
        assert os.path.isfile(saved_weights)
        model.load_weights(saved_weights)
        f = h5py.File(saved_weights, mode = 'r')
        if 'optimizer_weights' in f:
            if isinstance(model, Sequential):
                model.model._make_train_function()
            else:
                model._make_train_function()

            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [n.decode('utf8') for n in
                                      optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]    

            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')

    model.summary()
    # generators
    if mean_img_file is not None:
        train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file, if_xscale = if_xscale)
        val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, mean_file = mean_img_file , if_xscale = if_xscale)
    else:
        train_datagen = FaceAugDataGen(mode = 'training', batch_size=b_size ,im_shape = (224,224), source = data_path,if_xscale = if_xscale)
        val_datagen = FaceAugDataGen(mode = 'validation', batch_size=b_size ,im_shape = (224,224), source = data_path, if_xscale = if_xscale)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean-img-file', help = 'path point to mean img file (.npy)', default = None)
    parser.add_argument('--issum', help='whether use sum or mean for loss function', default = False, action = 'store_true')
    parser.add_argument('--ifxscale', help='whether scale input to [-1,1] (if not, then [-0.5, 0.5])', default = False, action = 'store_true')
    parser.add_argument('--iflocalconv', help='whether use localconv for bottleneck layer', default = False, action = 'store_true')
    parser.add_argument('--bottleneck-filter-size', help = 'bottleneck filter size', default = 1, type = int)
    parser.add_argument('--bsize', help = 'batch size ', default = 32, type = int)
    parser.add_argument('--saved-weights', help = 'path point to checkpoint file', default = None)
    parser.add_argument('--lr', help = 'learning rate', default = 0.0001, type = float)
    parser.add_argument('--train-steps-per-epoch', help = 'the steps per epoch for training ', default = 100, type = int)
    parser.add_argument('--val-steps-per-epoch', help = 'the steps per epoch for validation ', default = 100, type = int)
    parser.add_argument('--ith', help = 'the ith training ', default = 0, type = int)
    args = parser.parse_args()
    main(args)
