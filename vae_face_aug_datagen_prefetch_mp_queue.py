# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
#import caffe
import util
import cv2
import sys
import os
import lmdb

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
import Queue as std_Queue
from multiprocessing import Queue, Process, Event, Pipe
from mytools import MySimpleTransformer
import aug_tracker
import time
## Ugly global variables #######################
#_firstTimeTrain = True
#_firstTimeValid = True
_QSIZE = 1 #350
_nJobs = 8 #8
# _eventTrainList = [None]*(_nJobs+1)
# _eventValidList = [None]*(_nJobs+1)
# _evenTrainColl = Event()
# for j in range(0,_nJobs+1): #in the +1 we store the collector
#     _eventTrainList[j] = Event()
#     _eventValidList[j] = Event()

################################################
class FaceAugDataGen( object ):
    idx = 0
    def __init__ (self, mode = 'training', batch_size = 16, im_shape = (224,224), source = None, mean_file=None  , latent_dim = None, if_xscale = False):
        #mean_file = 'model/keras_mean_img.npy'
        self.mode = mode
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.source = source # path points to the dir with train/val list file
        self.mean_file = mean_file
        self.latent_dim = latent_dim
        self.if_xscale = if_xscale
        #self.n_classes = n_classes
        assert source is not None
        #assert mean_file is not None
        #assert n_classes is not None
        params = dict()
        params['batch_size'] = batch_size
        params['im_shape'] = im_shape
        params['split'] = mode # train, val, test
        params['source'] =  source
        params['mean_file'] = mean_file


        # only training and validation supported
        if mode == 'training':
            list_file = source + 'train.list'
            self.phase = 0
            self.indexlist = [line.rstrip('\n') for line in open(list_file)]
            ## Immediate Shuffling is important
            shuffle(self.indexlist)

        elif mode == 'validation':
            list_file = source + 'valid.list'
            self.phase = 1
            self.indexlist = [line.rstrip('\n') for line in open(list_file)]

        self.nb_samples = len(self.indexlist)
        indexlist_chunks = [ list(i) for i in np.array_split(np.array(self.indexlist), _nJobs) ]
        self.batch_loader_list = []
        for j in range(0,_nJobs):
            util.myprint("Starting pre-fetching processes id: "+str(j))
            batch_loader = BatchLoader(params, indexlist_chunks[j], self.phase, j)
            ## Starting the process formally ( it will wait until we event.set() )
            batch_loader.start()
            self.batch_loader_list.append(batch_loader)

        self.collector = Collector( self.batch_loader_list,self.batch_size, self.phase )
        def cleanup():
                util.myprint('Terminating BatchLoader')
                for j in range(0,_nJobs):
                    self.batch_loader_list[j].terminate()
                    self.batch_loader_list[j].join()
                #self.collector.terminate()
                #self.collector.join()
        import atexit
        atexit.register(cleanup)

        ########### reshape tops####################
        #TODO
        
        if self.phase == 1:
            print_info("FaceAugDataGen_valid", params)
        else:
            print_info("FaceAugDataGen_train", params)

        return
    '''
    def fit():
        self.nb_samples =
    '''
    def __getitem__(self, batch_idx):
        '''
        if (self.mode = 'training'):
            # shuffle index
        else:
            sample_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        '''
        #start = time.time()
        listData = self.collector.gatherData()
        X = []
        Y = []
        for itt in range(self.batch_size):
            x = listData[itt]['img']
            #x = np.expand_dims( x, axis = 0 )
            y = listData[itt]['label']
            X.append(x)
            Y.append(y)
            '''
        def sparsify(y):
            return np.array([[1 if y[i] == j else 0 for j in range(self.n_classes)]
                   for i in range(y.shape[0])])
        '''
        np.concatenate( X, axis = 0 )
        X = np.array(X)
        X = X/255.
        if self.if_xscale == True:
            X = X*2.
        Y = np.array(Y)
        #Y = sparsify(Y)
        #from keras.utils import to_categorical
        #Y = to_categorical(Y, num_classes = self.n_classes)
        #end = time.time()
        #print batch_idx, end-start
        #return X,Y
        if self.latent_dim is None:
            return X,X
        else:
            #return X,[X, np.zeros((self.batch_size, self.latent_dim))]
            return X, [X, np.zeros((self.batch_size,1))]

    def __iter__(self):
        return self
    def next(self):
        
        old_idx = self.idx
        idx = (1 + self.idx) # % self.nb_batches
        self.idx = idx
        return self[old_idx]
        
        
#class FaceAugDataLayer(caffe.Layer):
    
    """
    This is a simple asynchronous datalayer for training a model on with domain (face)
    specific data augmentation in 2D and 3D with pre-fetching in different proceses.
    """
'''
    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # get list of image indexes.
        if self.phase == 1:
            list_file = params['source']+'valid.list'
        else:
            list_file = params['source']+'train.list'
        self.indexlist = [line.rstrip('\n') for line in open(
            list_file)]
        ## Immediate Shuffling is important
        shuffle(self.indexlist)
        indexlist_chunks = [ list(i) for i in np.array_split(np.array(self.indexlist), _nJobs) ]
        # store input as class variables
        self.batch_size = params['batch_size']
        self.pref_process = []
        self.batch_loader_list = []
        for j in range(0,_nJobs):
            util.myprint("Starting pre-fetching processes id: "+str(j))
            batch_loader = BatchLoader(params, indexlist_chunks[j], self.phase, j)
            ## Starting the process formally ( it will wait until we event.set() )
            batch_loader.start()
            self.batch_loader_list.append(batch_loader)
        #let train prefetching start immediately
        # if self.phase != 1:
        #     for j in range(0,_nJobs+1):
        #         _eventTrainList[j].set()
        self.collector = Collector( self.batch_loader_list,self.batch_size, self.phase )
        #self.collector.start()
        ## To clean up the process
        ## from https://github.com/rbgirshick/fast-rcnn/blob/master/lib/roi_data_layer/layer.py#L61
        def cleanup():
                util.myprint('Terminating BatchLoader')
                for j in range(0,_nJobs):
                    self.batch_loader_list[j].terminate()
                    self.batch_loader_list[j].join()
                self.collector.terminate()
                self.collector.join()
        import atexit
        atexit.register(cleanup)
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 1 channels (because we have only one label)
        top[1].reshape(self.batch_size, 1)

        if self.phase == 1:
            print_info("FaceAugDataLayer_valid", params)
        else:
            print_info("FaceAugDataLayer_train", params)
    
    def forward(self, bottom, top):
        """
        Load data.
        """
        # global _firstTimeValid
        # global _firstTimeTrain
        # ####### Part to start and stop each pre-fetching process
        # # If we are in validation and is the first time
        # if self.phase == 1 and _firstTimeValid: 
        #     util.myprint('Starting Validation process, stop Training process')
        #     for j in range(0,_nJobs+1):
        #         _eventTrainList[j].clear() #wait
        #         _eventValidList[j].set() #start
        #     _firstTimeValid = False
        #     _firstTimeTrain = True
        # if self.phase == 0 and _firstTimeTrain: 
        #     util.myprint('Starting Training process, stop Validation process')
        #     for j in range(0,_nJobs+1):
        #         _eventTrainList[j].set() #start
        #         _eventValidList[j].clear() #wait
        #     _firstTimeValid = True
        #     _firstTimeTrain = False
        #before = util.current_milli_time()
        # listData = [None]*self.batch_size
        # for j in range(0,_nJobs):
        #     batch_ck_size = self.batch_loader_list[j].batch_ck_size
        #     #ck_data = self.batch_loader_list[j].queue.get()
        #     ck_data = self.batch_loader_list[j].rec_conn.recv()
        #     stt=j*batch_ck_size
        #     endd=stt+batch_ck_size
        #     listData[stt:endd] = ck_data[0:batch_ck_size]
        listData = self.collector.gatherData()
        ######## OK now fill the batch
        for itt in range(self.batch_size):
                top[0].data[itt, ...] = listData[itt]['img']
                top[1].data[itt, ...] = listData[itt]['label']
        #after = util.current_milli_time()
        #util.myprint('Elapsed time ' + str((after-before)) + ' ms')
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass
'''
#class Collector(Process):
class Collector():
    def __init__(self, batch_loader_list, batch_size, phase):
        #super(Collector, self).__init__()
        self.batch_loader_list = batch_loader_list
        #rec_conn, send_conn = Pipe()
        self.queue = Queue(_QSIZE)
        #self.rec_conn = rec_conn
        #self.send_conn = send_conn
        self.batch_size = batch_size
        self.phase = phase
    # def run(self):
    #     j = 0
    #     countStep = 0
    #     listData = [None]*self.batch_size
    #     while True:
    #         # ######## Part to resume/wait a certain process when the other is operating
    #         # if self.phase == 1:
    #         #     if not _eventValidList[_nJobs].is_set():
    #         #         util.myprint('Waiting Collector Collector  to start again')
    #         #         _eventValidList[_nJobs].wait()
    #         # else:
    #         #     if not _eventTrainList[_nJobs].is_set():
    #         #         util.myprint('Waiting Train Collector to start again')
    #         #         _eventTrainList[_nJobs].wait()
    #         ### Doing the job
    #         if j == _nJobs:
    #             j = 0
    #         try:
    #             ck_data = self.batch_loader_list[j].queue.get_nowait()
    #             batch_ck_size = self.batch_loader_list[j].batch_ck_size
    #             stt=countStep
    #             endd=stt+batch_ck_size
    #             listData[stt:endd] = ck_data[0:batch_ck_size]
    #             countStep+=batch_ck_size
    #         except std_Queue.Empty as e:
    #             #util.myprint('Empty queue in collector')
    #             #util.myprint(e.message())
    #             #exit(0)
    #             pass
    #         if countStep == self.batch_size:
    #             #self.send_conn.send(list(listData))
    #             self.queue.put(list(listData))
    #             countStep = 0
    #             listData = [None]*self.batch_size
    #         j+=1
            #self.send_conn.send(listData)
    # def gatherData(self):
    #     #return self.rec_conn.recv()
    #     while True:
    #         try:
    #             return self.queue.get_nowait()
    #         except std_Queue.Empty as e:
    #             pass
    def gatherData(self):
        j = 0
        countStep = 0
        listData = [None]*self.batch_size
        while True:
            # ######## Part to resume/wait a certain process when the other is operating
            # if self.phase == 1:
            #     if not _eventValidList[_nJobs].is_set():
            #         util.myprint('Waiting Collector Collector  to start again')
            #         _eventValidList[_nJobs].wait()
            # else:
            #     if not _eventTrainList[_nJobs].is_set():
            #         util.myprint('Waiting Train Collector to start again')
            #         _eventTrainList[_nJobs].wait()
            ### Doing the job
            if j == _nJobs:
                j = 0
            try:
                ck_data = self.batch_loader_list[j].queue.get_nowait()
                batch_ck_size = self.batch_loader_list[j].batch_ck_size
                stt=countStep
                endd=stt+batch_ck_size
                listData[stt:endd] = ck_data[0:batch_ck_size]
                countStep+=batch_ck_size
            except std_Queue.Empty as e:
                #util.myprint('Empty queue in collector')
                #util.myprint(e.message())
                #exit(0)
                pass
            if countStep == self.batch_size:
                #self.send_conn.send(list(listData))
                #self.queue.put(list(listData))
                return listData
            j+=1
            #self.send_conn.send(listData)                


class BatchLoader(Process):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, indexlist, phase, proc_id):
        super(BatchLoader, self).__init__()
        self.indexlist = indexlist
        self.proc_id = proc_id
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.phase = phase
        self.queue = Queue(_QSIZE)
        #rec_conn, send_conn = Pipe()
        # self.rec_conn = rec_conn
        # self.send_conn = send_conn
        ## Dividing with rest the batch size for the jobs we have
        self.batch_ck_size = self.batch_size//_nJobs
        ## in case of the last jobs adding the rest
        if self.proc_id == (_nJobs - 1):
                self.batch_ck_size += self.batch_size % _nJobs
        ## Opening LMDB
        lmdb_output_pose_env = lmdb.Environment(params['source']+'/pose_lmdb/', readonly=True, lock=False)
        self.cur_pose = lmdb_output_pose_env.begin().cursor()
        lmdb_output_flip_env = lmdb.Environment(params['source']+'/flip_lmdb/', readonly=True, lock=False)
        self.cur_flip = lmdb_output_flip_env.begin().cursor()
        lmdb_output_land_env = lmdb.Environment(params['source']+'/land_lmdb/', readonly=True, lock=False)
        self.cur_land = lmdb_output_land_env.begin().cursor()
        ################
        self.Nimgs = len(self.indexlist)
        # this class does some simple data-manipulations
        #proto_data = open(params['mean_file'], "rb").read()
        
        #a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        #mean  = caffe.io.blobproto_to_array(a)[0]
        ## mean is read BGR and c,h,w; we convert it to h,w,c.
        ## BGR is OK since OpenCV and caffe are BGR
        ## Then MySimpleTransformer will remove mean after that the image
        ## has been changed to BGR as well. So apple-to-apple.
        self.transformer = MySimpleTransformer()
        self.aug_tr = aug_tracker.AugmentationTracker()

        if params['mean_file'] is not None:
            mean = np.load(params['mean_file'])
            mean = mean.transpose(1, 2, 0)
            mean = np.float32(mean)
            self.transformer.set_mean(mean)

        if self.phase == 1:
            util.myprint("BatchLoader_valid" + str(self.proc_id) + " initialized with " + str(self.Nimgs) +" images")
        else:
            util.myprint("BatchLoader_train" + str(self.proc_id) + "  initialized with " + str(self.Nimgs) +" images")
            util.myprint("This will process: " + str(self.batch_ck_size)+'/'+str(self.batch_size) )

    def run(self):
        if self.phase == 1:
            util.myprint("Process started pre-fetching for Validation " + str(self.proc_id) + " : nimgs " + str(self.Nimgs) )
        else:
            util.myprint("Process started pre-fetching for Training " + str(self.proc_id) + " : nimgs " + str(self.Nimgs) )
        ## Counter to the entire augmented set
        count = 0
        ## Counter to the relative mini-batch
        countStep = 0
        ## Pre-allocate the data for the mini-batch
        listData = [None]*self.batch_ck_size
        while True:
            for ii in range(0,self.Nimgs):
                ####### Checking if we finished an (augmented) epoch
                if  count == self.Nimgs:
                    util.myprint("Finished an (augmented) epoch for loader id " + str(self.proc_id) + "...shuffling")
                    count = 0
                    shuffle(self.indexlist)

                # ######## Part to resume/wait a certain process when the other is operating
                # if self.phase == 1:
                #     if not _eventValidList[self.proc_id].is_set():
                #         util.myprint('Waiting Validation Loader ' + str(self.proc_id) + ' to start again')
                #         _eventValidList[self.proc_id].wait()
                # else:
                #     if not _eventTrainList[self.proc_id].is_set():
                #         util.myprint('Waiting Train Loader ' + str(self.proc_id) + ' to start again')
                #         _eventTrainList[self.proc_id].wait()

                ### Starting to do augmentation
                batch_img = None
                #index is of form:
                #blur_fr_13 XXXm.0hhvfrvXXX_MS000024 !!TMPDIR!!/imgs/XXXm.0hhvfrvXXX/XXXm.0hhvfrvXXX_MS000024.jpg 0
                index = self.indexlist[ii]
                index = index.split(' ')
                aug_type = index[0] #augemntation type
                image_key = index[1] # image key
                image_file_name = index[2] #image
                label = np.float32(index[3]) #label
                ## Loading the image with OpenCV
                flipON = int( np.frombuffer( self.cur_flip.get(image_key) )[1] ) == 1
                im = cv2.imread(image_file_name,cv2.CV_LOAD_IMAGE_COLOR)
                ## Check immediately if we have to flip an image
                if flipON:
                    im = cv2.flip(im, 1)
                im_arr = np.asarray(im)
                aug_im = None
                if 'align2d' in aug_type or 'blur' in aug_type:
                    lmark = self.cur_land.get(image_key)
                    lmark = np.frombuffer(lmark, dtype='float64').reshape(68,2)
                    lmarks = np.zeros((1,68,2))
                    lmarks[0] = lmark
                    aug_im = self.aug_tr.augment_fast(aug_type=aug_type,img=im,landmarks=lmarks,flipON=flipON)
                elif 'render' in aug_type:                    
                    prj_matrix = np.frombuffer(self.cur_pose.get(image_key+'_'+aug_type), dtype='float64').reshape(3,4)
                    prj_matrix = np.asmatrix(prj_matrix)
                    aug_im = self.aug_tr.augment_fast(aug_type=aug_type,img=im,prj_matrix=prj_matrix,flipON=flipON)
                try:
                    aug_im = cv2.resize(aug_im, ( self.im_shape[0], self.im_shape[1] ),\
                                          interpolation=cv2.INTER_LINEAR )
                    batch_img = self.transformer.preprocess(aug_im)
                except Exception as ex:
                    util.myprint("Warning: Was not able to use aug_img because: " + str(ex))
                    util.myprint( "Skipping the image: " + image_file_name)
                count += 1
                ##If image have been processes correctly, add it to the mini-batch            
                if batch_img is not None:
                    data = {'img': batch_img , 'label' : label}
                    listData[countStep] = data 
                    countStep+=1
                    if countStep == self.batch_ck_size:
                        isDone = False
                        while not isDone:
                            try:
                                ##This mini-batch is ready to be sent for train
                                ## Resetting the relative listData and countStep
                                self.queue.put_nowait( list(listData) )
                            except std_Queue.Full as full:
                                pass
                            else:
                                #self.send_conn.send( (listData) )
                                countStep = 0
                                isDone = True
                                listData = [None]*self.batch_ck_size
                    # if countStep == self.batch_ck_size:
                    #     ##This mini-batch is ready to be sent for train
                    #     ## Resetting the relative listData and countStep
                    #     self.queue.put( list(listData) )
                    #     #self.send_conn.send( (listData) )
                    #     countStep = 0
                    #     listData = [None]*self.batch_ck_size
def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'im_shape','split','source','mean_file']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape']) 
