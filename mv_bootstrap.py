import sys
import os
import utils
import glob
import math
import random
import fileinput
from sh import tail
import subprocess
import shutil
import multiprocessing as mp

dataTars_string = '/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran_prepare_poseCNN_COW2/expts/COW_data'
dataTars = glob.glob( dataTars_string + '/*.tar' )

precompute_data = '/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran_clean_MsCeleb_poseCNN/expts/final_data/'
TMPDIR = os.environ['TMPDIR']+'/'

log_file = '/02_cow_sum_nomean_dim2048_log.training'

def myprint(sometext, printON=False):
    if printON:
        print sometext
        sys.stdout.flush()

utils.myprint( 'Working in ' +  TMPDIR )


def moveData(data):
    chunk_id = utils.getName(data).zfill(4)
    utils.myprint('Moving chunck ' + chunk_id)
    utils.mymkdir(TMPDIR + chunk_id)
    utils.mycmd('cp ' + data + ' ' + TMPDIR)
#   utils.mycmd( 'tar --strip-components=7 -C ' + REF_TMPDIR + chunk_id + ' -xf '+ REF_TMPDIR + '/' + data.split('/')[-1] + ' --exclude \'*.txt\'' )
    utils.mycmd( 'tar -C ' + TMPDIR + chunk_id + ' -xf '+ TMPDIR + data.split('/')[-1] )
    utils.mycmd( 'rm -r ' +  TMPDIR + '/' + chunk_id + '/landmarks/')
    utils.mycmd( 'rm -r ' +  TMPDIR + '/' + chunk_id + '/confidence/')
    if os.path.exists( TMPDIR + '/' + data.split('/')[-1] ):
       utils.mycmd( 'rm ' + TMPDIR + '/' + data.split('/')[-1] )

#if False:
if True:
    #pool = mp.Pool(processes=+processes+) #use all available cores, otherwise specify the number you want as an argument
    for data in dataTars:
        moveData(data)
    #pool.close()
    #pool.join()

for dirr in ['flip_lmdb','pose_lmdb','land_lmdb']:
    if os.path.exists(TMPDIR + dirr):
        shutil.rmtree(TMPDIR + dirr)
        utils.myprint('removed dir ' + TMPDIR +  dirr)
    utils.myprint('copying ' + dirr)
    shutil.copytree(precompute_data + dirr, TMPDIR + dirr)
utils.mycmd('cp ' + precompute_data + 'train_valid.tar.gz '+ TMPDIR)
utils.mycmd('tar -xf ' +  TMPDIR + 'train_valid.tar.gz -C '+ TMPDIR)
utils.mycmd('rm ' + TMPDIR +  'train_valid.tar.gz')
utils.replaceInFile(TMPDIR + '/valid.list', '!!TMPDIR!!', TMPDIR)
utils.replaceInFile(TMPDIR + '/train.list', '!!TMPDIR!!', TMPDIR)
