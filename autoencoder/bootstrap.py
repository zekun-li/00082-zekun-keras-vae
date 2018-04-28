import sys
import os
#import utils
import glob
import math
import random
import fileinput
from sh import tail
import subprocess
import shutil
import multiprocessing as mp

#dataTars_string = '/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran_prepare_poseCNN_COW2/expts/COW_data'
dataTars_string = '/nfs/isicvlnas01/projects/glaive/expts/00082-zekun-vggdata/full'
dataTars = glob.glob( dataTars_string + '/*.tar' )

#precompute_data = '/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran_clean_MsCeleb_poseCNN/expts/final_data/'
precompute_data = '/nfs/isicvlnas01/projects/glaive/expts/00082-zekun-keras-vae/final_data/'
TMPDIR = os.environ['TMPDIR']+'/'

'''
log_file = sys.argv[1]
print ('log saved to:',log_file)
'''
def myprint(sometext, printON=False):
    if printON:
        print sometext
        sys.stdout.flush()

myprint( 'Working in ' +  TMPDIR )

def mycmd(cmd, printON=True):
    if printON:
        myprint( cmd )
    os.system(cmd)

def mymkdir(dire):
    if not os.path.exists( dire):
        os.mkdir( dire )

def getNameDirTag(stri):
    fmt = stri.split('/')[-1].split('.')[-1]
    lists = stri.split('/')[-1].split('.' + fmt)[0].split('_')
    stringg=''
    for i in range(0,len(lists)-1):
        stringg+=lists[i]+'_'
    return stringg[0:-1]

def getSubjTag(stri):
    lists = stri.split('_')[0:-1]
    stringg=''
    for i in range(0,len(lists)):
        stringg+=lists[i]+'_'
    return stringg[0:-1]


def getName(stri):
    fmt = stri.split('/')[-1].split('.')[-1]
    return stri.split('/')[-1].split('.' + fmt)[0].split('_')[-1]

def replaceInFile(filep, before, after):
    for line in fileinput.input(filep, inplace=True):
        print line.replace(before,after),


def moveData(data):
    chunk_id = getName(data).zfill(4)
    myprint('Moving chunck ' + chunk_id)
    #mymkdir(TMPDIR + chunk_id)
    mycmd('cp ' + data + ' ' + TMPDIR)
#   utils.mycmd( 'tar --strip-components=7 -C ' + REF_TMPDIR + chunk_id + ' -xf '+ REF_TMPDIR + '/' + data.split('/')[-1] + ' --exclude \'*.txt\'' )
    mycmd( 'tar -C ' + TMPDIR + ' -xf '+ TMPDIR + data.split('/')[-1] )
    #mycmd( 'tar -C ' + TMPDIR + chunk_id + ' -xf '+ TMPDIR + data.split('/')[-1] )
    #mycmd( 'rm -r ' +  TMPDIR + '/' + chunk_id + '/landmarks/')
    #mycmd( 'rm -r ' +  TMPDIR + '/' + chunk_id + '/confidence/')
    if os.path.exists( TMPDIR + '/' + data.split('/')[-1] ):
       mycmd( 'rm ' + TMPDIR + '/' + data.split('/')[-1] )

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
        myprint('removed dir ' + TMPDIR +  dirr)
    myprint('copying ' + dirr)
    shutil.copytree(precompute_data + dirr, TMPDIR + dirr)
#mycmd('cp ' + precompute_data + 'train_valid.tar.gz '+ TMPDIR)
#mycmd('tar -xf ' +  TMPDIR + 'train_valid.tar.gz -C '+ TMPDIR)
#mycmd('rm ' + TMPDIR +  'train_valid.tar.gz')
mycmd('cp ' + precompute_data + 'ox_train_align2d.list ' + TMPDIR+'train.list' )
mycmd('cp ' + precompute_data + 'ox_valid_align2d.list ' + TMPDIR+'valid.list' )
replaceInFile(TMPDIR + '/valid.list', '!!TMPDIR!!', TMPDIR)
replaceInFile(TMPDIR + '/train.list', '!!TMPDIR!!', TMPDIR)

'''
#train_string = 'source ~/.bashrc; source ~srawls/.bashrc; python2 train_fixval.py'
train_string = 'source ~/.bashrc;  python2 train_fixval_conv14.py'
myprint( train_string )
train_bash = open(TMPDIR+'train_bash.sh','w')
output_file = open('logs'+log_file,'w')
train_bash.write('#!/bin/bash\n'+ train_string+'\n')
train_bash.close()
#utils.mycmd( 'cp '+ TMPDIR + '/train_bash.sh ' + output)
mycmd('chmod +x ' + TMPDIR + 'train_bash.sh')
subprocess.call(TMPDIR+"train_bash.sh", shell=True, executable='/bin/bash',stdout=output_file, stderr=output_file)
output_file.close()
'''
