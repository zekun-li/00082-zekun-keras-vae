import sys
import os
import glob
import math
import random
import fileinput
from sh import tail
import subprocess
import multiprocessing as mp

dataTars_string = '/nfs/isicvlnas01/projects/glaive/expts/00036-anhttran_prepare_poseCNN_COW2/expts/COW_data'
dataTars = glob.glob( dataTars_string + '/*.tar' )
#TMPDIR = os.environ['TMPDIR']+'/'
TMPDIR = os.getcwd()+'/'

def myprint(sometext, printON=False):
    if printON:
        print sometext
        sys.stdout.flush()

def mycmd(cmd, printON=True):
    if printON:
        myprint( cmd )
    os.system(cmd)

def mymkdir(dire):
    if not os.path.exists( dire):
        os.mkdir( dire )

def getName(stri):
    fmt = stri.split('/')[-1].split('.')[-1]
    return stri.split('/')[-1].split('.' + fmt)[0].split('_')[-1]

myprint( 'Working in ' +  TMPDIR )

def moveData(data):
    chunk_id = getName(data).zfill(4)
    myprint('Moving chunck ' + chunk_id)
    mymkdir(TMPDIR + chunk_id)
    mycmd('cp ' + data + ' ' + TMPDIR)
#   utils.mycmd( 'tar --strip-components=7 -C ' + REF_TMPDIR + chunk_id + ' -xf '+ REF_TMPDIR + '/' + data.split('/')[-1] + ' --exclude \'*.txt\'' )
    mycmd( 'tar -C ' + TMPDIR + chunk_id + ' -xf '+ TMPDIR + data.split('/')[-1] )
    mycmd( 'rm -r ' +  TMPDIR + '/' + chunk_id + '/landmarks/')
    mycmd( 'rm -r ' +  TMPDIR + '/' + chunk_id + '/confidence/')
    if os.path.exists( TMPDIR + '/' + data.split('/')[-1] ):
       mycmd( 'rm ' + TMPDIR + '/' + data.split('/')[-1] )

#if False:
if True:
    #pool = mp.Pool(processes=+processes+) #use all available cores, otherwise specify the number you want as an argument
    for data in dataTars:
        moveData(data)
