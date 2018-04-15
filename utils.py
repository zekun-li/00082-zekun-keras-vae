import os
import sys
import glob
import math
import random
import fileinput
from sh import tail
import subprocess
from PIL import Image
import numpy as np
import cv2

def myprint(sometext, flushON=True):
    print sometext
    if flushON:
        sys.stdout.flush()

## FUNCs ################################
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


