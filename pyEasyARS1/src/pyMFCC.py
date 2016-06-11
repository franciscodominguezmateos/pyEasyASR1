'''
Created on Apr 25, 2016

@author: Francisco Dominguez
'''
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy.linalg as la
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import scale
import sklearn.preprocessing as pp
from sklearn.cluster import KMeans
import cv2
import path
import os

basepath="/home/francisco/voz"
kMns=KMeans(n_clusters=25)
mfccs={}
mfccAll=[]
labels=os.listdir(basepath)
for i,nn in enumerate(labels):
    n=nn.split(".")[0]
    (rate1,sig1)=wav.read(basepath+'/{}.wav'.format(n))
    sig1=pp.maxabs_scale(sig1)
    mfcc_feat1 = mfcc(sig1,rate1)
    #mfcc_feat1=scale(mfcc_feat1)#Standarizar?
    #for f in mfcc_feat1:
    #    mfccAll.append(f)
    mfccs[n]=mfcc_feat1
#mfccAll=np.array(mfccAll)
#print "mfccAll",mfccAll.shape
#mfccAll=scale(mfccAll)#whitenning,standarar
#kMns.fit(mfccAll)
#print kMns.predict(mfccs["paco_no_001"])
#print kMns.predict(mfccs["paco_uno_001"])

os.system("sox -r 16000 -t alsa default recording.wav silence 1 0.1 1% 1 1.5 1%")
(rate2,sig2) = wav.read("recording.wav")
#sig2=pp.maxabs_scale(sig2)
sig2=pp.maxabs_scale(sig2)#
mfcc_feat2 = mfcc(sig2,rate2)
#mfcc_feat2=scale(mfcc_feat2)#Standarizar?

mind=1e40
minrd=1e40
for k in sorted(mfccs.keys()):
    mfcc_feat1=mfccs[k]
    distance, path = fastdtw(mfcc_feat1, mfcc_feat2, dist=euclidean)
    rd=distance/len(path)
    #print k,distance, len(path),rd
    if distance<mind:
        mind=distance
        mk=k
    if rd<minrd:
        minrd=rd
        mkrd=k
print mk,mind
print mkrd,minrd

#cmd="cp recording.wav /home/francisco/voz/{}.wav".format(mk+"0")
#os.system(cmd)

mfcc_feat1=mfccs[mk]
d=np.zeros((mfcc_feat1.shape[0],mfcc_feat2.shape[0]))
for i in range(mfcc_feat1.shape[0]):
    for j in range(mfcc_feat2.shape[0]):
        dist=la.norm(mfcc_feat1[i]-mfcc_feat2[j])
        d[i,j]=dist
dn=d/abs(d).max()
dif=la.norm(mfcc_feat1)
#print mfcc_feat1.shape,mfcc_feat2.shape
#fbank_feat1 = logfbank(sig1,rate1)

distance, path = fastdtw(mfcc_feat1, mfcc_feat2, dist=euclidean)
#print distance, len(path),distance/len(path)
for x,y in path:
    dn[x,y]=1
cv2.imshow(mk,dn)
cv2.waitKey()

if __name__ == '__main__':
    pass