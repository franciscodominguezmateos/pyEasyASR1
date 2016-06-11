'''
Created on Jun 11, 2016

@author: Francisco Dominguez
'''

from features import mfcc
from fastdtw import fastdtw
import scipy.io.wavfile as wav
from   scipy.spatial.distance import euclidean
import sklearn.preprocessing as pp
import os

class EasyASR(object):
    '''
    classdocs
    '''

    def __init__(self, filePath="/home/francisco/voz",ldis=0.15,dthres=49):
        '''
        Constructor
        '''
        self.ldis=ldis #length discrimination rate a 15% work fine
        self.distThreshold=dthres #distance threshold 
        basepath=filePath
        self.mfccs={}
        mfccAll=[]
        labels=os.listdir(basepath)
        for i,nn in enumerate(labels):
            n=nn.split(".")[0]#take away extension .wav
            (rate1,sig1)=wav.read(basepath+'/{}.wav'.format(n))
            sig1=pp.maxabs_scale(sig1) #Normalize amplitude to +-1
            mfcc_feat1 = mfcc(sig1,rate1)
            self.mfccs[n]=mfcc_feat1
            
    #return the kNN where k=1 dscriminating by length
    def get1NN(self,mfcc_feat2):
        minrd=1e40
        mkrd="NONE"
        for k in sorted(self.mfccs.keys()):
            mfcc_feat1=self.mfccs[k]
            l1=len(mfcc_feat1)
            l2=len(mfcc_feat2)
            #discriminating by length
            if abs(l1-l2)<l2*self.ldis:
                distance, path = fastdtw(mfcc_feat1, mfcc_feat2, dist=euclidean)
                rd=distance/len(path) #Normalize distance bi path length
                #print k,distance,len(mfcc_feat2),len(mfcc_feat1), len(path),rd
                print k,len(mfcc_feat2),len(mfcc_feat1),rd
                if rd<minrd:
                    minrd=rd
                    mkrd=k
        if minrd>self.distThreshold:
            return "NONE",minrd
        else:
            return mkrd,minrd
    
    def record(self):
        os.system("sox -r 16000 -t alsa default recording.wav silence 1 0.1 1% 1 1.5 1%")

    def recognize(self,fileName):
        (rate2,sig2) = wav.read(fileName)
        sig2=pp.maxabs_scale(sig2) #Normalize amplitude to +-1
        mfcc_feat2 = mfcc(sig2,rate2)
        return self.get1NN(mfcc_feat2)
    
    def recognizeFromMic(self):
        self.record()
        return self.recognize("recording.wav")
    
if __name__ == '__main__':
    asr=EasyASR()
    while True:
        print asr.recognizeFromMic()

        
