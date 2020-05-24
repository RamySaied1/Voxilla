'''
This Files takes as input the processed dataset and with shape (num_speakers,num_utterances_per_speaker,utterance_size)
It selects randomly a batch of size (N*M)
according to GE2E we use partial utterances of size [140:180] 
This implementation uses partial utterances of size 150 or 5 frames from the utterance 
'''
import numpy as np
import random
import os
from paramaters import *
from tensorflow import convert_to_tensor

def readUtterances(speakerNumber,path):
    print(f"{path}/{speakerNumber}")
    utterancesNames=next(os.walk(f"{path}/{speakerNumber}"))[2]
    output=[]
    for utteranceName in utterancesNames:
        #f=open(f"{path}/{speakerNumber}/{utteranceName}")
        fbank=np.loadtxt(f"{path}/{speakerNumber}/{utteranceName}")
        #f.readlines()
        output.append(fbank)
    return output

def generateBatch(N,M,numSpeakers,path):
    #N:number of speakers in a single batch
    #M:number of utterances from a speaker in a single batch
    #200:is the number of features of the 5 frames concatanated 
    batch=np.zeros((N,M,200))
    #get random N speaker numbers
    speakerNumbers=random.sample(range(1,numSpeakers),N)
    N_Index=0
    M_Index=0
    #for every speaker get M partial utterances
    for speakerNumber in speakerNumbers:
        utteranceList=readUtterances(speakerNumber,path)
        utteranceNumbers=random.sample(range(1,len(utteranceList)),M)
        #for every utterance get 5  frames to form a partial utterance
        for utteranceNumber in utteranceNumbers:
            M_Index=0
            offsetNumber=random.sample(range(0,len(utteranceList[utteranceNumber])-5),1)[0]
            partialUtterance=utteranceList[utteranceNumber][:,offsetNumber:offsetNumber+5]
            partialUtterance=np.concatenate(partialUtterance)
            batch[N_Index][M_Index]=partialUtterance
            M_Index=M_Index+1
        N_Index=N_Index+1
    return  batch


if  __name__=="__main__":
    print(generateBatch(2,2,5,"out").shape)
