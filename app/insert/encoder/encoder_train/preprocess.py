'''
This File takes as input a list with shape (Num speaker,Num utterances for that speaker ) 
each row will be all the utterances said by a specific speaker
then this file will  save the processed files in processed_data folder 
'''

import sys
import os
import re
sys.path.insert(0,"../../FeatureExtraction")

from speech_sigproc import FrontEnd
import argparse
import soundfile as sf
import numpy as np

def saveSpeakerUtterances(utteranceList,speakerNumber,outputPath):
    os.makedirs(f"{outputPath}/{speakerNumber}")
    utteranceNumber=0
    for utterance in utteranceList:
        #f=open(f"{outputPath}/{speakerNumber}/{utteranceNumber}.feat",'w')
        #f.writelines(utterance)
        #f.close()
        np.savetxt(f"{outputPath}/{speakerNumber}/{utteranceNumber}.feat",utterance)
        utteranceNumber=utteranceNumber+1

def preprocessSpeaker(utteranceList):
    #create frontend object that contain methods for computing mel spectogram
    frontend=FrontEnd()
    output=[]
    #loop on all utterances
    for utterance in utteranceList:
        mel=frontend.process_utterance(utterance)
        output.append(mel)
    return output

def preprocessDataset(dataset,outputPath):
    #Loop on all speakers and for every speaker save all it's utterances in processed_data folder
    speakerNumber=0
    for speaker_utterances in dataset:   
        processedSpeaker=preprocessSpeaker(speaker_utterances)
        saveSpeakerUtterances(processedSpeaker,speakerNumber,outputPath)
        speakerNumber=speakerNumber+1
def readDataset(path):
    dataset=[]
    speakersNames=next(os.walk(f"{path}/"))[1]
    print(len(speakersNames))
    for speaker in speakersNames:
        speakerUtterancesFolders=next(os.walk(f"{path}/{speaker}"))[1]
        output=[]
        for speakerUtterancesFolder in speakerUtterancesFolders:
            files=next(os.walk(f"{path}/{speaker}/{speakerUtterancesFolder}"))[2]
            for file in files:
                #Check if it's .wav file
                if  re.search(".flac",file):
                    x,_=sf.read(f"{path}/{speaker}/{speakerUtterancesFolder}/{file}")
                    output.append(x)
        dataset.append(output)
    return dataset




def main(args):
    dataset=readDataset(args.inputPath)
    preprocessDataset(dataset,args.outputPath)

if  __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputPath",type=str,required=True,help="Input Path for the dataset")
    parser.add_argument("--outputPath",type=str,required=True,help="Output Path for dataset")
    args = parser.parse_args()
    main(args)

