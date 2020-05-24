'''
This File reads dataset and saves it in list with shaper (num_speakers,num_utterances)
'''
import os
import soundfile as sf

def readDataset(path):
    speakersFolders=next(os.walk(path))[1]
    for x,dirr,z in os.walk(path):
        print(dirr,x,z)
    #print(next(os.walk(path)))
    dataset=[]
    for speaker in speakersFolders:
        utterancesNames=next(os.walk(path))[2]
        utteranceList=[]
        for utteranceName in utterancesNames:
            wavFile=sf.read(f"{path}/{speaker}/{utteranceName}")
            utteranceList.append(wavFile)
        dataset.append(utteranceList)
    return dataset

