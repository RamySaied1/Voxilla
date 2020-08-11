from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder.audio import preprocess_wav   # We want to expose this function from here
from matplotlib import cm
from encoder import audio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

_model = None # type: SpeakerEncoder
_device = None # type: torch.device

#load_model("./app/insert/encoder/saved_models/pretrained.pt")
def load_model(weights_fpath: Path, device=None):
    '''
    This Method loads the model 
    PARAMS:
    weights_fpath: The path to the weights of the pretrained model(Must be a Path object)   
    '''
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    _model =SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint["model_state"])
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath, checkpoint["step"]))
    


def computeEmbeddingForBatch(framesInBatches):
    '''
    This method computes the embedding for a batch of partial utterances
    it done so by feeding the batch of frames(in mel spectogram) 
    to the network
    '''
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    frames = torch.from_numpy(framesInBatches).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed


def computeSlices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    '''
    This Method computes for the utterance and the mel spectorgram that will be obtained from it,
    the begining and end of every slice.
    This method returns two lists of slices:
    1-wav slices:where to slice the wav every element of this list is a slice object 
    2-mel slices:where to slice the mel 
    '''
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    numSamplesInFrame = int((sampling_rate * mel_window_step / 1000))#This number is divided by 1000 because the unit in sampling rate is sample/sec
    totalFrames = int(np.ceil((n_samples + 1) / numSamplesInFrame))#total number of frames
    frameStep = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)#This variable represents the step we need to move from one partial utterance to the other one
    # Compute the slices
    wSlices, mSlices = [], []#a list that will contain the slices of the wav and mel spectograms respectively
    numPartialUtterances = max(1, totalFrames - partial_utterance_n_frames + frameStep + 1)#this variable represents the number of partial utterances in the wav
    for i in range(0, numPartialUtterances, frameStep):
        mRange = np.array([i, i + partial_utterance_n_frames])# "i" represents the start frame of this partial utterance and i+partial_utterance_n_frames represents the end frame
        wRange = mRange * numSamplesInFrame #the start of this range is the start sample number and end is the end sample number 
        mSlices.append(slice(*mRange))
        wSlices.append(slice(*wRange))
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wSlices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mSlices) > 1:
        mSlices = mSlices[:-1]
        wSlices = wSlices[:-1]
    return wSlices, mSlices


def computeEmbedding(wav, **kwargs):
    '''
    This Method computes the embedding vector for the wav paramater

    PARAMS:
    wav:the preprocessed wav file for which the e-vector will be calculated

    RETURNS:
    the embedding of the wav object 
    '''
   
    # If the last slice size is larger than the length of the wav then we 
    #must zero-pad the wav 
    wSlices, mSlices = computeSlices(len(wav), **kwargs)
    lastSliceStop = wSlices[-1].stop
    if lastSliceStop >= len(wav):
        wav = np.pad(wav, (0, lastSliceStop - len(wav)), "constant")
    
    #compute the mel spectogram of the wav
    frames = audio.wav_to_mel_spectrogram(wav)
    #group every mslice into an array which will be fed to the network
    framesInBatches = np.array([frames[s] for s in mSlices])
    #for every member in partialEmbeddings is the e-vector for the partial utterance 
    partialEmbeddings = computeEmbeddingForBatch(framesInBatches)
    # The embedding vector of the complete utterance will be the normalization of the averaged version
    averageEmbedding = np.mean(partialEmbeddings, axis=0)
    embed = averageEmbedding / np.linalg.norm(averageEmbedding, 2)
    
    return embed



