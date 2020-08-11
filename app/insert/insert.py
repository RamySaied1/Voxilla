from config import Config
import sys
sys.path.append(Config.INSERT_DIR)
from encoder.params_model import model_embedding_size as speaker_embedding_size
from lib.synthesizer.inference import Synthesizer
from encoder import inference as encoder
from lib.vocoder import inference as vocoder
from encoder.audio import trim_long_silences
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import soundfile as sf
def amplifySound(generated_wav,audioDir):
    print(audioDir)
    audio,_ = sf.read(audioDir) 
    print("H1")
    origialAudioAverage=audio.mean()
    print("H2")
    newWordAverage=audio.mean()
    print("H3")
    amplificationFactor=newWordAverage/origialAudioAverage
    print("H4")
    generated_wav=amplificationFactor*generated_wav
    print("222222222222222")
    return generated_wav    
  
class Insert():
    def __init__(self,encoderModelPath,synthesizerModelPath,vocoderModelPath):
        encoder.load_model(encoderModelPath)
        self.synthesizer=Synthesizer(synthesizerModelPath.joinpath("taco_pretrained"))
        vocoder.load_model(vocoderModelPath)

    def getWav(self,referenceVoiceWavPath,words):
        print("getWav1")
        preprocessed_wav = encoder.preprocess_wav(referenceVoiceWavPath)
        embed = encoder.computeEmbedding(preprocessed_wav)
        print("getWav2")
        embeds=[embed]
        specs = self.synthesizer.synthesize_spectrograms(words, embeds)
        spec = specs[0]
        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
        generated_wav=trim_long_silences(generated_wav)
        return generated_wav,self.synthesizer.sample_rate