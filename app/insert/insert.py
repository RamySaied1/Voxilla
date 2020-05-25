from config import Config
import sys
sys.path.append(Config.INSERT_DIR)
from encoder.params_model import model_embedding_size as speaker_embedding_size
from lib.synthesizer.inference import Synthesizer
from encoder import inference as encoder
from lib.vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch

class Insert():
    def __init__(self,encoderModelPath,synthesizerModelPath,vocoderModelPath):
        encoder.load_model(encoderModelPath)
        self.synthesizer=Synthesizer(synthesizerModelPath.joinpath("taco_pretrained"))
        vocoder.load_model(vocoderModelPath)

    def getWav(self,referenceVoiceWavPath,words):
        preprocessed_wav = encoder.preprocess_wav(referenceVoiceWavPath)
        embed = encoder.embed_utterance(preprocessed_wav)
        embeds=[embed]
        specs = self.synthesizer.synthesize_spectrograms(words, embeds)
        spec = specs[0]
        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
        return generated_wav,self.synthesizer.sample_rate