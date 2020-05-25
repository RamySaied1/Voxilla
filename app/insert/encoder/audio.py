from scipy.ndimage.morphology import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import webrtcvad
import librosa
import struct

int16_max = (2 ** 15) - 1


class MelSpectogram:
    def __init__(self,samp_rate=16000 ,frame_duration=0.025, frame_shift=0.010,
                 num_mel=40, power=2 ,lo_freq=0, hi_freq=None):
        self.samp_rate = samp_rate
        self.win_size = int(np.floor(frame_duration * samp_rate))
        self.win_shift = int(np.floor(frame_shift * samp_rate))
        self.lo_freq = lo_freq
        if (hi_freq == None):
            self.hi_freq = samp_rate//2
        else:
            self.hi_freq = hi_freq

        self.frame_shift=frame_shift
        self.num_mel = num_mel
        self.fft_size= self.win_size
        self.hamwin = np.hamming(self.win_size)
        self.make_mel_filterbank()
        self.power=power


    def mel2f(self,mels):

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region

        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs


    def lin2mel(self,frequencies):
        f_min = 0.0
        f_sp = 200.0 / 3


        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region

        print(min_log_hz,min_log_mel,logstep)

        mels=0
        if frequencies >= min_log_hz:
          mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep


        return mels


    def make_mel_filterbank(self):

        filter_bank = np.zeros((self.num_mel, int(1 + self.fft_size // 2)), dtype=np.float32)

        min_mel = self.lin2mel(self.lo_freq)
        max_mel = self.lin2mel(self.hi_freq)
        mel_f = np.linspace(min_mel, max_mel, self.num_mel + 2)
        mel_f=self.mel2f(mel_f)

        fftfreqs = [((i*self.samp_rate)/self.fft_size) for i in range(0,int(1+self.fft_size/2))]
        fftfreqs = np.array(fftfreqs)

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(self.num_mel):
            lower_value = -ramps[i] / fdiff[i]
            upper_value = ramps[i+2] / fdiff[i+1]
            filter_bank[i] = np.maximum(0, np.minimum(lower_value, upper_value))

        enorm = 2.0 / (mel_f[2:self.num_mel+2] - mel_f[:self.num_mel])
        filter_bank *= enorm[:, np.newaxis]

        self.mel_filterbank=filter_bank
        return filter_bank


    # for each frame (column of 2D array 'frames'), compute the magnitude spectrum using the fft
    def frames_to_magspec(self, frames):
        hop_length=int(self.samp_rate * self.frame_shift)
        magspec = np.abs(librosa.stft(frames, n_fft=self.fft_size, hop_length=hop_length,))**self.power
        return magspec

    # for each frame(column of 2D array 'magspec'), compute the log mel spectrum, by applying the mel filterbank to the magnitude spectrum
    def magspec_to_fbank(self, magspec):
        fbank=self.mel_filterbank.dot(magspec)
        return fbank
   
    def process_utterance(self, wav):
        magspec = self.frames_to_magspec(wav)
        fbank   = self.magspec_to_fbank(magspec)
        return fbank




def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    fe=MelSpectogram(samp_rate=sampling_rate,frame_duration=(mel_window_length / 1000.0),frame_shift=(mel_window_step / 1000.0),num_mel=mel_n_channels)
    return fe.process_utterance(wav).T


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))
