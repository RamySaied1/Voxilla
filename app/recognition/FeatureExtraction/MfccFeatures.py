from scipy.ndimage.interpolation import shift

import numpy as np

class MFCC:
    def __init__(self, sampling_rate=16000, frame_duration=0.025, frame_shift=0.010,power_feat=False,delta_feat=False,average_feat=False, pre_emphasis=0.97,
                 num_mel_filters=40,high_freq=None, low_freq=0,normalize_feat=True, mean_norm_wav=True, compute_global_stats=False):
        self.sampling_rate = sampling_rate
        self.pre_emphasis = pre_emphasis
        self.num_mel_filters = num_mel_filters
        self.global_mean_vector = np.zeros([num_mel_filters])
        self.global_var_vector = np.zeros([num_mel_filters])
        self.num_global_frames = 0
        self.compute_global_stats = compute_global_stats


        self.window_size = int(np.floor(frame_duration * sampling_rate))
        self.fft_len = 2
        while (self.fft_len<self.window_size):
            self.fft_len *= 2
            
        self.window_shift = int(np.floor(frame_shift * sampling_rate))
        if (high_freq == None):
            self.high_freq = sampling_rate//2
        else:
            self.high_freq = high_freq

        self.low_freq = low_freq


        self.hamming = np.hamming(self.window_size)

        self.make_melFilterBank()
        self.mean_normalize = normalize_feat
        self.wav_zero_mean = mean_norm_wav


    def make_melFilterBank(self):

        lo_mel = self.freq2mel(self.low_freq)
        hi_mel = self.freq2mel(self.high_freq)

        mel_freqs = np.linspace(lo_mel, hi_mel,self.num_mel_filters+2)


        bin_width = self.sampling_rate/self.fft_len 
        bins_mel = np.floor(self.mel2freq(mel_freqs)/bin_width)

        num_bins = self.fft_len//2 + 1
        self.filterbank = np.zeros([self.num_mel_filters,num_bins])
        for i in range(0,self.num_mel_filters):
            bin_left = int(bins_mel[i])
            bin_center = int(bins_mel[i+1])
            bin_right = int(bins_mel[i+2])
            up_slope = 1/(bin_center-bin_left)
            for j in range(bin_left,bin_center):
                self.filterbank[i,j] = (j - bin_left)*up_slope
            down_slope = -1/(bin_right-bin_center)
            for j in range(bin_center,bin_right):
                self.filterbank[i,j] = (j-bin_right)*down_slope


    def dithering(self, wav):
        n = 2*np.random.rand(wav.shape[0])-1
        n *= 1/(2**15)
        return wav + n

    def pre_emphasize_signal(self, wav):
        preemph_wav = []
        preemph_wav=np.append(wav[0], wav[1:] - self.pre_emphasis * wav[:-1])
        return preemph_wav

    def audio_to_frames(self, wav):
        # only process whole frames
        num_frames = self.num_frames
        frames = np.zeros([self.window_size, num_frames])
        for t in range(0, num_frames):
            frame = wav[t * self.window_shift:t * self.window_shift + self.window_size]
            if (self.wav_zero_mean):
                frame = frame - np.mean(frame)
            frames[:, t] = self.hamming * frame
        return frames

    def frames_to_spectrum(self, frames):
        magspec = np.zeros([self.fft_len//2+1,len(frames[0])])
        frames = frames.transpose()
        for i in range (0,len(frames)):
            frame_fft=np.fft.fft(frames[i])
            frame_fft=np.abs(frame_fft[0:(self.fft_len//2+1)])
            magspec[:,i]=frame_fft

        return magspec

    def magspec_to_mel(self, magspec):
        fbank=self.filterbank.dot(magspec)
        fbank=np.log(fbank)
        return fbank

    def mean_norm_features(self, fbank):
        mean=np.mean(fbank,axis=1)
        fbank=fbank-mean[:,None]
        return fbank

    def accumulate_stattistics(self, features):
        self.global_mean_vector += np.sum(features,axis=1)
        self.global_var_vector += np.sum(features**2,axis=1)
        self.num_global_frames += features.shape[1]

    def calculate_statistics(self):
        self.global_mean_vector /= self.num_global_frames
        self.global_var_vector /= self.num_global_frames
        self.global_var_vector -= self.global_mean_vector**2

        return self.global_mean_vector, 1.0/np.sqrt(self.global_var_vector)

    def delta(self,feature):
      tm1=shift(feature,[0,1],mode='nearest')
      tp1=shift(feature,[0,-1],mode='nearest')
      tm2=shift(feature,[0,2],mode='nearest')
      tp2=shift(feature,[0,-2],mode='nearest')
      denomenator=10
      result=((tp1-tm1)+2*(tp2-tm2))/denomenator
      return result
    
    def energy(self,frames):
      result = np.sum(frames**2,axis=0)
      result= result.reshape((1,-1))
      return result

    def average(self,features):
      result=np.sum(features,axis=0)/features.shape[0]
      result= result.reshape((1,-1))
      return result


    def calculate_features(self, audio):
        wave     = self.dithering(audio)
        self.num_frames = int(np.floor((wave.shape[0] - self.window_size) / self.window_shift) + 1)
        wave     = self.pre_emphasize_signal(wave)
        frames  = self.audio_to_frames(wave)
        magspec = self.frames_to_spectrum(frames)
        features   = self.magspec_to_mel(magspec)
        if (self.mean_normalize and self.num_frames > 1):
            features = self.mean_norm_features(features)

        if (self.compute_global_stats):
            self.accumulate_stattistics(features)

        return features

    def freq2mel(self,frequency):
        return 2595*np.log10(1+frequency/700)

    def mel2freq(self,mells):
        return (10**(mells/2595)-1)*700

