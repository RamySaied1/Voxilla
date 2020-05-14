import numpy as np
from scipy.ndimage.interpolation import shift
class FrontEnd:

    def __init__(self,samp_rate=16000, frame_duration=0.025, frame_shift=0.010, preemphasis=0.97,delta=0,
                 num_mel=40, average=0, power=0 ,lo_freq=0, hi_freq=None, mean_norm_feat=True, mean_norm_wav=True, compute_stats=False):
        self.samp_rate = samp_rate
        self.win_size = int(np.floor(frame_duration * samp_rate))
        self.win_shift = int(np.floor(frame_shift * samp_rate))
        self.lo_freq = lo_freq
        if (hi_freq == None):
            self.hi_freq = samp_rate//2
        else:
            self.hi_freq = hi_freq

        self.preemphasis = preemphasis
        self.num_mel = num_mel
        self.fft_size = 2
        while (self.fft_size<self.win_size):
            self.fft_size *= 2

        self.hamwin = np.hamming(self.win_size)

        self.make_mel_filterbank()
        self.mean_normalize = mean_norm_feat
        self.zero_mean_wav = mean_norm_wav
        self.global_mean = np.zeros([((num_mel+power)*(delta+1))+average])
        self.global_var  = np.zeros([((num_mel+power)*(delta+1)+average)])
        self.global_frames = 0
        self.compute_global_stats = compute_stats
        self.is_power=power
        self.is_delta=delta
        self.is_average=average
    # linear-scale frequency (Hz) to mel-scale frequency
    def lin2mel(self,freq):
        return 2595*np.log10(1+freq/700)

    # mel-scale frequency to linear-scale frequency
    def mel2lin(self,mel):
        return (10**(mel/2595)-1)*700

    def make_mel_filterbank(self):

        lo_mel = self.lin2mel(self.lo_freq)
        hi_mel = self.lin2mel(self.hi_freq)

        # uniform spacing on mel scale
        mel_freqs = np.linspace(lo_mel, hi_mel,self.num_mel+2)

        #print("mel FREQ",mel_freqs.shape)

        # convert mel freqs to hertz and then to fft bins
        bin_width = self.samp_rate/self.fft_size # typically 31.25 Hz, bin[0]=0 Hz, bin[1]=31.25 Hz,..., bin[256]=8000 Hz
        mel_bins = np.floor(self.mel2lin(mel_freqs)/bin_width)

        num_bins = self.fft_size//2 + 1
        self.mel_filterbank = np.zeros([self.num_mel,num_bins])
        #print("mul size,",self.mel_filterbank.shape)
        for i in range(0,self.num_mel):
            left_bin = int(mel_bins[i])
            center_bin = int(mel_bins[i+1])
            right_bin = int(mel_bins[i+2])
            up_slope = 1/(center_bin-left_bin)
            for j in range(left_bin,center_bin):
                self.mel_filterbank[i,j] = (j - left_bin)*up_slope
            down_slope = -1/(right_bin-center_bin)
            for j in range(center_bin,right_bin):
                self.mel_filterbank[i,j] = (j-right_bin)*down_slope

        #print("mel",num_bins)

    def plot_mel_matrix(self):
        for i in range(0, self.num_mel):
            plt.plot(self.mel_filterbank[i,:])
        plt.show()

    def dither(self, wav):
        n = 2*np.random.rand(wav.shape[0])-1
        n *= 1/(2**15)
        return wav + n

    # replace this with proper pre-emphasis filtering, using the self.preemphasis coefficient
    def pre_emphasize(self, wav):
        # apply pre-emphasis filtering on waveform
        preemph_wav = []
        preemph_wav=np.append(wav[0], wav[1:] - self.preemphasis * wav[:-1])
        return preemph_wav

    def wav_to_frames(self, wav):
        # only process whole frames
        num_frames = int(np.floor((wav.shape[0] - self.win_size) / self.win_shift) + 1)
        #print(num_frames,"num frames")
        frames = np.zeros([self.win_size, num_frames])
        for t in range(0, num_frames):
            frame = wav[t * self.win_shift:t * self.win_shift + self.win_size]
            #print(frame)
            if (self.zero_mean_wav):
                frame = frame - np.mean(frame)
            frames[:, t] = self.hamwin * frame
        #print(frames)
        return frames

    # for each frame (column of 2D array 'frames'), compute the magnitude spectrum using the fft
    def frames_to_magspec(self, frames):
        magspec = np.zeros([self.fft_size//2+1,len(frames[0])])
        frames = frames.transpose()
        for i in range (0,len(frames)):
            frame_fft=np.fft.fft(frames[i])
            frame_fft=np.abs(frame_fft[0:(self.fft_size//2+1)])
            magspec[:,i]=frame_fft

        return magspec

    # for each frame(column of 2D array 'magspec'), compute the log mel spectrum, by applying the mel filterbank to the magnitude spectrum
    def magspec_to_fbank(self, magspec):
        fbank=self.mel_filterbank.dot(magspec)
        fbank=np.log(fbank)
        return fbank

    # compute the mean vector of fbank coefficients in the utterance and subtract it from all frames of fbank coefficients
    def mean_norm_fbank(self, fbank):
        mean=np.mean(fbank,axis=1)
        fbank=fbank-mean[:,None]
        return fbank

    # accumulates sufficient statistics for corpus mean and variance
    def accumulate_stats(self, fbank):
        self.global_mean += np.sum(fbank,axis=1)
        self.global_var += np.sum(fbank**2,axis=1)
        self.global_frames += fbank.shape[1]


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



    


    # compute corpus mean and variance based on sufficient statistics
    def compute_stats(self):
        self.global_mean /= self.global_frames
        self.global_var /= self.global_frames
        self.global_var -= self.global_mean**2

        return self.global_mean, 1.0/np.sqrt(self.global_var)

    def process_utterance(self, utterance):
        wav     = self.dither(utterance)
        wav     = self.pre_emphasize(wav)

        frames  = self.wav_to_frames(wav)
        magspec = self.frames_to_magspec(frames)
        fbank   = self.magspec_to_fbank(magspec)

        if self.is_power==1:
          energy = self.energy(frames)
          fbank=np.concatenate((fbank,energy),axis=0)

        if self.is_delta==2:
          delta1=self.delta(fbank)
          delta2=self.delta(delta1)
          fbank=np.concatenate((fbank,delta1,delta2),axis=0)

        if self.is_average==1:
          average= self.average(fbank)
          fbank=np.concatenate((fbank,average),axis=0)

        if (self.mean_normalize):
            fbank = self.mean_norm_fbank(fbank)

        if (self.compute_global_stats):
            self.accumulate_stats(fbank)

        return fbank



