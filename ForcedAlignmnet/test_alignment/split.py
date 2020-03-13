import soundfile as sf
import numpy as np

wrods = ranges = []
with open("transcript.txt") as f:
    words = f.read().split()

with open("alignment.txt") as f:
    ranges = f.read().split()
ranges = list(map(int,ranges))
ranges = list(zip(ranges[::2], ranges[1::2]))

print(ranges)
data, samplerate = sf.read("test.flac")
samplerate = 16000
frame_duration=0.025; frame_shift=0.010

win_size = int(np.floor(frame_duration * samplerate))
win_shift = int(np.floor(frame_shift * samplerate))


def samples(n):
    return win_size + win_shift*(n)

for i,((start,end), word) in enumerate(zip(ranges,words)):
    start = samples(start)
    end = samples(end)
    print(start, end)
    sf.write(f'./segments/{i}_{word}.flac', data[start:end], samplerate)