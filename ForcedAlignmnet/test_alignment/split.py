import soundfile as sf
import numpy as np

wrods = ranges = []
with open("transcript.txt") as f:
    words = f.read().split()

with open("alignment.txt") as f:
    ranges = f.read().split()
ranges = list(map(int,ranges))
ranges = list(zip(ranges[::2], ranges[1::2]))

data, samplerate = sf.read("test.flac")
assert(samplerate == samplerate)
frame_duration=0.025; frame_shift=0.010

win_size = int(frame_duration * samplerate)
win_shift = int(frame_shift * samplerate)

print(f"frame count is: {(len(data)-win_size+win_shift)/win_shift}")

def samples(n):
    intial_shift = win_size-win_shift if n >0 else 0
    return intial_shift + win_shift*n

for i,((start,end), word) in enumerate(zip(ranges,words)):
    start = samples(start)
    end = samples(end)
    print(start, end)
    if(start>=len(data)):
        print(f"range exeded frames limit which is {len(data)}")
        exit()
    sf.write(f'./segments/{i}_{word}.flac', data[start:end], samplerate)

print(f"OK Last number should be {len(data)}")