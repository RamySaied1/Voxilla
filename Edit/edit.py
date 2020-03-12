import argparse
from tgt import read_textgrid,write_to_file
import soundfile as sf
import sys

SAMP_RATE = 16000

#input -------
# quary ----> tuple (word,beginTime,endTime)
# audio ---->  matrix
# transcript ---> textgrid
#output
#after modification + transcript

def delete_word(audio,sample_rate,transript,quary):
    word_tier = transript.get_tier_by_name("words")
    tier_end=word_tier._get_end_time()

    ############ remove it from audio
    ann= word_tier.get_annotations_between_timepoints(quary[1], quary[2])
    begin=round(quary[1]*sample_rate)
    end = round(quary[2]*sample_rate)
    
    audio = [*audio[:begin],*audio[end:]]
    ########### align new transcript
    word_tier.delete_annotation_by_start_time(quary[1])
    anns = word_tier.get_annotations_between_timepoints(quary[2],tier_end)
    #print(anns)
    for interval in anns:
        interval._set_start_time(round(interval._get_start_time()-(quary[2]-quary[1]),4))
        interval._set_end_time(round(interval._get_end_time()-(quary[2]-quary[1]),4))

    #print(word_tier._get_annotations())
    return audio, transript


x, s = sf.read("19-198-0037.flac")
grid = read_textgrid("19-198-0037.TextGrid")
quary= ("seek",0.690,0.910)

audio, transript = delete_word(x,s,grid,quary)
sf.write("modified.flac", audio, s,)


write_to_file(transript,"modified.TextGrid",format='long')


#
    





    
