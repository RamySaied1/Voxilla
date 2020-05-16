def frames_to_seconds(alignments,frame_duration,frame_shift):
    result=[]
    for alignment in alignments:
        word=alignment[0]
        start_frame=alignment[1][0]
        end_frame=alignment[1][1]

        start_time=(frame_shift*start_frame)+(frame_duration/2)
        end_time=(frame_shift*end_frame)+(frame_duration/2)

        result.append([word,[start_time,end_time]])

    return result
