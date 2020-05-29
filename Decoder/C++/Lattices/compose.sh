for file in `ls lat*.txt`; do
    echo compiling $file
    newFileName=`echo $file | cut -f 1 -d '.'`.fst
    fstcompile  $file | fstmap --map_type=rmweight | fstarcsort --sort_type="olabel" > $newFileName
    # echo composing 
    # fstcompose $newFileName G3_comp_ready.fst | fstprint  > composed_$file
    # echo pushing weights
    # fstpush --push_weights --push_labels temp.fst | fstprint  > composed_$file
done


g++ -std=c++11 -I /usr/local/include compose.cpp  -lfst -lpthread

g3="G3_comp_ready.fst"
lats=`ls lat*.fst`
input=$g3$'\n'$lats
echo "$input" | ./a.out

# for file in `ls composed_lat*.fst`; do
#     echo optimizing $file
#     file=`echo $file | cut -f 1 -d '.'`
#     fstencode --encode_labels --encode_weights $file.fst encoder $file\_enc.fst
#     fstdeterminize $file\_enc.fst | fstminimize > $file\_opt\_enc.fst
#     fstencode --decode $file\_opt\_enc.fst encoder $file.fst
# done
# /bin/rm *_enc.fst

for file in `ls composed_lat*.fst`; do
    echo printing $file
    cat $file | fstprint  > `echo $file | cut -f 1 -d '.'`.txt 
done

/bin/rm lat*.fst
/bin/rm composed_lat*.fst


# fstcompile $file | fstdeterminize --det_type="disambiguate" | fstminimize | fstprune --weight=25 | fstmap --map_type=rmweight | fstarcsort --sort_type="olabel" > $newFileName
