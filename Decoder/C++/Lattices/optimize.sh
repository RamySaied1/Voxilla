fstfile=$1
echo "Staring Encoding"
fstencode --encode_labels --encode_weights $fstfile.fst encoder $fstfile\_enc.fst
echo "Encoding done"
fstdeterminize $fstfile\_enc.fst | fstminimize > $fstfile\_opt\_enc.fst
echo "Determinization and minimization done"
fstencode --decode $fstfile\_opt\_enc.fst encoder $fstfile.fst
echo "Optimization done done"