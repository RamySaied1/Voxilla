echo > dec_lat.txt
for i in  `seq 1 50`; do
    file=lat$i.txt
    fstcompile $file | fstshortestpath | fstprint | grep -e "[0-9]\+\s[0-9]\+\s[0-9]\+\s[0-9]\+" | cut -f 4 | sed "/^0$/d" > temp.txt && awk 'NR==FNR{a[$2]=$1;next}$1 in a{print $1,a[$1]}'  output.syms  temp.txt | cut -f 2 -d " " | tac |sed -e :a -e '$!N;s/\n/ /;ta' -e 'P;D' >> dec_lat.txt
done
sed -i "/^$/d" dec_lat.txt

echo > dec_composed_lat.txt
for i in `seq 1 50`; do
    file=composed_lat$i.txt
    fstcompile $file |  fstshortestpath | fstprint | grep -e "[0-9]\+\s[0-9]\+\s[0-9]\+\s[0-9]\+" | cut -f 4 | sed "/^0$/d" > temp.txt && awk 'NR==FNR{a[$2]=$1;next}$1 in a{print $1,a[$1]}'  output.syms  temp.txt | cut -f 2 -d " " | tac |sed -e :a -e '$!N;s/\n/ /;ta' -e 'P;D' >> dec_composed_lat.txt
done
sed -i "/^$/d" dec_composed_lat.txt