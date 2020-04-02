beamWidth=$1
acceptingThresh=$2
amwMax=$3
lmwMax=$4
hmmwMax=$4
echo beamWidth is $beamWidth , acceptingThresh is $acceptingThresh, amwMax $amwMax , lmwMax $lmwMax, hmmwMax $hmmwMax

g++ -g *.cpp
./a.out $beamWidth $acceptingThresh $amwMax $lmwMax $hmmwMax > out_$beamWidth\_$acceptingThresh.txt 

sed -i "/Token/d;/^\s*$/d;s/ $//g" out_$beamWidth\_$acceptingThresh.txt
sed -i "s/\.//g" $out_$beamWidth\_$acceptingThresh.txt
# rm  ./splilt *
splilt -l 11 out_$beamWidth\_$acceptingThresh.txt -d ./splilt
for file in `ls ./split/`;
do
    echo "-------------------------------------------------------------" >> reports.txt
    tail -n 1 ./splilt/$file >>  reports.txt
    sed -i "$ d" ./splilt/$file
    sed -i "/^\s*$/d" ./splilt/$file
    python3 ../../PerformanceMeasure/performance.py -hyp ./splilt/$file  -ref ../test_ref | tail -n 2 >> reports.txt
    echo "-------------------------------------------------------------" >> reports.txt
done