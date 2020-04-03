beamWidth=$1
acceptingThresh=$2
amwMax=$3
echo beamWidth is $beamWidth , acceptingThresh is $acceptingThresh, amwMax $amwMax

g++ -g *.cpp
./a.out $beamWidth $acceptingThresh $amwMax > out_$beamWidth\_$acceptingThresh.txt 

echo > reports.txt
sed -i "/Token/d;/^\s*$/d;s/ $//g" out_$beamWidth\_$acceptingThresh.txt
# sed -i "s/\.//g" out_$beamWidth\_$acceptingThresh.txt
trash-put ./split/*
split -l 11 out_$beamWidth\_$acceptingThresh.txt -d ./split/
for file in `ls ./split/`;
do
    echo "-------------------------------------------------------------" >> reports.txt
    tail -n 1 ./split/$file >>  reports.txt
    sed -i "$ d" ./split/$file
    sed -i "/^\s*$/d" ./split/$file
    python3 ../../PerformanceMeasure/performance.py -hyp ./split/$file  -ref ../test_ref | tail -n 2 >> reports.txt
    echo "-------------------------------------------------------------" >> reports.txt
done