beamWidth=$1
acceptingThresh=$2
amwStart=$3
amwEnd=$4
amwStep=$5
# numOfUlts=$4
echo beamWidth is $beamWidth , acceptingThresh is $acceptingThresh, amwStart $3, amwEnd $amwEnd, amwStep $amwStep

if [ ! $amwStart ]
then
    amwStart=`cat reports.txt | grep -o "[0-9]\+.[0-9]*" > amwEnd`
    if [ ! $amwStart ]
    then
        amwStart="1."
    fi
fi

g++ -g *.cpp
./a.out $beamWidth $acceptingThresh $amwStart $amwEnd $amwStep> out_$beamWidth\_$acceptingThresh.txt 

sed -i "/Token/d;/^\s*$/d;s/ $//g" out_$beamWidth\_$acceptingThresh.txt
sed -i "/amw:/d" out_$beamWidth\_$acceptingThresh.txt
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
