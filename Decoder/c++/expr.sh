beamWidth=$1
acceptingThresh=$2
amw=$3
echo beamWidth is $beamWidth , acceptingThresh is $acceptingThresh, amw $amw

g++ -g *.cpp
./a.out $beamWidth $acceptingThresh $amw $amw $amw > out_$beamWidth\_$acceptingThresh.txt

sed -i "/Token/d;/^\s*$/d;s/ $//g" out_$beamWidth\_$acceptingThresh.txt
sed -i "s/\!//g" out_$beamWidth\_$acceptingThresh.txt
echo "-------------------------------------------------------------" >> reports.txt
tail -n 1 out_$beamWidth\_$acceptingThresh.txt >>  reports.txt
sed -i "/amw:/d" out_$beamWidth\_$acceptingThresh.txt
sed -i "/^\s*$/d" out_$beamWidth\_$acceptingThresh.txt
python3 ../../PerformanceMeasure/performance.py -hyp out_$beamWidth\_$acceptingThresh.txt  -ref ../ref_orig.txt | tail -n 2 >> reports.txt
echo "-------------------------------------------------------------" >> reports.txt