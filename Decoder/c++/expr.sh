beamWidth=$1
acceptingThresh=$2
echo beamWidth is $beamWidth , acceptingThresh is $acceptingThresh

g++ -g *.cpp

./a.out $beamWidth $acceptingThresh > out_$beamWidth\_$acceptingThresh.txt

sed -i "/Token/d;/^\s*$/d;s/ $//g" out_$beamWidth\_$acceptingThresh.txt

echo "beamwidth: $1, acceptingThresh: $2" >> reports.txt
tail -n 1 out_$beamWidth\_$acceptingThresh.txt >> reports.txt
echo "" >> reports.txt
sed -i "$ d" out_$beamWidth\_$acceptingThresh.txt
python3 ../../PerformanceMeasure/performance.py -hyp out_$beamWidth\_$acceptingThresh.txt -ref ../ref.txt | tail -n 2 >> reports.txt
echo "-------------------------------------------------------------" >> reports.txt
rm out_$beamWidth\_$acceptingThresh.txt