#!/bin/bash

ex='./searchParams.sh New_notrans_noself BLSTM1_activations files_dev.txt 7000 13 2 2 1 5 ref_orig_19.txt true'

if [ ! $1 ] || [ ! $2 ] || [ ! $3 ] || [ ! $4 ] || [ ! $5 ] || [ ! !6 ] || [ ! $7 ] || [ ! $8 ] || [ ! $9 ] || [ ! ${10} ]
then
    echo Wrong num of params
    echo Usage: $ex
    exit 1
fi

graphFolder=./Graphs/$1/
activationsFolder=./Activations/$2/
filesFile=$3
maxActiveTokens=$4
beam=$5
amwStart=$6
amwEnd=$7
amwStep=$8
latticeBeam=$9
refFilePath=./RefTrans/${10}
showPerfMeasureOutput=${11}
refFileLen=`cat $refFilePath | wc -l `

echo graphFolder: $graphFolder ,activationsFolder: $activationsFolder ,filesFile: $filesFile
echo maxActiveTokens is $maxActiveTokens ,beam is $beam ,amwStart $amwStart ,amwEnd $amwEnd ,amwStep $amwStep
echo ReferencePath fie is $refFilePath, lattice beam is $latticeBeam, showPerfMeasureOutput is $showPerfMeasureOutput


params=$graphFolder\ $activationsFolder\ $filesFile\ $maxActiveTokens\ $beam\ $amwStart\ $amwEnd\ $amwStep\ $latticeBeam
params_=$1\_$2\_$3\_$4\_$5\_$6\_$7\_$8\_$9\_on_$refFileLen\_examples
outFilePath=./PredTrans/out_$params_.txt

lasAmwStart=`cat ./PredTrans/out_$params_.txt | grep "amw:" | tail -n 1 | grep -o "[0-9]\+.\?[0-9]*"`
if [ $lasAmwStart ]
then
    let amwStart="lasAmwStart+$amwStep"
    echo "Continue on perivous amw amwStart set to $amwStart"
fi


echo "---------------------------------------------------------------" >> reports.txt
echo graphFolder: $graphFolder ,activationsFolder: $activationsFolder ,filesFile: $filesFile >> reports.txt
echo maxActiveTokens is $maxActiveTokens ,beam is $beam ,amwStart $amwStart ,amwEnd $amwEnd ,amwStep $amwStep >> reports.txt
echo ReferencePath fie is $refFilePath >> reports.txt

g++ -g ./C++/*.cpp
./a.out  $params >> $outFilePath
/bin/rm a.out

sed -i "s/^\s*$/---/g" $outFilePath
sed  "/Total Path Cost:/d;/^\s*$/d;" $outFilePath > tmp.txt
cat tmp.txt | head -n 1 >> reports.txt
sed -i "/Parsing/d" tmp.txt


/bin/rm ./PredTrans/Split/*
let refFileLen="refFileLen+2"
split -l $refFileLen tmp.txt -d ./PredTrans/Split/
/bin/rm tmp.txt

for file in `ls ./PredTrans/Split/`;
do
    numOfLines=`cat ./PredTrans/Split/$file | wc -l`
    if [ "$numOfLines" == "$refFileLen" ] 
    then
        tail -n 2 ./PredTrans/Split/$file >>  reports.txt
        sed -i "/amw:/d;/in:/d" ./PredTrans/Split/$file
        perfMeasureOutput=`python3 ../PerformanceMeasure/performance.py -hyp ./PredTrans/Split/$file  -ref $refFilePath`
        echo "$perfMeasureOutput" | tail -n 2 >> reports.txt
        if [ $lasAmwStart ]
        then
            echo "$perfMeasureOutput"
        fi
    else
        echo "Wrong number of lines discard split: $file"
    fi
    echo "-------------------------------------------------------------" >> reports.txt
done
cat reports.txt | tail -n 100