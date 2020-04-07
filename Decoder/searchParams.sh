#!/bin/bash

ex='./searchParams.sh New_notrans_noself BLSTM1_activations files_dev.txt 7000 13 2 2 1 ref_orig_19.txt'

if [ ! $1 ] || [ ! $2 ] || [ ! $3 ] || [ ! $4 ] || [ ! $5 ] || [ ! !6 ] || [ ! $7 ] || [ ! $8 ] || [ ! $9 ]
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
refFilePath=./RefTrans/$9
refFileLen=`cat $refFilePath | wc -l `

echo graphFolder: $graphFolder, activationsFolder: $activationsFolder, filesFile: $filesFile
echo maxActiveTokens is $maxActiveTokens , beam is $beam, amwStart $amwStart, amwEnd $amwEnd, amwStep $amwStep
echo ReferencePath fie is $refFilePath


params=$graphFolder\ $activationsFolder\ $filesFile\ $maxActiveTokens\ $beam\ $amwStart\ $amwEnd\ $amwStep
params_=$1\_$2\_$3\_$4\_$5\_$6\_$7\_$8\_on_$refFileLen\_examples
outFilePath=./PredTrans/out_$params_.txt

lasAmwStart=`cat ./PredTrans/out_$params_.txt | grep "amw:" | tail -n 1 | grep -o "[0-9]\+.\?[0-9]*"`
if [ $lasAmwStart ]
then
    let amwStart="lasAmwStart+$amwStep"
    echo "Continue on perivous amw amwStart set to $amwStart"
fi


# g++ -g ./C++/*.cpp
# ./a.out  $params >> $outFilePath 

sed -i "/Token/d;/^\s*$/d;/in:/d" $outFilePath
trash-put ./PredTrans/Split/*


echo > reports.txt
let refFileLen="refFileLen+2"
split -l $refFileLen $outFilePath -d ./PredTrans/Split/
for file in `ls ./PredTrans/Split/`;
do
    numOfLines=`cat ./PredTrans/Split/$file | wc -l`
    if [ "$numOfLines" == "$refFileLen" ] 
    then
        echo "-------------------------------------------------------------" >> reports.txt
        tail -n 1 ./PredTrans/Split/$file >>  reports.txt
        sed -i "/amw:/d" ./PredTrans/Split/$file
        python3 ../PerformanceMeasure/performance.py -hyp ./PredTrans/Split/$file  -ref $refFilePath | tail -n 2 >> reports.txt
    else
        echo "Wrong number of lines discard split: $file"
    fi
    echo "-------------------------------------------------------------" >> reports.txt
done
cat reports.txt