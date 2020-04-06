graphFolder=./graphs/$1/
activationsFolder=./activations/$2
filesFile=$3
maxActiveTokens=$4
beam=$5
amwStart=$6
amwEnd=$7
amwStep=$8
refFilePath=$9

echo graphFolder: $graphFolder, activationsFolder: $activationsFolder, filesFile: $filesFile
echo maxActiveTokens is $maxActiveTokens , beam is $beam, amwStart $amwStart, amwEnd $amwEnd, amwStep $amwStep
echo ReferencePath fie is $refFilePath

if [ ! $amwStart ]
then
    amwStart=`cat reports.txt | grep -o "[0-9]\+.[0-9]*" > amwEnd`
    if [ ! $amwStart ]
    then
        amwStart="1."
    fi
fi

params=$graphFolder\ $activationsFolder\ $filesFile\ $maxActiveTokens\ $beam\ $amwStart\ $amwEnd\ $amwStep
params_=`echo $params | sed 's/_/ //g'`

g++ -g *.cpp
./a.out  $params > out_$params_.txt 

sed -i "/Token/d;/^\s*$/d;s/ $//g;/in:/d" out_$params_.txt
trash-put ./split/*

refFileLen=`cat $refFilePath | wc -l `
let refFileLen="refFileLen+1"

split -l $refFileLen out_$params_.txt -d ./split/
for file in `ls ./split/`;
do
    echo "-------------------------------------------------------------" >> reports.txt
    tail -n 1 ./split/$file >>  reports.txt
    sed -i "/amw:/d" ./split/$file
    sed -i "s/\!//g" ./split/$file
    python3 ../../PerformanceMeasure/performance.py -hyp ./split/$file  -ref $refFilePath | tail -n 2 >> reports.txt
    echo "-------------------------------------------------------------" >> reports.txt
done
