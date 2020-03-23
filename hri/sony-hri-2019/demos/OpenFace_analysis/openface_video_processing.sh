#!/bin/bash
#echo "This script is about to run another script."

a=$1

#echo $a

b="../../../sony-hri/example-inputs/$a"

#b = "$(cd "$(dirname "$OPTARG")"; pwd)/$(basename "$OPTARG")$a"
echo $b


cd $(dirname $0)
cd ./../../../OpenFace/build/bin
#ls
#echo $b
#echo $PATH
./FaceLandmarkVidMulti -f $b

mkdir -p ../../../sony-hri/output/processed_${a%.*}
mv processed ../../../sony-hri/output/processed_${a%.*}/openface


echo "Done"
#. "./../../OpenFace/build/bin/FaceLandmarkVidMulti -f $OPTARG"

#sh "./../../OpenFace/build/bin/FaceLandmarkVidMulti -f $OPTARG"
#echo "This script has just run another script."
