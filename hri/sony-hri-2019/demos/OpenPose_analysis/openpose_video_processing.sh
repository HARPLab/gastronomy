#!/bin/bash
#echo "This script is about to run another script."

a=$1

#echo $a

b="../sony-hri/example-inputs/$a"

#b = "$(cd "$(dirname "$OPTARG")"; pwd)/$(basename "$OPTARG")$a"
echo $b


cd $(dirname $0)
cd ./../../../openpose
#cd ./../../../openpose/build/examples/openpose
#ls
#echo $b
#ls
#echo $PATH

mkdir -p ../sony-hri/output/processed_${a%.*}


./build/examples/openpose/openpose.bin --video $b --face --hand --write_json ../sony-hri/output/processed_$a/openpose --display 0 --render_pose 0

#mv processed ../../../sony-hri/output/processed_${a%.*}/openface/


echo "Done"
#. "./../../OpenFace/build/bin/FaceLandmarkVidMulti -f $OPTARG"

#sh "./../../OpenFace/build/bin/FaceLandmarkVidMulti -f $OPTARG"
#echo "This script has just run another script."
