Author: Steven Lee
Date: 10/20/19

Notes from Author:
    * This file has information on how to train and test a YOLO image detector.
    
    * I trained the system with custom images, but used pretrained weights.

Some system configuration information:
    * Running ubuntu 16.04.6
    
    * Using YOLOv3 from offical darknet website [3][4]. See the site
      for info on YOLO and how to use it  
    
    * Using darknet to train and test YOLO system
    
    * Using a python wrapper for everything other than training.
      The wrapper I am using is: https://github.com/ayooshkathuria/pytorch-yolo-v3
        - Honestly, I don't recommend it. The guy who made it is
          extremely unorganized and sloppy. The creator of YOLO
          has his own pytorch implementation that is cleaner,
          but pretty bare bones
        - Some of the chanegs I made to the wrapper are in the 
          notes.txt file describing this (11/4/19)
    
    * Also using a virtual environment.
    
    * Using the github repo labelImg to do image annotation

Some helpful notes:
    * To use the GPU, change the GPU parameter in the Makefile to 1, as
      well as the cudnn. Can use OpenCV if you want.
        - If you change the Makefile you will need to remake darknet
    
    * (You can probably just ignore this bullet, isn't that helpful)
      The maximum number of objects the system identifies is controlled
      by the mask value in the last yolo layer of the cfg file, around 
      line 781. These numbers point to the anchor values.
        - if mask = 0,1,2 then it is refering to the 1st 2nd and 3rd
          anchors. Each cell of the detection layer predicts 3 boxes, 
          so there will be a max of 9 anchors or boxes.
    
    * The formatting of the YOLO .txt files for image annotation is as
      follows:
        - <object-class> <x> <y> <width> <height>
        - where object-class is the class number specified in the 
          .names file ranging from 0 to class_num-1, the first class
          listed in the .names file is zero and moves down sequentially 
    
    * Use virtual environments when running python stuff. will help with
      compatability issues caused by conflicting programs, ie ROS, use [7]
    
    * I think you need to put your training images inside the darknet
      directory and maybe even inside a directory called labels 
        - Remember to put the .txt label files in the same folder as the
          images too
        - Some random thoughts (not important):
          was getting an error when they weren't, but I might've just messed 
          something up (still happens for some reason, must be something
          with the darknet c code, also the name  can't be too long? idk
          I tried images4training too and it didn't work, maybe it is 
          something to do with the word image?)
    
    * Am occasionally getting images corrupted during training or 
      image processing this is an issue with cv2 and PIL, see [8] 
   
   * To change when the weights are saved, see [9]

Other Issues:
    * After updating cuda10.0 to cuda 10.1 the folder in /usr/local/bin
      changed from cuda to cuda-10.1. So you'll need to change the paths
      in your ~/.bashrc file to cuda-10.1 as well as the call to cuda in
      the Makefile when you make the directory 
    
    * consider having multiple installed: 
      https://medium.com/@peterjussi/multicuda-multiple-versions-of-cuda-on-one-machine-4b6ccda6faae

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Commands to run for training and testing
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Run this command in the terminal to train a set of weights with custom images:
iam-lab@iam-ego:~/Downloads/darknet$ ./darknet detector train cfg/moz-tom-obj.data cfg/moz-tom-yolov3.cfg darknet53.conv.74
    - ./darknet is the command
    - detector is an argument of the command
    - train is an argument of the command
    - cfg/mox-tom-obj.data is the file path of the .data file you should make for
            running commands. It describes your data set. see section 5 of [1] or
            the first code snippet under "Preparing the YOLOv3 configuration files"
            in [2]
    - cfg/mox-tom-yolov3.cfg is the configuration file path for running yolo. I used [2]
            but [1] has information on seting I should have also changed. It still
            works with just the settings in [2] under the heading,
            "Step2: (if you choose yolov3.cfg)"
    - darknet53.conv74 is the set of prettrained weights I used, you can use a different one
    - Can add an additional argument at the end to specify hwo many gpus you want to use
            ie: -gpus 0,1,2,3
    - MAKE SURE YOU CHANGE THE .cfg and Makefile FILE PARAMETERS ACCORDINGLY. see [6]

Run this command in the terminal to run a prediction on an image:
iam-lab@iam-ego:~/Downloads/darknet$ ./darknet detector test cfg/moz-tom-obj.data cfg/moz-tom-yolov3.cfg backup/moz-tom-yolov3_50000.weights YOLOtest01/testImages01/20191017_123822.jpg
    Note: make sure you are in the darknet diectory
    - ./darknet is the command
    - detector is an argument of the command
    - test is an argument of the command
    - cfg/mox-tom-obj.data is the file path of the .data file you should make for
            running commands. It describes your data set. see section 5 of [1] or
            the first code snippet under "Preparing the YOLOv3 configuration files"
            in [2]
    - cfg/mox-tom-yolov3.cfg is the configuration file path for running yolo. I used [2]
            but [1] has information on seting I should have also changed. It still
            works with just the settings in [2] under the heading "Step2: (if you ch
            oose yolov3.cfg)"
    - backup/moz-tom-yolov3_50000.weights is the trained weights file path that was 
            created when you trained your system
    - the last argument is the file path of the image that you would like to test
            with to obtain a prediction  

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
References
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

[1] https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/
[2] https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2
[3] https://pjreddie.com/darknet/install/
[4] https://pjreddie.com/darknet/yolo/
[5] https://pypi.org/project/darknetpy/
[6] https://github.com/AlexeyAB/darknet.git
[7] https://docs.python-guide.org/dev/virtualenvs/
[8] https://github.com/Oslandia/deeposlandia/issues/129
[9] https://github.com/pjreddie/darknet/issues/190
