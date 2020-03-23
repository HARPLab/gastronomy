HRI 


OpenFace Analysis
---------
Within the directory demos/OpenFace_analysis:
`./openface_video_processing.sh $video_name.mp4`

Output will appear in the outputs folder in "output/processed_$video_name/openface"

OpenPose Analysis 
---------
Within the directory demos/OpenPose_analysis:
`./openpose_video_processing.sh $video_name.mp4`

Output will appear in the outputs folder in "output/processed_$video_name/openpose"

OpenFace Live Demo
---------
Within the directory demos/OpenFace_livedemo:
`./openface_livedemo.sh`
This downloads and unpacks a pre-compiled Windows Visual Studio exe from the OpenFace project, and prints the path to the specific demo file. The enclosed file "OpenFaceOffline.exe" can be run directly on a Windows machine without further installs.


### Examples can be found in the example-inputs directory. These include:

| File          | Content       |
| -------------:|:-------------:| 
| cooking_show.mp4           | A clip taken from a cooking show that demonstrates facial expressions at a standing and table setting |
| actual_restaurant.mp4      | A clip taken from a public use webcam at the restaurant Ridtydz in Myrtle Beach      |
| matrix.mp4      | A restaurant scene taken from The Matrix. Note the relative robustness during chewing, and despite sunglasses      |

### Code
Additional modules that need to be built before being run are in the code directory.










