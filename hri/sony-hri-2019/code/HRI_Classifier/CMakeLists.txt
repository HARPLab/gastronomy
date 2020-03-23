# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(HRI_Classifier HRI_Classifier.cpp)
target_link_libraries(HRI_Classifier LandmarkDetector)
target_link_libraries(HRI_Classifier FaceAnalyser)
target_link_libraries(HRI_Classifier GazeAnalyser)
target_link_libraries(HRI_Classifier Utilities)

install (TARGETS HRI_Classifier DESTINATION bin)
