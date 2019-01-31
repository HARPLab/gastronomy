#ifndef IAM_ROBOLIB_DEFINITIONS_H_
#define IAM_ROBOLIB_DEFINITIONS_H_

// SharedBuffer type to share memory (Change size later)
typedef float* SharedBuffer;

// Enum for Robot Types
enum class RobotType {
	FRANKA = 0,
	UR5E = 1
};

#endif  // IAM_ROBOLIB_DEFINITIONS_H_