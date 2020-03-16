#ifndef IAM_ROBOLIB_COMMON_DEFINITIONS_H_
#define IAM_ROBOLIB_COMMON_DEFINITIONS_H_

#include <stdint.h>

/*
 *
 *  Important: Any Changes here should also be reflected in changes
 *  in the frankpy iam_robolib_common_definitions.py file as well.
 *
 */

// SharedBuffer type to share memory (Change size later)
typedef double SharedBufferType;
typedef SharedBufferType* SharedBufferTypePtr;

typedef uint8_t SensorBufferType;
typedef SensorBufferType* SensorBufferTypePtr;

// Enum for Robot Types
enum class RobotType : uint8_t {
    FRANKA = 0,
    UR5E = 1
};

// Enum for Skill Types
enum class SkillType : uint8_t {
    CartesianPoseSkill = 0,
    ForceTorqueSkill = 1,
    GripperSkill = 2,
    ImpedanceControlSkill = 3,
    JointPositionSkill = 4,
    JointPositionDynamicInterpolationSkill = 5,
};

// Enum for Meta Skill Types
enum class MetaSkillType : uint8_t {
    BaseMetaSkill = 0,
    JointPositionContinuousSkill = 1
};

// Enum for Trajectory Generator Types
enum class TrajectoryGeneratorType : uint8_t {
    GoalPoseDmpTrajectoryGenerator = 0,
    GripperTrajectoryGenerator = 1,
    ImpulseTrajectoryGenerator = 2,
    JointDmpTrajectoryGenerator = 3,
    LinearPoseTrajectoryGenerator = 4,
    LinearJointTrajectoryGenerator = 5,
    MinJerkJointTrajectoryGenerator = 6,
    MinJerkPoseTrajectoryGenerator = 7,
    PoseDmpTrajectoryGenerator = 8,
    RelativeLinearPoseTrajectoryGenerator = 9,
    RelativeMinJerkPoseTrajectoryGenerator = 10,
    SineJointTrajectoryGenerator = 11,
    SinePoseTrajectoryGenerator = 12,
    StayInInitialJointsTrajectoryGenerator = 13,
    StayInInitialPoseTrajectoryGenerator = 14
};

// Enum for Feedback Controller Types
enum class FeedbackControllerType : uint8_t {
    CartesianImpedanceFeedbackController = 0,
    ForceAxisImpedenceFeedbackController = 1,
    JointImpedanceFeedbackController = 2,
    NoopFeedbackController = 3,
    PassThroughFeedbackController = 4,
    SetInternalImpedanceFeedbackController = 5
};

// Enum for Termination Handler Types
enum class TerminationHandlerType : uint8_t {
    ContactTerminationHandler = 0,
    FinalJointTerminationHandler = 1,
    FinalPoseTerminationHandler = 2,
    NoopTerminationHandler = 3,
    TimeTerminationHandler = 4
};

// Enum for Skill Statuses
enum class SkillStatus : uint8_t { 
  TO_START = 0, 
  RUNNING = 1, 
  FINISHED = 2,
  VIRT_COLL_ERR = 3
}; 


enum class SensorDataManagerReadStatus : uint8_t {
  FAIL_TO_GET_LOCK = 0,
  FAIL_TO_READ = 1,
  NO_NEW_MESSAGE = 2,
  SUCCESS = 3,
};

enum class SensorDataMessageType : uint8_t {
  JOINT_POSITION = 4,
  BOUNDING_BOX = 5,
};

#endif  // IAM_ROBOLIB_COMMON_DEFINITIONS_H_