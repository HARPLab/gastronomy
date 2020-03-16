#ifndef IAM_ROBOLIB_TRAJECTORY_GENERATOR_FACTORY_H_
#define IAM_ROBOLIB_TRAJECTORY_GENERATOR_FACTORY_H_

#include "iam_robolib_common/definitions.h"
#include "iam_robolib/run_loop_shared_memory_handler.h"

class TrajectoryGenerator;

class TrajectoryGeneratorFactory {
 public:

  TrajectoryGeneratorFactory() {};

  /**
   * Get trajectory generator for skill.
   *
   * @param memory_region  Region of the memory where the parameters
   * will be stored.
   */
  TrajectoryGenerator* getTrajectoryGeneratorForSkill(SharedBufferTypePtr buffer);

};

#endif  // IAM_ROBOLIB_TRAJECTORY_GENERATOR_FACTORY_H_