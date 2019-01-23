#pragma once

#include "run_loop_shared_memory_handler.h"
#include "definitions.h"

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
  TrajectoryGenerator* getTrajectoryGeneratorForSkill(SharedBuffer buffer);

};

