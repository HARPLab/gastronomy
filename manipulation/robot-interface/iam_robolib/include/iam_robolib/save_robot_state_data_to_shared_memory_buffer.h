#ifndef IAM_ROBOLIB_SAVE_ROBOT_STATE_TO_SHARED_MEMORY_BUFFER_H_
#define IAM_ROBOLIB_SAVE_ROBOT_STATE_TO_SHARED_MEMORY_BUFFER_H_

#include <cstring>
#include <iostream>

#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/run_loop_shared_memory_handler.h"

#include <iam_robolib_common/definitions.h>

void save_current_robot_state_data_to_shared_memory_buffer(RunLoopSharedMemoryHandler* shared_memory_handler,
                                                           RobotStateData* robot_state_data);

#endif  // IAM_ROBOLIB_SAVE_ROBOT_STATE_TO_SHARED_MEMORY_BUFFER_H_