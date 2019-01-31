//
// Created by mohit on 11/25/18.
//

#include "iam_robolib/termination_handler/termination_handler.h"

void TerminationHandler::check_terminate_preempt() {
    done_ = run_loop_info_->get_skill_preempted();
};