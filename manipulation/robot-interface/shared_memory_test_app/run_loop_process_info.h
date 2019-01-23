//
// Created by mohit on 11/20/18.
//

#include <string>

class RunLoopProcessInfo {
    public:
        RunLoopProcessInfo(int memory_region_idx): current_memory_region_(memory_region_idx) {};

        bool new_task_available_{false};
        bool is_running_task_{false};

        bool can_run_new_task();

        std::string  get_current_shared_memory_name();

        std::string get_shared_memory_for_actionlib();

        int get_current_shared_memory_index();

        /**
         * Update shared memory region.
         */
        void update_shared_memory_region();

        /**
         * Return the id for the latest skill available.
         */
        int get_new_skill_id();

        /**
         * Update current skill being executed.
         */
        void update_current_skill(int new_skill_id);

        void update_new_skill(int new_skill_id);

    private:
        int current_memory_region_{0};
        int current_skill_id_{-1};
        int new_skill_id_{-1};

        std::string get_shared_memory_name_for_memory_idx(
            int memory_idx);
};

