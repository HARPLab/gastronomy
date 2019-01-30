#ifndef IAM_ROBOLIB_SKILL_INFO_MANAGER_H_
#define IAM_ROBOLIB_SKILL_INFO_MANAGER_H_

#include <vector>

#include "iam_robolib/skills/base_skill.h"
#include "iam_robolib/skills/base_meta_skill.h"

class BaseSkill;
class BaseMetaSkill;

class SkillInfoManager {
 public:
  SkillInfoManager();

  BaseSkill* get_current_skill();

  BaseMetaSkill* get_current_meta_skill();

  bool is_currently_executing_skill();

  bool is_waiting_for_new_skill();

  void add_skill(BaseSkill *skill);

  void add_meta_skill(BaseMetaSkill* skill);

  BaseMetaSkill* get_meta_skill_with_id(int meta_skill_id);

 private:
  std::vector<BaseSkill *> skill_list_{};
  std::vector<BaseMetaSkill *> meta_skill_list_{};
};

#endif  // IAM_ROBOLIB_SKILL_INFO_MANAGER_H_