#!/bin/bash

# python3 State_Machine.py 90 3 3 True True deterministic_r
# python3 State_Machine.py 90 3 3 True True deterministic_no_op
# python3 State_Machine.py 90 3 3 True True deterministic_hybrid
# python3 State_Machine.py 90 3 3 True True deterministic_hybrid_3T
# python3 State_Machine.py 90 3 3 True True deterministic_no_op_hybrid
# python3 State_Machine.py 90 3 3 True True deterministic_no_op_hybrid_3T

# python3 State_Machine.py 90 3 3 True True deterministic_all_greedy_no_op
# python3 State_Machine.py 90 3 3 True True deterministic_all_greedy_hybrid
# python3 State_Machine.py 90 3 3 True True deterministic_all_greedy_no_op_hybrid

# python3 State_Machine.py 90 3 3 True True simple_r

# test_folder=tests_new_env
test_folder=tests

cp -p ../${test_folder}/simple_r_model/*greedy-False* ../${test_folder}/simple_no_op_hybrid_model/
python3 State_Machine.py 90 3 3 True True simple_no_op_hybrid

cp -p ../${test_folder}/simple_r_model/*greedy-False* ../${test_folder}/simple_no_op_hybrid_3T_model/
python3 State_Machine.py 90 3 3 True True simple_no_op_hybrid_3T

cp -p ../${test_folder}/simple_r_model/*greedy-False* ../${test_folder}/simple_no_op_hybrid_shani_model/
python3 State_Machine.py 90 3 3 True True simple_no_op_hybrid_shani

cp -p ../${test_folder}/simple_r_model/*greedy-False* ../${test_folder}/simple_no_op_hybrid_3T_shani_model/
python3 State_Machine.py 90 3 3 True True simple_no_op_hybrid_3T_shani

cp -p ../${test_folder}/simple_no_op_hybrid_model/*greedy-True* ../${test_folder}/simple_H_POMDP_model/
python3 State_Machine.py 90 3 3 False True simple_H_POMDP

cp -p ../${test_folder}/simple_r_model/*greedy-False* ../${test_folder}/simple_no_op_model/
python3 State_Machine.py 90 3 3 True True simple_no_op


cp -p ../${test_folder}/simple_no_op_hybrid_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_model/
cp -p ../${test_folder}/simple_no_op_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_model/
python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op_hybrid

cp -p ../${test_folder}/simple_no_op_hybrid_3T_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_3T_model/
cp -p ../${test_folder}/simple_no_op_hybrid_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_3T_model/
python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op_hybrid_3T

cp -p ../${test_folder}/simple_no_op_hybrid_shani_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_shani_model/
cp -p ../${test_folder}/simple_no_op_hybrid_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_shani_model/
python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op_hybrid_shani

cp -p ../${test_folder}/simple_no_op_hybrid_3T_shani_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_3T_shani_model/
cp -p ../${test_folder}/simple_no_op_hybrid_3T_model/*greedy-True* ../${test_folder}/simple_all_greedy_no_op_hybrid_3T_shani_model/
python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op_hybrid_3T_shani


# cp -p ../${test_folder}/complex_r_model/*greedy-False* ../${test_folder}/complex_no_op_hybrid_model/
# python3 State_Machine.py 90 3 3 True False complex_no_op_hybrid

# cp -p ../${test_folder}/complex_r_model/*greedy-False* ../${test_folder}/complex_no_op_hybrid_3T_model/
# python3 State_Machine.py 90 3 3 True False complex_no_op_hybrid_3T

# cp -p ../${test_folder}/complex_r_model/*greedy-False* ../${test_folder}/complex_no_op_hybrid_shani_model/
# python3 State_Machine.py 90 3 3 True False complex_no_op_hybrid_shani

# cp -p ../${test_folder}/complex_r_model/*greedy-False* ../${test_folder}/complex_no_op_hybrid_3T_shani_model/
# python3 State_Machine.py 90 3 3 True False complex_no_op_hybrid_3T_shani

# cp -p ../${test_folder}/complex_no_op_hybrid_model/*greedy-True* ../${test_folder}/complex_H_POMDP_model/
# python3 State_Machine.py 90 3 3 False True complex_H_POMDP

# cp -p ../${test_folder}/complex_r_model/*greedy-False* ../${test_folder}/complex_no_op_model/
# python3 State_Machine.py 90 3 3 True False complex_no_op

# cp -p ../${test_folder}/complex_no_op_hybrid_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_model/
# cp -p ../${test_folder}/complex_no_op_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_model/
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op_hybrid

# cp -p ../${test_folder}/complex_no_op_hybrid_3T_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_3T_model/
# cp -p ../${test_folder}/complex_no_op_hybrid_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_3T_model/
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op_hybrid_3T

# cp -p ../${test_folder}/complex_no_op_hybrid_shani_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_shani_model/
# cp -p ../${test_folder}/complex_no_op_hybrid_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_shani_model/
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op_hybrid_shani

# cp -p ../${test_folder}/complex_no_op_hybrid_3T_shani_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_3T_shani_model/
# cp -p ../${test_folder}/complex_no_op_hybrid_3T_model/*greedy-True* ../${test_folder}/complex_all_greedy_no_op_hybrid_3T_shani_model/
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op_hybrid_3T_shani


###################################################################################################


# python3 State_Machine.py 90 3 3 True True simple_hybrid
# python3 State_Machine.py 90 3 3 True True simple_hybrid_3T
# python3 State_Machine.py 90 3 3 True True simple_hybrid_shani


# python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op_hybrid
# python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op
# python3 State_Machine.py 90 3 3 True True simple_all_greedy_hybrid
# python3 State_Machine.py 90 3 3 True True simple_all_greedy_hybrid_3T
# python3 State_Machine.py 90 3 3 True True simple_all_greedy_no_op_hybrid_3T

# python3 State_Machine.py 90 3 3 True False complex_r


# python3 State_Machine.py 90 3 3 True False complex_hybrid
# python3 State_Machine.py 90 3 3 True False complex_hybrid_3T
# python3 State_Machine.py 90 3 3 True False complex_hybrid_shani

# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op_hybrid
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_hybrid
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_hybrid_3T
# python3 State_Machine.py 90 3 3 True False complex_all_greedy_no_op_hybrid_3T

# python3 State_Machine.py 90 3 3 True True Complex_hybrid_shani
# python3 State_Machine.py 90 3 3 True True Complex_no_op_hybrid_shani



