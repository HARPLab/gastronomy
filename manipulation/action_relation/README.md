# Relational Learning for Skill Preconditions

To learn relational models for skill preconditions requires three steps. First, we need to generate parwise interaction data in simulation. Once the pairwise object interaction data has been generated we can use it to learn object relation model. Our object relation model consists of predictive losses on position, orientation and contacts and contrastive losses on positions and orientations. Once a pairwise object relation model has been trained on simulation data, we use it to learn precondition models on real world robot data. This real world robot data is collected through multiple sensors which fuse their output to create a 3D scene representation. This 3D scene representation is converted into a voxel representation which is then used as input to our precondition learning algorithm. For our precondition learning algorithm we use two different neural network architectures, one based on relational neural networks and the other a more generic graph neural network. Both of these algorithms are used in conjunction with the relational momdel learnt in step 2 to predict precondition models.

Although, there exist multiple different scripts that run the above algorithms, below we list three scripts which can be run by themselves to learn the precondition models. For more info on the runtime arguments and options please look at other bash scripts in the `run_scripts` folder as well as use the `--help` argument for each python executable to find the different runtime arguments for each script.

- To generate simulation scenes run the following scripts 
`python ./action_relation/sim_vrep/robot_scene_4_orient_change_cut_food.py --port 19997 --scene_file ./scenes/invisible_hand/test_scene_4_paper.ttt --anchor_in_air 0 --save_dir /tmp/whatever/sample_2_edge_1_out/try_0 --num_scenes 2000 --reset_every_action 0`

    - There are other scripts which generate other different types of simulation scenes.

- Once the simulation data has been generated run the relation learning algorithm on this data, using the following script
    `bash ./action_relation/run_scripts/robot_data/action_relative_absolute/run_multidata_unscaled_online_contrastive.sh`

- Once the relational model has been learnt we can use it to learn preconditions from real robot data. To run this execute the following script
    `bash ./action_relation/run_scripts/real_robot_data/run_box_stacking_emb_gnn.sh`

