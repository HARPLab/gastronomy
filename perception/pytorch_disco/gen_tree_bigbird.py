import pickle
import errno    
import ipdb
st = ipdb.set_trace

import copy
import glob
import os

mapping = {'3m_high_tack_spray_adhesive':'1', 'advil_liqui_gels':'2','aunt_jemima_original_syrup':'3','bai5_sumatra_dragonfruit':'4','band_aid_clear_strips':'5','band_aid_sheer_strips':'5','blue_clover_baby_toy':'6','bumblebee_albacore':'7','campbells_chicken_noodle_soup':'8','campbells_soup_at_hand_creamy_tomato':'8','canon_ack_e10_box':'9','cheez_it_white_cheddar':'10','chewy_dipps_chocolate_chip':'10','chewy_dipps_peanut_butter':'10','cholula_chipotle_hot_sauce':'11','cinnamon_toast_crunch':'10','clif_crunch_chocolate_chip':'10','clif_crunch_peanut_butter':'10','clif_crunch_white_chocolate_macademia_nut':'10','clif_z_bar_chocolate_chip':'10','clif_zbar_chocolate_brownie':'10','coca_cola_glass_bottle':'12','coffee_mate_french_vanilla':'13','colgate_cool_mint':'14','crayola_24_crayons':'15','crayola_yellow_green':'16','crest_complete_minty_fresh':'17', 'crystal_hot_sauce':'11','cup_noodles_chicken':'18','cup_noodles_shrimp_picante':'18','detergent':'19','dove_beauty_cream_bar':'20','dove_go_fresh_burst':'20','dove_pink':'20','eating_right_for_healthy_living_apple':'10','eating_right_for_healthy_living_blueberry':'10','eating_right_for_healthy_living_mixed_berry':'10','eating_right_for_healthy_living_raspberry':'10','expo_marker_red':'16','fruit_by_the_foot':'10','gushers_tropical_flavors':'10','haagen_dazs_butter_pecan':'21','haagen_dazs_cookie_dough':'21','hersheys_bar':'22','hersheys_cocoa':'23','honey_bunches_of_oats_honey_roasted':'10','honey_bunches_of_oats_with_almonds':'10','hunts_paste':'8','hunts_sauce':'8','ikea_table_leg_blue':'24','krylon_crystal_clear':'25','krylon_low_odor_clear_finish':'25','krylon_matte_finish':'25','krylon_short_cuts':'26','listerine_green':'27','mahatma_rice':'28','mom_to_mom_butternut_squash_pear':'29','mom_to_mom_sweet_potato_corn_apple':'29','motts_original_assorted_fruit':'10','nature_valley_crunchy_oats_n_honey':'10','nature_valley_crunchy_variety_pack':'10','nature_valley_gluten_free_roasted_nut_crunch_almond_crunch':'10','nature_valley_granola_thins_dark_chocolate':'10','nature_valley_soft_baked_oatmeal_squares_cinnamon_brown_sugar':'10','nature_valley_soft_baked_oatmeal_squares_peanut_butter':'10','nature_valley_sweet_and_salty_nut_almond':'10','nature_valley_sweet_and_salty_nut_cashew':'10','nature_valley_sweet_and_salty_nut_peanut':'10','nature_valley_sweet_and_salty_nut_roasted_mix_nut':'10','nice_honey_roasted_almonds':'8','nutrigrain_apple_cinnamon':'10','nutrigrain_blueberry':'10','nutrigrain_cherry':'10','nutrigrain_chotolatey_crunch':'10','nutrigrain_fruit_crunch_apple_cobbler':'10','nutrigrain_fruit_crunch_strawberry_parfait':'10','nutrigrain_harvest_blueberry_bliss':'10','nutrigrain_harvest_country_strawberry':'10','nutrigrain_raspberry':'10','nutrigrain_strawberry':'10','nutrigrain_strawberry_greek_yogurt':'10','nutrigrain_toffee_crunch_chocolatey_toffee':'10','palmolive_green':'30','palmolive_orange':'30','paper_cup_holder':'31','paper_plate':'32','pepto_bismol':'33','pop_secret_butter':'10','pop_secret_light_butter':'10','pop_tarts_strawberry':'10','pringles_bbq':'34','progresso_new_england_clam_chowder':'8','quaker_big_chewy_chocolate_chip':'10','quaker_big_chewy_peanut_butter_chocolate_chip':'10','quaker_chewy_chocolate_chip':'10','quaker_chewy_dipps_peanut_butter_chocolate':'10','quaker_chewy_low_fat_chocolate_chunk':'10','quaker_chewy_peanut_butter':'10','quaker_chewy_peanut_butter_chocolate_chip':'10','quaker_chewy_smores':'10','red_bull':'35','red_cup':'36','ritz_crackers':'10','softsoap_clear':'37','softsoap_gold':'37','softsoap_green':'37','softsoap_purple':'37','softsoap_white':'37','south_beach_good_to_go_dark_chocolate':'10','south_beach_good_to_go_peanut_butter':'10','spam':'8','spongebob_squarepants_fruit_snaks':'10','suave_sweet_guava_nectar_body_wash':'38','sunkist_fruit_snacks_mixed_fruit':'10','tapatio_hot_sauce':'11','v8_fusion_peach_mango':'39','v8_fusion_strawberry_banana':'39','vo5_extra_body_volumizing_shampoo':'38','vo5_split_ends_anti_breakage_shampoo':'38','vo5_tea_therapy_healthful_green_tea_smoothing_shampoo':'38','white_rain_sensations_apple_blossom_hydrating_body_wash':'38','white_rain_sensations_ocean_mist_hydrating_body_wash':'38','white_rain_sensations_ocean_mist_hydrating_conditioner':'38','windex':'40','zilla_night_black_heat':'41'}
# mapping = {'3m_high_tack_spray_adhesive':'1'}
# st()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


join = os.path.join
tree = pickle.load(open("template.tree","rb"))
mod_name = "gg"
base_dir =  "/projects/katefgroup/datasets/bigbird_processed"
bigbird_loc = f"{base_dir}/npy/{mod_name}/*"
trees_dir = f"{base_dir}/{mod_name}/trees_updated/train"
trees_display_dir = f"{mod_name}/trees_updated/train"
mkdir_p(trees_dir)
all_files = glob.glob(bigbird_loc)
st()
for file in all_files:
	if "*" not in file:
		val = pickle.load(open(file,"rb"))
		bbox_camR = val["bbox_camR"]
		obj_name = val["obj_name"]
		tree_copy = copy.deepcopy(tree)
		file_name = file.split("/")[-1]
		tree_copy.bbox_camR_corners = bbox_camR
		tree_copy.word = obj_name
		tree_file_name = join(trees_dir,file_name.replace(".p",".tree"))
		display_file_name = join(trees_display_dir,file_name.replace(".p",".tree"))
		pickle.dump(tree_copy,open(tree_file_name,"wb"))
		val["tree_seq_filename"] = display_file_name
		pickle.dump(val,open(file,"wb"))
		print("done",file)
	# st()
	# print("done")

