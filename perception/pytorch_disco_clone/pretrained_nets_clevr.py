feat_init = ""
view_init = ""
flow_init = ""
emb2D_init = ""
vis_init = ""
occ_init = ""
ego_init = ""
tow_init = ""
preocc_init = ""

# emb_dim = 8
# # occ_cheap = False
# feat_dim = 32
# feat_do_vae = False

# view_depth = 32
# view_pred_rgb = True
# view_use_halftanh = True
# view_pred_embs = False

# occ_do_cheap = False
# this is the previous winner net, from which i was able to train a great flownet in 500i
feat_init = "02_m280x56x280_1e-3_F32_Oc_c1_s1_aet_aev_trainer_occ_lr3"
occ_init = "02_m280x56x280_1e-3_F32_Oc_c1_s1_aet_aev_trainer_occ_lr3"
# view_init = "04_m128x32x128_p64x192_1e-3_F32fr_Oc_c1_s1_V_d32_c1_E_s.1_a1_b.1_i1_j.1_caus2i6c1o0t_b13"
# emb2D_init = "04_m128x32x128_p64x192_1e-3_F32lri_Oc_c1_s1_V_d32_c1_E2_s.01_m1_e.1_n2_d32_E3_s.01_m1_e.1_n2_d16_caus2i6c1o0t_caus2i6c1o0v_money43"
