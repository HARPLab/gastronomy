

# for foldername in CLEVR_SINGLE_LARGE_OBJ_256_A CLEVR_SINGLE_LARGE_OBJ_256_B CLEVR_SINGLE_LARGE_OBJ_256_C CLEVR_SINGLE_LARGE_OBJ_256_D CLEVR_SINGLE_LARGE_OBJ_256_E CLEVR_SINGLE_LARGE_OBJ_256_F
# do
# 	echo $foldername
# 	scp -r /projects/katefgroup/datasets/clevr_veggies/$foldername/trees_updated  aws:/projects/katefgroup/datasets/clevr_veggies/$foldername/
# done



for foldername in CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_A CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_B CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_C CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_D CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_E CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_F
do
	echo $foldername
	scp -r /projects/katefgroup/datasets/clevr_veggies/$foldername/trees_updated  aws:/projects/katefgroup/datasets/clevr_veggies/$foldername/
done
