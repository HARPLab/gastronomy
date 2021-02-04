import glob
import subprocess

annotation_file_paths = glob.glob('/home/klz/food_training_annotations/*.png')
test_image_file_paths = glob.glob('/home/klz/food_test_images/*.png')
unlabeled_dir = '/home/klz/unlabeled_food_images/'

for file_path in test_image_file_paths:
    file_name = file_path[file_path.rfind('/')+1:]
    print(file_name)
    file_is_annotated = False
    for annotation_file_path in annotation_file_paths:
        if file_name in annotation_file_path:
            file_is_annotated = True
            
    if not file_is_annotated:
        outfile_path = unlabeled_dir + file_name
        command = "mv " + file_path + " " + outfile_path
        subprocess.run(command, shell=True)