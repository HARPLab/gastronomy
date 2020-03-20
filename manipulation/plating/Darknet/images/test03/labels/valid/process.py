import glob, os# Current directory
#manually putting in path file for now
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = '/home/darknet/YOLO_test2/images/trainImages'
#print(current_dir)

# current_dir = '{}/darknetFormat'.format(current_dir)

# Percentage of images to be used for the test set
#percentage_test = 10; #using 100%

# Create and/or truncate train.txt and test.txt
#file_train = open('food-train.txt', 'w')  
file_test = open('food-test.txt', 'w')

# Populate train.txt and test.txt
counter = 1  
#index_test = round(100 / percentage_test)  
#index_test = 0 #using all of the data
#for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
#    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
#    if counter == index_test:
#        counter = 1
#        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
#        # file_test.write(current_dir + "/" + title + ext + "\n")
#    else:
#        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
#        # file_train.write(current_dir + "/" + title + ext + "\n")
#        counter = counter + 1
index_test = 0 #using all of the data
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.png")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
#    if counter == index_test:
#        counter = 1
    file_test.write(current_dir + "/" + title + '.png' + "\n")
#        # file_test.write(current_dir + "/" + title + ext + "\n")
#    else:
#        file_train.write(current_dir + "/" + title + '.png' + "\n")
#        # file_train.write(current_dir + "/" + title + ext + "\n")
#        counter = counter + 1

