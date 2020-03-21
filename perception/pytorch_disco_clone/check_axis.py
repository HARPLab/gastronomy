import cv2  
import ipdb
st = ipdb.set_trace
path = r'messi.jpg'
   
image = cv2.imread(path) 
# st()
print(image.shape,"image shape")
window_name = 'Image'

# but image is Y,X both pointing downwards
  
# maybe bbox is X,Y

start_point = (205, 100) 
  
end_point = (300, 500) 
  
color = (255, 0, 0) 
  
thickness = 2
st()
image = cv2.rectangle(image, start_point, end_point, color, thickness) 
  
cv2.imshow(window_name, image)  
cv2.waitKey(0)


cv2.imshow(window_name, image[start_point[1]:end_point[1],start_point[0]:end_point[0]])  
cv2.waitKey(0)