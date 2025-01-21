#importing the libraries
import cv2
import dlib
import face_recognition

#printing the versions
print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)

#loading the imae to detect
image_test = cv2.imread('test_img.jpg')


#showing the current image with title
cv2.imshow('Image',image_test)



# Resizing the image
#desired_width = 800  
#desired_height = 600 
#resized_image = cv2.resize(image_test, (desired_width, desired_height))
#cv2.imshow('Resized Image', resized_image)
