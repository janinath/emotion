# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:18:56 2025

@author: janisha
"""

#importing  the libraries
import cv2
import face_recognition

#load the image to detect
image_to_detect=cv2.imread('images/two_persons.jpg')

#test whether the imgae is loading
#cv2.imshow('test', image_to_detect)


#detect all faces in the image
#find all face locations using face_locations() functions
#model can be 'cnn' or 'hog'
#number_of_times_to_upsample=1 higher and detect more faces 
 
all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')

#print the number of faces detetcted
print('there are {} no of faces in this image'.format(len(all_face_locations)))


#Now change the model 

#all_face_locations = face_recognition.face_locations(image_to_detect,model='cnn')
#print('there are {} no of faces in this image'.format(len(all_face_locations)))

#since the cnn is taking more time we use hog


#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {},at top:{},right:{},bottom:{},left :{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    cv2.imshow("Face No"+str(index+1),current_face_image)
