# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:42:13 2025

@author: janisha
"""



#importing  the libraries
import cv2
import face_recognition

#loading the original image
original_image = cv2.imread('images/person_test.jpg')

#load the sample images and get the 128 face embeddings from them
deepika_image = face_recognition.load_image_file('images/deepika.jpg')
deepika_image_encoding = face_recognition.face_encodings(deepika_image)[0]
priyanka_image = face_recognition.load_image_file('images/priyanka.jpg')
priyanka_image_encoding = face_recognition.face_encodings(priyanka_image)[0]
#save the encodings and the corresponding labels in separate arrays in the same order
known_face_encodings = [deepika_image_encoding,priyanka_image_encoding]
known_face_names = ['Deepika Padukon','Priyanka Chopra']
#load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file('images/person_test.jpg')
all_face_locations = face_recognition.face_locations(image_to_recognize,model='hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)
two_persons_image = face_recognition.load_image_file('images/person_test.jpg')

#loop through each face location and face encodingsto found in the unknown image
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    #string to hold the label
    name_of_person = 'Unknown faces'
    #check if the all matches have at least one item
    #if yes , get the index number of face that is located in the first index of all matches
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    #draw rectangle around the face
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    #write name below face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image,name_of_person,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
    cv2.imshow('Identified Faces',original_image)
    