# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:35:26 2025

@author: janisha
"""

import face_recognition
import cv2

#get  the webcam #0 (the default one, 1,2 etc means additional attached cams)

webcam_video_stream = cv2.VideoCapture(0)

#load the sample images and get the 128 face embeddings from them
deepika_image = face_recognition.load_image_file('images/deepika.jpg')
deepika_image_encoding = face_recognition.face_encodings(deepika_image)[0]
priyanka_image = face_recognition.load_image_file('images/priyanka.jpg')
priyanka_image_encoding = face_recognition.face_encodings(priyanka_image)[0]
janisha_image = face_recognition.load_image_file('images/jani.jpg')
janisha_image_encoding = face_recognition.face_encodings(janisha_image)[0]
subi_image = face_recognition.load_image_file('images/subi.jpg')
subi_image_encoding = face_recognition.face_encodings(subi_image)[0]
subi_image2 = face_recognition.load_image_file('images/subitha.jpg')
subi_image_encoding2 = face_recognition.face_encodings(subi_image2)[0]

#save the encodings and the corresponding labels in separate arrays in the same order
known_face_encodings = [deepika_image_encoding,priyanka_image_encoding,janisha_image_encoding,subi_image_encoding,subi_image_encoding2]
known_face_names = ['Deepika Padukon','Priyanka Chopra','Janisha','Subitha','Subitha']

#initialize the array to hold all face locations, encodings and labels in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

while True:
    #get the current frame from the video stream as an image
    ret , current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #find all face locations using face locations() functions
    #arguments are image, no_of_times_to_upsample,model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    two_persons_image = face_recognition.load_image_file('images/person_test.jpg')

    #loop through each face location and face encodingsto found in the unknown image
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        #change the position magnitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        #string to hold the label
        name_of_person = 'Unknown faces'
        #check if the all matches have at least one item
        #if yes , get the index number of face that is located in the first index of all matches
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        #draw rectangle around the face
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        #write name below face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,name_of_person,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
        cv2.imshow('Identified Faces',current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()