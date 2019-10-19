'''


Need to update

'''

import cv2
import imutils
import json
import numpy as np
import os
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.utils import np_utils
from imutils import paths
from keras.models import model_from_yaml

def getNames(frame, face_detect, extract_feature, findPeople, aligner):

    minsize = 40

    rects, landmarks = face_detect.detect_face(frame,minsize);#min face size is set to 40x40
    aligns = []
    positions = []

    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
        if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
            aligns.append(aligned_face)
            positions.append(face_pos)
        else: 
            print("Align face failed") #log        
    if(len(aligns) > 0):
       
        features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions, 'VGG_pic')
        
        for (i,rect) in enumerate(rects):
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
            cv2.putText(frame,f'{recog_data[i][0]}'+" - "+f'{recog_data[i][1]/100:.2%}',(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    

def readNames(imagePaths, face_detect, extract_feature, aligner):
    f = open('./face_data/facerec_VGG_pic.txt', 'r')

    data_set = json.loads(f.read());

    person_imgs = {"Left": [], "Right": [], "Center": []}
    person_features = {"Left": [], "Right": [], "Center": []}
	
	#Ankur code
    print(imagePaths)
    #NewimagePaths = imagePaths

    NewimagePaths = new_images(imagePaths)
    print(NewimagePaths)
    # loop over the image paths
    for (i, imagePath) in enumerate(NewimagePaths):

        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(NewimagePaths)))
        name = imagePath.split(os.path.sep)[-2]  # -2 is the name of directory, -1 is the name of the file.
        # load the input image and convert it from BGR (OpenCV ordering)

        frame = cv2.imread(imagePath)

        rects, landmarks = face_detect.detect_face(frame, 40);  # min face size is set to 80x80

        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160, frame, landmarks[:, i])

            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                # cv2.imshow("Captured face", aligned_frame)
                print('Recognized position:' + str(pos))

    for pos in person_imgs:
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]
    data_set[name] = person_features

    f = open('./face_data/facerec_VGG_pic.txt', 'w')

    f.write(json.dumps(data_set))

	
def new_images(pathOfImages):	
    #print("coming here")
    if (len( pathOfImages))==0:
        print(pathOfImages, "folder has no image")
        exit()     
    transformed_images(pathOfImages)
    AE_images(pathOfImages)
    arr = pathOfImages[0].split(os.path.sep)
    dirctory = (os.path.sep).join(arr[0:len(arr)-1])

    allnew = list(paths.list_images(dirctory))
    #print(dirctory)
    #print(allnew)	
    return allnew
	
	
def transformed_images(pathOfImages):

    for ipath in pathOfImages:
        print(ipath)
        img=cv2.imread(ipath)
        horizontal_img = img.copy()
        vertical_img = img.copy()
        both_img = img.copy()
        		 
        		# flip img horizontally, vertically,
        		# and both axes with flip()
        horizontal_img = cv2.flip( img, 0 )
        vertical_img = cv2.flip( img, 1 )
        both_img = cv2.flip( img, -1 )
        
        name = ipath.split(os.path.sep)[-1]
        arr = ipath.split(os.path.sep)
        dirctory = (os.path.sep).join(arr[0:len(arr)-1])
        cv2.imwrite(dirctory+os.path.sep +name+"_hori.jpg",horizontal_img)
        cv2.imwrite(dirctory+os.path.sep +name+"_ver.jpg", vertical_img)
        cv2.imwrite(dirctory+os.path.sep +name+"_both.jpg",both_img)
	
	
def AE_images(pathOfImages):
    print(" image for AE: ", len(pathOfImages))
    yaml_file = open('./models/model_AllImage.json', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("./models/model_AllImage.h5")
    print("Loaded model from disk")
    for ipath in pathOfImages:
        print(ipath)
        features=cv2.imread(ipath)
		
        features_resized = cv2.resize(features,(500,800))
        resize = features_resized.shape
        pred = loaded_model.predict(features_resized.reshape(-1,resize[0], resize[1],resize[2]))
        transformed = pred.reshape(resize[0], resize[1],-1)
        
        arr = ipath.split(os.path.sep)
        name = ipath.split(os.path.sep)[-1]

        dirctory = (os.path.sep).join(arr[0:len(arr)-1])
		
   
        cv2.imwrite(dirctory+os.path.sep +name+"_AE.jpg",transformed)
    print("Autoencoder done.")
		


