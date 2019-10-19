"""
face_data Data Structure:
{
"Person ID": {
    "Center": [[128/512D vector]],
    "Left": [[128/512D vector]],
    "Right": [[128/512D Vector]]
    }
}
This function basically does a simple linear search for
the 128/512D vector with the min distance to the 128/512D vector of the face on screen

"""
import json
import numpy as np
import sys

def findPeople(features_arr, positions, model, thres=0.6, percent_thres=60):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    if model == 'VGG':
        f = open('./face_data/facerec_VGG.txt','r')
    elif model == 'FaceNet':
        f = open('./face_data/facerec_FaceNet.txt','r')
    elif model == 'CASIA':
        f = open('./face_data/facerec_CASIA.txt','r')
    elif model == 'VGG_pic':
        f = open('./face_data/facerec_VGG_pic.txt','r')
    else:
        return None
    data_set = json.loads(f.read());
    returnRes = [];
    for (i,features_D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes