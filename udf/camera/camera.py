'''
camera_recog function:
Used for recognizing faces in camera mode:
    -> detect face
    -> align face

create_manual_data function:
Used for input new user's face features in camera mode:
    -> Images from Video Capture
    -> detect the face
    -> crop the face and align it
    -> face is then categorized in 3 types: Center, Left, Right
    -> Extract 128/512D vectors(face features)
    -> Append each newly extracted face 128D vector to its corresponding position type (Center, Left, Right)
    -> Press Q to stop capturing
    -> Find the center (the mean) of those 128D vectors in each category.
    -> Save
'''

import cv2
import time
import numpy as np
import json

def camera_recog(model, extract_feature, face_detect, aligner, findPeople):
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0);  # get input from webcam
    while True:
        # frame has shape of (720,1080,3)
        _, frame = vs.read();

        # u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame, 40);  # min face size is set to 40x40
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160, frame, landmarks[:, i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else:
                print("Align face failed")  # log
        if (len(aligns) > 0):

            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr, positions, model)

            for (i, rect) in enumerate(rects):
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                              (255, 0, 0))  # draw bounding box for the face
                cv2.putText(frame, f'{recog_data[i][0]}' + " - " + f'{recog_data[i][1] / 100:.2%}', (rect[0], rect[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break



def create_manual_data(model, extract_feature, face_detect, aligner):
    vs = cv2.VideoCapture(0);  # get input from webcam
    print("Please input new user ID:")
    new_name = input();  # ez python input()
    if model == 'VGG':
        f = open('./face_data/facerec_VGG.txt', 'r')
    if model == 'FaceNet':
        f = open('./face_data/facerec_FaceNet.txt', 'r')
    if model == 'CASIA':
        f = open('./face_data/facerec_CASIA.txt', 'r')

    data_set = json.loads(f.read());
    person_imgs = {"Left": [], "Right": [], "Center": []};
    person_features = {"Left": [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame, 70);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160, frame, landmarks[:, i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if model == 'VGG':
        print("Using pretrained VGGFace model.")
    if model == 'FaceNet':
        print("Using pretrained FaceNet model.")
    if model == 'CASIA':
        print("Using pretrained CASIA model.")

    for pos in person_imgs:  # there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]
    data_set[new_name] = person_features;

    if model == 'VGG':
        f = open('./face_data/facerec_VGG.txt', 'w')
    if model == 'FaceNet':
        f = open('./face_data/facerec_FaceNet.txt', 'w')
    if model == 'CASIA':
        f = open('./face_data/facerec_CASIA.txt', 'w')

    f.write(json.dumps(data_set))
