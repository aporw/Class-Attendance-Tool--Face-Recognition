'''
Main program

Use FaceNet architecture:
    Camera mode:
        To do face recognition:
        Use pre-trained FaceNet model on VGG dataset to recognize faces:
        main.py --mode="VGGcamera"
        Use pre-trained FaceNet model on CASIA dataset to recognize faces:
        main.py --modeCASIAcamera"
        Use pre-trained FaceNet model on old 128D dataset to recognize faces:
        main.py --mode="FaceNetcamera"

        To input new user:
        Use pre-trained FaceNet model on VGG dataset to input faces:
        main.py --mode="VGGinput"
        Use pre-trained FaceNet model on CASIA dataset to input faces:
        main.py --modeCASIAinput"
        Use pre-trained FaceNet model on old 128D dataset to input faces:
        main.py --mode="FaceNetinput"

    Picture mode(only implemented on VGG dataset):
        To do face recognition:
        main.py --mode="VGGpicture"

        To input one new user:
        main.py --mode="VGGpic_input_one"
        To input all new users together:
        main.py --mode="VGGpic_input_all"


Credit: David Vu for part of the code.
        DavidSandBerg for the pretrained models.
'''




import cv2
from udf.align_custom import AlignCustom
from udf.mtcnn_detect import MTCNNDetect
from udf.tf_graph import FaceRecGraph
import argparse
import sys
import os
import imutils
from udf.face_model import FaceFeature
from udf.pic.pic import getNames, readNames
from udf.find_people import findPeople
from udf.camera.camera import camera_recog, create_manual_data
from imutils import paths
TIMEOUT = 10 #10 seconds

# define main program
def main(args):
    mode = args.mode
    if mode == "VGGcamera" :
        extract_feature = FaceFeature(FRGraph, 'models/20180402-114759/20180402-114759.pb')
        print("Using pre-trained VGGFace model.")
        camera_recog('VGG', extract_feature, face_detect, aligner, findPeople)
    elif mode == "FaceNetcamera":
        print("Using pre-trained old128D model.")
        extract_feature = FaceFeature(FRGraph, 'models/20170512-110547/20170512-110547.pb')
        camera_recog('FaceNet', extract_feature, face_detect, aligner, findPeople)
    elif mode == "CASIAcamera":
        print("Using pre-trained CASIA model.")
        extract_feature = FaceFeature(FRGraph, 'models/20180408-102900/20180408-102900.pb')
        camera_recog('CASIA', extract_feature, face_detect, aligner, findPeople)
    elif mode == "VGGinput":
        print("Using pre-trained VGGFace model.")
        extract_feature = FaceFeature(FRGraph, 'models/20180402-114759/20180402-114759.pb')
        create_manual_data('VGG', extract_feature, face_detect, aligner)
    elif mode == "FaceNetinput":
        print("Using pre-trained old128D model.")
        extract_feature = FaceFeature(FRGraph, 'models/20170512-110547/20170512-110547.pb')
        create_manual_data('FaceNet', extract_feature, face_detect, aligner)
    elif mode == "CASIAinput":
        print("Using pre-trained CASIA model.")
        extract_feature = FaceFeature(FRGraph, 'models/20180408-102900/20180408-102900.pb')
        create_manual_data('CASIA', extract_feature, face_detect, aligner)
    elif mode == "VGGpicture":
        print("Starting Recognize picture using FaceNet Model trained on VGG dataset.")
        extract_feature = FaceFeature(FRGraph, 'models/20180402-114759/20180402-114759.pb')
        img = input("Enter image's direct location: ")
        frame = cv2.imread(img)
        frame = imutils.resize(frame, width=500)
        getNames(frame, face_detect, extract_feature, findPeople, aligner)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == "VGGpic_input_one":
        print("Starting Input picture using FaceNet Model trained on VGG dataset.")
        extract_feature = FaceFeature(FRGraph, 'models/20180402-114759/20180402-114759.pb')
        dirs = input("Enter path to input directory of faces + images: ")
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(dirs))
        readNames(imagePaths, face_detect, extract_feature, aligner)
    elif mode == "VGGpic_input_all":
        print("Starting Input picture using FaceNet Model trained on VGG dataset.")
        extract_feature = FaceFeature(FRGraph, 'models/20180402-114759/20180402-114759.pb')
        directory = input("Enter path to directory of all image directories: ")
        all_folders = [i for (i,j,k) in os.walk(directory) if i != directory]
        for i in range(len(all_folders)):
            try:
                print(f"{all_folders[i]}")
                print("[INFO] quantifying faces ...")
                imagePaths = list(paths.list_images(all_folders[i]))
                readNames(imagePaths, face_detect, extract_feature, aligner)
            except:
                print(f"Not available for {all_folders[i]}")
    else:
        raise ValueError("Unimplemented mode")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="FaceNetcamera")
    args = parser.parse_args(sys.argv[1:])
    # Create a graph called FRGraph
    FRGraph = FaceRecGraph()
    # Create a graph called MTCNNGraph
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=1) #scale_factor, rescales image for faster detection
    main(args)
