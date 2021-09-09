import torch
import torchvision

import numpy as np
import sys
import os
import glob
import dlib
import gdown
import json

import scipy
import scipy.ndimage
import PIL
import PIL.Image

from pathlib import Path

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

class Preprocessing():

  def __init__(self, output_size, fl=None):

    #output dimension of the image.
    self.output_size = output_size

    #the path of the shape_predictor model.
    self.fl = fl

    #the dlib face detector.
    self.detector = dlib.get_frontal_face_detector() 

    #if the user doesn't have the shape_predictor model then this will automatically download the model from drive.
    if fl is not None:
      self.predictor = dlib.shape_predictor(self.fl)

    else:
      with open( __PREFIX__+"/config/shape_predictor.json", 'rb') as fp:
        json_file = json.load(fp)
        url = 'https://drive.google.com/uc?id={}'.format(json_file['shape_predictor'])
        gdown.download(url, 'shape_predictor.dat', quiet=False)
        file_new = 'shape_predictor.dat'
        self.predictor = dlib.shape_predictor(file_new)

    #transformation dimension of the image.
    self.transform_size = 1024
  
  def landmark_detection(self, path):

    #The faces are detected using the face detector of dlib
    img = dlib.load_rgb_image(path)
    faces = self.detector(img, 1)

    #the landmark co-ordinates of the faces are extracted 
    shapes = [self.predictor(img, shape) for index, shape in enumerate(faces)]

    #the landmarks of all the faces are converted into a numpy array and stored in a list 
    landmarks = [np.array([[points.x, points.y] for points in shape.parts()]) for shape in shapes]

    return landmarks
  
  def align_faces(self, path):

    #the landmarks are extracted
    landmarks = self.landmark_detection(path)
    imgs = []

    for landmark in landmarks:
        landmark_chin = landmark[0: 17]  # left-right
        landmark_eyebrow_left = landmark[17: 22]  # left-right
        landmark_eyebrow_right = landmark[22: 27]  # left-right
        landmark_nose = landmark[27: 31]  # top-down
        landmark_nostrils = landmark[31: 36]  # top-down
        landmark_eye_left = landmark[36: 42]  # left-clockwise
        landmark_eye_right = landmark[42: 48]  # left-clockwise
        landmark_mouth_outer = landmark[48: 60]  # left-clockwise
        landmark_mouth_inner = landmark[60: 68]  # left-clockwise

        #calculate the auxiliary vectors.
        left_eye = np.mean(landmark_eye_left, axis=0)
        right_eye = np.mean(landmark_eye_right, axis=0)
        mid_eye = (left_eye + right_eye) * 0.5
        eye_to_eye = right_eye - left_eye
        mouth_left = landmark_mouth_outer[0]
        mouth_right = landmark_mouth_outer[6]
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_mid -mid_eye

        #the binocular distance + eye-to-mouth distance is calculated
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]

        #the two sides of the traingle is normalized using the length of the hypotenuse
        x /= np.hypot(*x)

        #the larger one of the normalized binocular distance and normalized eye-to-mouth distance is chosen for reference
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

        #y acts as the reference for the vertical direction
        y = np.flipud(x) * [-1, 1]

        #c acts as the reference position of the face
        c = mid_eye + eye_to_mouth * 0.1

        #four corners of the quadrilateral are formed by translating the reference position up and down, left and right.
        frame = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

        #scaled size of the quadrilateral
        frame_size = np.hypot(*x) * 2

        img = PIL.Image.open(path)

        img = img.transform((self.transform_size, self.transform_size), PIL.Image.QUAD, (frame + 0.5).flatten(),
                            PIL.Image.BILINEAR)
  
        img = img.resize((self.output_size, self.output_size), PIL.Image.ANTIALIAS)

        #save the final aligned image.
        imgs.append(img)

    return imgs