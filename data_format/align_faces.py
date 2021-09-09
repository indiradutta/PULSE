import torchvision
import torchvision.utils as vutils

import scipy
import scipy.ndimage
import PIL
import PIL.Image

import numpy as np
import os
import dlib
from pathlib import Path

#from bicubic import BicubicDownSample
from image_preprocess import Preprocessing

def make_data(input_dir_path=None, output_size=32, output_dir_path=None):

  if not os.path.exists(output_dir_path):
    output_dir_path = 'processed images/'
    os.mkdir(output_dir_path)
  
  preprocess = Preprocessing(output_size)

  for image in os.listdir(input_dir_path):
    faces = preprocess.align_faces(os.path.join(input_dir_path,image))

    for i,face in enumerate(faces):
      factor = 1024//output_size
      #downsample = BicubicDownSample(factor=factor)
      face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0)
      face_tensor = face_tensor[0].cpu().detach().clamp(0, 1)
      #face_tensor_low_res = downsample(face_tensor)[0].cpu().detach().clamp(0, 1)
      face = torchvision.transforms.ToPILImage()(face_tensor)

      print('Saving images to the new directory: ')
      face_image = '{}/image_{}.png'.format(output_dir_path,i)
      vutils.save_image(face_tensor, face_image, padding = 0) 

input_path = input('Enter the path for the input directory: ')
print('Are the images aligned and downscaled?: ')
choice = input('yes or no: ')
if choice=="no":
  size = int(input('Enter the output resolution of the downscaled images (must be in powers of 2): '))
  output_path = input('Do you have a directory made for storing the downscaled images? If yes enter the path: ')
  make_data(input_path, size, output_path)
print('Downscaled images saved to the new directory')