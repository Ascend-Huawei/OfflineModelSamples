import numpy as np
import sys

import cv2
import os
import datetime

sys.path.append('..')
from acl_model import Model
from acl_resource import AclResource

# preprocessing to convert image to correct colour format, size and channels format
def PreProcessing(image):
	# BGR to RGB colour conversion
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# resize image
	image = cv2.resize(image, (96, 112))
	# NHWC to NCHW channel format conversion 
	image = np.asarray(image, dtype=np.float32).transpose([2,0,1]).copy()
	return image

# post processing on the output list to calculate similarity between two faces
def PostProcessing(resultList1, resultList2):
	# convert result face vectors from list of arrays to flat lists 
	a = [r for r in resultList1.flat]
	b = [r for r in resultList2.flat]
	# calculate cosine similarity distance between the two face vectors
	cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
	print('Cosine similarity between faces: ', cos_sim)

if __name__ == '__main__':
	
	model_name = 'sphereface'
	img_file1 = cv2.imread('face1.jpeg')
	img_file2 = cv2.imread('test_human.jpg')
	
	# initialize acl resource
	acl_resource = AclResource()
	acl_resource.init()

	# load offline model
	MODEL_PATH = model_name + ".om"
	print("MODEL_PATH:", MODEL_PATH)
	model = Model(acl_resource, MODEL_PATH)

	# load image file 
	input_image1 = PreProcessing(img_file1)
	print(input_image1.shape)
	input_image2 = PreProcessing(img_file2)
	print(input_image2.shape)

	# om model inference
	resultList1  = model.execute([input_image1])[0].copy()
	resultList2  = model.execute([input_image2])[0].copy()

	#postprocessing to compare results
	PostProcessing(resultList1, resultList2)
	
