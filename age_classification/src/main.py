import numpy as np
import argparse
import random
import copy

import cv2
import os
import datetime

from acl_model import Model
from acl_resource import AclResource

def PostProcessing(resultList):
	if resultList is not None:
		# age dictionary to map the result index to age range
		age_dict = {0:'(0, 2)',1:'(4, 6)',2:'(8, 12)',3:'(15, 20)',4:'(25, 32)',5:'(38, 43)',6:'(48, 53)',7:'(60, 100)'}
		age_res = age_dict[np.argmax(resultList)]
		print("age of this person is in the range of:", age_res)

	else:
		print("graph inference failed")


def CreateGraphWithoutDVPP(model_name):
	acl_resource = AclResource()
	acl_resource.init()

	MODEL_PATH = "./om_model/" + model_name + ".om"
	model = Model(acl_resource, MODEL_PATH)

	return model

def PreProcessing(img_file):
	image = cv2.imread(img_file)
	# image data type conversion and resizing 
	image = image.astype('float32')
	image = cv2.resize(image, (227,227))
	return image

if __name__ == '__main__':
	# model name and input image file
	model_name = 'inception_age'
	img_file = 'test_img/age1.jpeg'

	# initialize graph
	acl_resource = AclResource()
	acl_resource.init()

	# create model
	MODEL_PATH = model_name + ".om"
	print("MODEL_PATH:", MODEL_PATH)
	model = Model(acl_resource, MODEL_PATH)

	# image preprocessing
	input_image = PreProcessing(img_file)

	# model inference 
	resultList  = model.execute([input_image])	

	# image postprocessing: print age range 
	PostProcessing(resultList)




