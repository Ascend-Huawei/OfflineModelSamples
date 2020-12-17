import numpy as np
import sys 

sys.path.append('../')
import cv2
import os
import datetime

from acl_model import Model
from acl_resource import AclResource

# draw the bounding boxes for all detected faces with confidence greater than a set threshold
def PostProcessing(image, resultList, threshold=0.9):
	detections = resultList[1]
	bbox_num = 0
	
	# loop through all the detections and get the confidence and bbox coordinates
	for i in range(detections.shape[1]):
		det_conf = detections[0,i,2]
		det_xmin = detections[0,i,3]
		det_ymin = detections[0,i,4]
		det_xmax = detections[0,i,5]
		det_ymax = detections[0,i,6]
		bbox_width = det_xmax - det_xmin
		bbox_height = det_ymax - det_ymin
		# the detection confidence and bbox dimensions must be greater than a minimum value to be a valid detection
		if threshold <= det_conf and 1>=det_conf and bbox_width>0 and bbox_height > 0:
			bbox_num += 1
			xmin = int(round(det_xmin * image.shape[1]))
			ymin = int(round(det_ymin * image.shape[0]))
			xmax = int(round(det_xmax * image.shape[1]))
			ymax = int(round(det_ymax * image.shape[0]))
			
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),1)
		else:
			continue

	print("detected bbox num:", bbox_num)
	cv2.imwrite("./out/output.jpg",image)

	
def PreProcessing(image):
	# resize image to 300*300 
	image = cv2.resize(image, (300,300))
	# type conversion to float 32
	image = image.astype('float32')
	# convert channel format from NHWC to NCHW
	image = np.transpose(image, (2, 0, 1)).copy()
	return image


if __name__ == '__main__':
	
	model_name = 'face_detection'
	img_file = 'face1.jpeg'

	# initialize acl resource
	acl_resource = AclResource()
	acl_resource.init()

	#load model
	MODEL_PATH = model_name + ".om"
	print("MODEL_PATH:", MODEL_PATH)
	model = Model(acl_resource, MODEL_PATH)

	# load image file 
	image = cv2.imread(img_file)
	input_image = PreProcessing(image)
	print(input_image.size)
	print(input_image.shape)
	
	# om model inference 
	resultList  = model.execute([input_image])

	# postprocessing and save inference results
	PostProcessing(image, resultList)
	



