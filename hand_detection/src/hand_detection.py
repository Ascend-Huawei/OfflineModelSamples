import numpy as np
import sys
sys.path.append('../')
import cv2
import datetime

from acl_model import Model
from acl_resource import AclResource

# draw the bounding boxes for all detected hands with confidence greater than a set threshold
def PostProcessing(image, resultList, threshold=0.6):
	num_detections = resultList[0][0].astype(np.int)
	scores = resultList[2]
	boxes = resultList[3]
	bbox_num = 0
	
	# loop through all the detections and get the confidence and bbox coordinates
	for i in range(num_detections):
		det_conf = scores[0,i]
		det_ymin = boxes[0,i,0]
		det_xmin = boxes[0,i,1]
		det_ymax = boxes[0,i,2]
		det_xmax = boxes[0,i,3]

		bbox_width = det_xmax - det_xmin
		bbox_height = det_ymax - det_ymin
		# the detection confidence and bbox dimensions must be greater than a minimum value to be a valid detection
		if threshold <= det_conf and 1>=det_conf and bbox_width>0 and bbox_height > 0:
			bbox_num += 1
			xmin = int(round(det_xmin * image.shape[1]))
			ymin = int(round(det_ymin * image.shape[0]))
			xmax = int(round(det_xmax * image.shape[1]))
			ymax = int(round(det_ymax * image.shape[0]))
			
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
		else:
			continue

	print("detected bbox num:", bbox_num)
	cv2.imwrite("./out/output.jpg",image)

	
def PreProcessing(image):
	# resize image to 300*300, and RGB
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (300,300))
	# type conversion to float 32
	image = image.astype(np.uint8).copy()
	return image


if __name__ == '__main__':
	
	model_name = 'Hand_detection'
	img_file = 'hand.jpeg'

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
	
	# om model inference 
	resultList  = model.execute([input_image])

	# postprocessing and save inference results
	PostProcessing(image, resultList)
	



