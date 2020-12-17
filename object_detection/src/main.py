import numpy as np
import cv2
import datetime

from acl_model import Model
from acl_resource import AclResource

# this model has 21 categories inlucing background
vggssdLabel = {
0:"background",1:"aeroplane",2:"bicycle",3:"bird",4:"boat",5:"bottle",6:"bus",7:"car",
8:"cat",9:"chair",10:"cow",11:"diningtable",12:"dog",13:"horse",14:"motorbike",15:"person",
16:"pottedplant",17:"sheep",18:"sofa",19:"train",20:"tvmonitor"}


def PostProcessing(resultList, conf_thres):
	if resultList is not None:
		# use the second list of model output only
		detections = resultList[1]
		# iterate detected bboxes and select
		for i in range(detections.shape[1]):
			# bboxes info: 0th element is not used. 1st is bbox label, 2nd is bbox confidence score, 3rd~6th are bbox coordinates 
			det_label = int(detections[0,i,1])

			det_conf = detections[0,i,2]
			det_xmin = detections[0,i,3]
			det_ymin = detections[0,i,4]
			det_xmax = detections[0,i,5]
			det_ymax = detections[0,i,6]
			# select bboxes's confidence score are higher than threshold(adjustable), plot and save
			if conf_thres <= det_conf and 1>=det_conf:
				if det_label<=20:
					print("detected object is :", vggssdLabel[det_label], det_conf)
				
					xmin = int(round(det_xmin * image.shape[1]))
					ymin = int(round(det_ymin * image.shape[0]))
					xmax = int(round(det_xmax * image.shape[1]))
					ymax = int(round(det_ymax * image.shape[0]))
					cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),1)
					font = cv2.FONT_HERSHEY_SIMPLEX 
					text = str(vggssdLabel[det_label])+ ": " + str(det_conf)
					cv2.putText(image, text, (xmin,ymin), font, 0.5, (0,0,255), 1)
				else:
					print("wrong detection:", det_label, det_conf)
			else:
				continue

		cv2.imwrite("./out/output.jpg",image)

	else:
		print("graph inference failed")


def CreateGraphWithoutDVPP(model_name):
	acl_resource = AclResource()
	acl_resource.init()

	MODEL_PATH = model_name + ".om"
	model = Model(acl_resource, MODEL_PATH)

	return model

def PreProcessing(image):
	# convert image data type
	image = image.astype('float32')
	
	# resize image
	image = cv2.resize(image, (300,300))
	
	# reduce mean for B,G,R channels 
	b,g,r = cv2.split(image)
	b -= 104
	g -= 117
	r -= 123

	image = cv2.merge([b,g,r])

	# convert image format from HWC to CHW
	image_new = np.transpose(image, (2,0,1)).copy()
	
	return image_new

if __name__ == '__main__':

	# model name, input image file, and confidence score for bboxes
	model_name = 'vgg_ssd'
	img_file = 'test_img/ship.jpg'
	conf_thres = 0.85
	# initialize graph and resource
	acl_resource = AclResource()
	acl_resource.init()

	# create model
	MODEL_PATH = model_name + ".om"
	print("MODEL_PATH:", MODEL_PATH)
	model = Model(acl_resource, MODEL_PATH)

	# load image file 
	image = cv2.imread(img_file)

	#image preprocessing
	input_image = PreProcessing(image)


	# model inference 
	resultList  = model.execute([input_image])
    
	# postprocessing: select bboxes with scores higher than confidence score(adjustable), plot bboxes/labels on the image and save to output dir
	PostProcessing(resultList, conf_thres)




