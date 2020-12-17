import numpy as np
import sys
import cv2
import datetime

from acl_model import Model
from acl_resource import AclResource

from PIL import Image

# this mask detection model can detect 3 catogeries: face, person, mask
labels = ["face","person", "mask"]

# model input requirement
MODEL_WIDTH = 640
MODEL_HEIGHT = 352
class_num = 3

# anchor settings for postprocessing
stride_list = [8, 16, 32]
anchors_1 = np.array([[10, 13], [16, 30], [33, 23]]) / stride_list[0]
anchors_2 = np.array([[30, 61], [62, 45], [59, 119]]) / stride_list[1]
anchors_3 = np.array([[116, 90], [156, 198], [163, 326]]) / stride_list[2]
anchor_list = [anchors_1, anchors_2, anchors_3]

# adjustable parameters: change them as you need (used in postprocessing). 
# because mask confidence score is always lower while person is higher, here we have a seperate threshold for person(0.2), and for mask(0.08)
mask_conf_threshold = 0.08
person_conf_threshold = 0.2
iou_threshold = 0.3

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def CreateGraphWithoutDVPP(model_name):
    
    acl_resource = AclResource()
    acl_resource.init()

    MODEL_PATH = "./om_model/" + model_name + ".om"
    model = Model(acl_resource, MODEL_PATH)
	
    return model

def PreProcessing(img_path):
    image = Image.open(img_path)
    img_h = image.size[1]
    img_w = image.size[0]
    net_h = MODEL_HEIGHT
    net_w = MODEL_WIDTH
    
    # image resize and shift
    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h

    image_ = image.resize( (new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    
    # convert image data type
    new_image = new_image.astype(np.float32)
    # image normalization
    new_image = new_image / 255

    return new_image, image


def overlap(x1, x2, x3, x4):
    # calculate overlap length of width/height of 2 boxes.
    # x1 and x2 belong to the first box, x3 and x4 belong to the second box
    # e.g. a and b are 2 boxes, and x1= xa_min, x2 = xa_max, x3 = xb_min, x4 = xb_max
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    # calculate iou between two bboxes
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    # if w<0 or h < 0, those two boxes are not overlapped 
    if w <= 0 or h <= 0:
        return 0
    # overlapped area between 2 boxes
    inter_area = w * h
    # union area between 2 boxes
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    # return overlap percentage
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    # non maximum supression for bboxes, thres is adjustable
    res = []
    # see nms algorithm for detail
    # firstly sort the boxes with condifence score. start from the box with the highest confidence score, sequentially calculate iou between 2 boxes
    # using a dictionay to record 
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
	# only keep boxes with iou lower than iou threshold 
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    # decode bboxes from model output feature map
   
    def _sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    h, w, _ = conv_output.shape
    
    pred = conv_output.reshape((h * w, 3, 5 + class_num))

    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
    pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
    pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
    pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    
    # select bboxes with confidence score higher then threshold
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    # conf_threshold is adjustable
    # pred[:, 4] here is the confidence score of box
    pred = pred[pred[:, 4] >= mask_conf_threshold]
    # pred[:, 5] here is label of box
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
    # if pred[ix, 5] == 0, label of this box would be person. if pred[ix, 5] == 1, label would be face
    # use person_conf_threshold(higher than mask_conf_threshold here, to filter boxes with person/face) 
        if pred[ix, 5] == 0 or pred[ix,5] == 1:
                if pred[ix, 4] < person_conf_threshold:
                    continue

        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    
    return all_boxes

def convert_labels(label_list):
    # convert labels from numbers to strings
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def post_process(infer_output, origin_img):
    print("post process")
    # parameter settings for model postprocessing (yolo postprocessing)
    result_return = dict()
    img_h = origin_img.size[1]
    img_w = origin_img.size[0]
    scale = min(float(MODEL_WIDTH) / float(img_w), float(MODEL_HEIGHT) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    shift_x_ratio = (MODEL_WIDTH - new_w) / 2.0 / MODEL_WIDTH
    shift_y_ratio = (MODEL_HEIGHT- new_h) / 2.0 / MODEL_HEIGHT
    class_num = len(labels)
    num_channel = 3 * (class_num + 5)
    x_scale = MODEL_WIDTH / float(new_w)
    y_scale = MODEL_HEIGHT / float(new_h)
    all_boxes = [[] for ix in range(class_num)]
    # output would be 3 feature maps
    for ix in range(3):
        pred = infer_output[2 - ix].reshape((MODEL_HEIGHT // stride_list[ix], MODEL_WIDTH // stride_list[ix], num_channel))
        anchors = anchor_list[ix]
        boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_num)]

    res = apply_nms(all_boxes, iou_threshold)
    # obtain final boxes 
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return

if __name__ == '__main__':
	
    model_name = 'mask_detection'
    # input image:  simply modify it to the image you are gonna use
    img_file = 'test_img/mask_1.jpg'

    # initialize graph and resource
    acl_resource = AclResource()
    acl_resource.init()
	
    # model path
    MODEL_PATH = model_name + ".om"
    print("MODEL_PATH:", MODEL_PATH)
    # create mask detection model
    model = Model(acl_resource, MODEL_PATH)

    # original bgr image for plotting
    ori_img = cv2.imread(img_file)
    # image preprocessing: "data" is the preprocesses image for inference, "orig" is the original image
    data, orig = PreProcessing(img_file)
	
    # om model inference 
    resultList  = model.execute([data])
	
    # image postprocessing: decoding model output to bboxes information, using nms to select bboxes. return detected bboxes info: axis, confidence score and category
    result_return = post_process(resultList, orig)

    print("result = ", result_return)
	
    # plot bbox/label on the image and save
    for i in range(len(result_return['detection_classes'])):
        box = result_return['detection_boxes'][i]
        class_name = result_return['detection_classes'][i]
        confidence = result_return['detection_scores'][i]
        cv2.rectangle(ori_img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i%6])
        p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
        out_label = class_name            
        cv2.putText(ori_img, out_label, p3, cv2.FONT_ITALIC, 0.6, colors[i%6], 1)

    cv2.imwrite("./out/mask_output.jpg", ori_img)
