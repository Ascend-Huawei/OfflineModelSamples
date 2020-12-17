# Object Detection
**21** categories can be detected:

**{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
"cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"}**

## Model Description

#### Model Name: VGG_SSD

#### Model Conversion to ATC Offline Model
* Step 1: Download the network and weight files to your project folder 'object_detection/src':

  - network:
  https://obs-book.obs.cn-east-2.myhuaweicloud.com/shaxiang/C73/vgg_ssd.prototxt
  - weights:
  https://obs-book.obs.cn-east-2.myhuaweicloud.com/shaxiang/C73/vgg_ssd.caffemodel

* Step 2: Navigate to the same project folder and execute the following command in the terminal to convert model to offline model (.om) format:

  **atc --output_type=FP32 --input_shape="data:1,3,300,300" --weight="vgg_ssd.caffemodel" --input_format=NCHW --output="vgg_ssd" --soc_version=Ascend310 --framework=0 --save_original_model=false --model="vgg_ssd.prototxt"**


#### Model Input
- **Input Shape**: [1,3,300,300]
- **Input Format** : NCHW
- **Input Type**: FLOAT32

#### Model Output
- 2 lists. Only the 2nd list is used.
  - **1st list shape**: [1,8]
  - **2nd list shape**: [1, 200, 8]
- For the second list: **200** represents 200 bounding boxes. **0-8** describe information of bounding box as below:
  - **0 position**: not used
  - **1 position**: label
  - **2 position**: confidence score
  - **3 position**: top left x coordinate
  - **4 position**: top left y coordinate
  - **5 position**: bottom right x coordinate
  - **6 position**: bottom right y coordinate
  - **7 position**: not used
  
## Sample Code:
  - Codes in https://github.com/Ascend-Huawei/OfflineModelSamples/tree/main/object_detection/src create a sample to quickly get how the model works, preprocessing, inference, postprocessing are already included.
  - Preprocessing: 
    - **Image Resize**: 300*300
    - **Image Type**: FLOAT32
    - **Mean Reduce**
      - **B**: 104
      - **G**: 117
      - **R**: 123
    - Change order from **[300, 300, 3]**(HWC) to **[3, 300, 300]**(CHW)
  - Postprocessing:
    - select bboxes with score higher than threshold score(0.85 here), filter bboxes with score higher than 1(which should be wrong bboxes detected)
    - plot bbox, category name and confidence score on image 
    
    *NOTE:* **threshold score** is an adjustable parameter here, you could simple change it according to your projects objective.
    
  - Result would be saved in "output/" folder, show as below:
    ![image](https://github.com/Ascend-Huawei/OfflineModelSamples/blob/main/object_detection/src/out/output.jpg)
    Photo By: Matt Hochberg
  - To run code，simply using commands below in the terminal:
  
    ``` 
    cd object_detection/src
    python3 main.py 
    ``` 
  
