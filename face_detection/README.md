# Face Detection
This model detects faces in an input image and returns the bounding box locations.

## Model Description:
Here we are using Caffe-based ResNet10-SSD300 converted to offline model to inference on board. 

Download the weight and network file to the project directory 'face_detection/src':

- Weights: https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/face_detection/face_detection.caffemodel
- Network: https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/face_detection/face_detection.prototxt

Execute the following command from the project directory 'face_detection/src' to convert the pre-trained model for face detection to offline model (.om) format:

**atc --output_type=FP32 --input_shape="data:1,3,300,300" --weight="face_detection.caffemodel" --input_format=NCHW --output="face_detection" --soc_version=Ascend310 
--framework=0 --save_original_model=false 
--model="face_detection.prototxt"**


#### Input
- **Input Shape**: [1,300,300, 3]
- **Input Image Format**: BGR
- **Input Format** : NCHW
- **Input Type**: FLOAT32

#### Output
- Output is a list of 2 numpy arrays with shapes: (1, 8) and (1, 100, 8). Only the 2nd list is used.
  - **1st list shape**: [1,8]
  - **2nd list shape**: [1, 100, 8]
- For the second list: **100** represents 100 bounding boxes. **0-8** describe information of bounding box as below:
  - **0 position**: not used
  - **1 position**: label
  - **2 position**: confidence score
  - **3 position**: top left x coordinate
  - **4 position**: top left y coordinate
  - **5 position**: bottom right x coordinate
  - **6 position**: bottom right y coordinate
  - **7 position**: not used
  
## Sample Code:
  - Codes in https://github.com/Ascend-Huawei/OfflineModelSamples/tree/main/face_detection/src create a sample to quickly understand how the model works, preprocessing, inference, postprocessing are already included.
  - Preprocessing: 
    - **Image Resize**: 300*300
    - **Image Type**: FLOAT32
    - Change order from **[300, 300, 3]**(HWC) to **[3, 300, 300]**(CHW)
  - Postprocessing:
    - select bboxes with score higher than threshold score, filter bboxes with score higher than 1(which should be wrong bboxes detected)
    - plot bbox, category name and confidence score on image 

  - Result would be printed in the terminal:
    ``` 
    resultList number: 1
    shape of output 0 is (1, 8) 
    ``` 
    
  - To run code， simply using commands below in the terminal:
  
    ``` 
    cd face_detection/src
    python3 face_detection.py 
    ``` 

