# Mask Detection

## Model Description

#### Model Name: mask_detection

#### Model Conversion to ATC Offline Model
* Step 1: Download the pb model file to your project folder 'mask_dteection/src':

  https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/mask_detection/mask_detection.pb

* Step 2: Navigate to the same project folder and execute the following command in terminal, to convert the model to offline model (.om) format:

  **atc --output_type=FP32 --input_shape="images:1,352,640,3" --input_format=NHWC --output="mask_detection" --soc_version=Ascend310 --framework=3 --save_original_model=false --model="mask_detection.pb"**


#### Model Input
- **Input Shape**: [1,352,640,3]
- **Input Format** : NHWC
- **Input Type**: BGR FLOAT32

#### Model Output
- 3 feature maps are obtained as the output of the model:
   - **1st feature map shape**: [1,11,20,24]
   - **2nd feature map shape**: [1,22,40,24]
   - **3rd feature map shape**: [1,44,80,24]
  
## Sample Code:
  - Codes in https://github.com/Ascend-Huawei/OfflineModelSamples/main/mask_detection/src create a sample to quickly get how the model works, preprocessing, inference, postprocessing are already included.
  - Preprocessing: 
    - **Image Resize**: (352,640)
    - **Image Type**: BGR FLOAT32
   
  - Postprocessing:
    - Yolo postprocessing(including bboxes decode, nms)
    - Because for this model, bboxes with person label always have higher score while mask label is lower. We have 2 seperate confidence scores here. Select bboxes with score higher than person confidence score(0.2 here) with "person" label, and mask confidence score(0.08 here).   
    - iou score(0.3 here)
    - plot bbox, category name and confidence score on image 
    
    *NOTE:* **mask confidence score**, **person confidence score** and **iou score** are three adjustable parameters here, you could simple change them according to your projects objective.
    
  - Result would be saved in "output/" folder, show as below:
  
    ![image](https://github.com/Ascend-Huawei/OfflineModelSamples/blob/main/mask_detection/src/out/mask_output.jpg)
    Photo by: Vergani_Fotografia / iStock
    
  - To run code， simply using commands below in the terminal:
  
    ``` 
    cd mask_detection/src
    python3 main.py 
    ``` 
  
