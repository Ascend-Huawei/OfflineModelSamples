# Age Classification
This model predicts the age of person in the input image.

## Model Description: Inception_Age

#### Model Conversion to ATC Offline Model
* Step 1: Download the pre-trained model file to your project folder 'age_classification/src':
  - PB Model:
https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/inception_age/inception_age.pb

  - Model Details:
https://github.com/Ascend-Huawei/models/blob/master/computer_vision/classification/inception_age/README_en.md

* Step 2: Navigate to project folder 'age_classification/src' in the terminal and execute the following command to convert the above model to offline model (.om) format:

  **atc --output_type=FP32 --input_shape="Placeholder:1,227,227,3" --input_format=NHWC --output="inception_age" --soc_version=Ascend310 --framework=3 --save_original_model=false --model="inception_age.pb"**


#### Model Input
- **Input Shape**: [1,227,227, 3]
- **Input Image Format**: BGR
- **Input Format** : NHWC
- **Input Type**: FLOAT32

#### Model Output
- 1 lists. **8** represents 8 age ranges which are {(0,2), (4,6), (8,12), (15,20), (25,32), (38,43), (48,53), (60,100)}
  - **List shape**: [1,8]
  
## Sample Code:
  - Codes in https://github.com/Ascend-Huawei/OfflineModelSamples/tree/main/age_classification/src create a sample to quickly get how the model works, preprocessing, inference, postprocessing are already included.
  - Preprocessing: 
    - **Image Resize**: 227*227
    - **Image Type**: FLOAT32
  - Postprocessing:
    - create a dictionary mapping 8 indexes and corresponding age range
    - using **argmax** to select the highest score in 8 possibilities
   
  - **Notes:** results would be affected when people are too close to camera, or the background is dark.
  
  - Result would be printed in the terminal:
    ``` 
    age of this person is in the range of (2,4)
    ``` 
    
  - To run code， simply using commands below in the terminal:
  
    ``` 
    cd age_classification/src
    python3 main.py 
    ``` 
  
