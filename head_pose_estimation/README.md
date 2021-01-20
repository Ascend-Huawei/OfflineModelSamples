# Head Pose Estimation
This model recognizes the head pose of a person in terms of 3 angles: *yaw*, *pitch* and *roll* in an image. The face detection model is used to locate the face region prior to inferring the head pose angles.

## Model Description

#### Models:
Here we are using offline models for 1. face detection 2. head pose estimation to inference on the board. 

**Face Detection**

- Weights: https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/face_detection/face_detection.caffemodel
- Network: https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/face_detection/face_detection.prototxt

Execute the following command from the directory of the downloaded files, to convert the pre-trained model to offline model (.om) format:

*atc --output_type=FP32 --input_shape="data:1,3,300,300" --weight="face_detection.caffemodel" --input_format=NCHW --output="face_detection" --soc_version=Ascend310 
--framework=0 --save_original_model=false 
--model="face_detection.prototxt"*

**Head Pose Estimation**

Download the weights and network files to your project directory 'head_pose_estimation/src':

- Weights:
https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/head_pose_estimation/head_pose_estimation.caffemodel
- Network: https://github.com/Ascend-Huawei/models/blob/master/computer_vision/object_detect/head_pose_estimation/head_pose_estimation.prototxt

**Note:** To download the Network file with **wget**, please use following command:

*wget https://raw.githubusercontent.com/Ascend-Huawei/models/master/computer_vision/object_detect/head_pose_estimation/head_pose_estimation.prototxt*

  Execute the following command from the project directory 'head_pose_estimation/src' to convert the pre-trained model for head pose estimation to offline model (.om) format:

*atc --framework=0 --model="head_pose_estimation.prototxt"  --weight="head_pose_estimation.caffemodel" --input_shape="data:1,3,224,224" --input_format=NCHW --output="head_pose_estimation"  --output_type=FP32 --soc_version=Ascend310*

**NOTE:** In case the above conversion step fails, please download the head pose estimation offline model directly from the link below:
https://drive.google.com/file/d/1vKyuRg_NIDBx2C-KxM9Mf_dGUhRj7qf9/view?usp=sharing


#### Inputs
The input for face detection model are as follows:
- **Input Shape**: [1,300,300, 3]
- **Input Format** : NCHW
- **Input Type**: BGR FLOAT32

The input for the head pose estimation model are as follows:
- **Input Shape**: [1,3, 224, 224]
- **Input Format** : NCHW
- **Input Type**: BGR FLOAT32

#### Outputs

Outputs for the face detection model:

- 2 lists. Only the 2nd list is used.
  - **1st list shape**: [1,8]
  - **2nd list shape**: [1, 100, 8]
- For the second list: **100** represents 100 bounding boxes. **0-8** describe information of each box as below:
  - **0 position**: not used
  - **1 position**: label
  - **2 position**: confidence score
  - **3 position**: top left x coordinate
  - **4 position**: top left y coordinate
  - **5 position**: bottom right x coordinate
  - **6 position**: bottom right y coordinate
  - **7 position**: not used
  
The outputs for the head pose estimation model are as follows:
- List of numpy arrays: 
  - **Array shapes**: (1, 136, 1, 1), (1, 3, 1, 1)
The first list is a set of 136 facial keypoints. The second list in the output containing the 3 values of yaw, pitch, roll angles predicted by the model, which are used to determine head pose based on some preset rules.

Output printed to terminal (sample):
```
Head angles: [array([[9.411621]], dtype=float32), array([[7.91626]], dtype=float32), array([[-1.0116577]], dtype=float32)]
Pose: Head Good posture
```
Result image with 64 keypoints plotted on detected face saved in 'out' folder.

  
## Code:

  - All code files needed to run the experiment are included in folder 'head_pose_estimation/src'. The script 'head_pose_estimation.py' contains all the preprocessing, model inference and post_processing methods. 
  - Preprocessing: 
    - **Resize**: (224, 224)
    - **Image Type**: FLOAT32
    - **Input Format** : NCHW
    - Change order from **[300, 300, 3]**(HWC) to **[3, 300, 300]**(CHW)
    
  - The om model file (.om) must be downloaded to the project folder 'head_pose_estimation/src'
 
  - Postprocessing:
    - Infer head pose from yaw, pitch and roll angles, using fixed range thresholds.

    
  - To run code， simply using commands below in the terminal:
  
    ``` 
    cd head_pose_estimation/src
    python3 head_pose_estimation.py 
    ``` 












  













