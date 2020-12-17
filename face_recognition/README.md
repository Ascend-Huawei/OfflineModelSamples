# Face Recognition
This model recognizes the face of a person in given image.

## Model Description

#### Model Name: Sphereface
Here we are using a Caffe model converted to offline model (.om) to inference on board. Download the weights and network files to the project folder 'face_recognition/src':

- Caffe Model:

  - Weights: https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/sphereface/sphereface.caffemodel

  - Network: https://github.com/Ascend-Huawei/models/blob/master/computer_vision/classification/sphereface/sphereface.prototxt

Execute the following command in project folder 'face_recognition/src' to obtain model converted to offline (.om) format:

**atc --output_type=FP32 --input_shape="data:1,3,112,96" --weight="sphereface.caffemodel" --input_format=NCHW --output="sphereface" --soc_version=Ascend310 --framework=0 --save_original_model=false --model="sphereface.prototxt"**


#### Input
- **Input Shape**: [1,3, 112, 96]
- **Input Format** : NCHW
- **Input Type**: RGB FLOAT32

#### Output
- The pre-trained model is trained on CAISA-WebFace and testing on LFW using the 20-layer CNN architecture. It will recognize face and return 512 vector.
  
  - **List shape**: [1, 512, 1, 1]

The script outputs the cosine similarity value between two given input face images. A value of 1.0 indicates perfect similarity. 

[Example] output printed to terminal:

```
Cosine similarity between faces:  0.94596004
```
  
## Sample Code:
  - Codes in https://github.com/Ascend-Huawei/OfflineModelSamples/tree/main/face_recognition/src create a sample to quickly understand how the model works, preprocessing, inference, postprocessing are already included.
  - Preprocessing: 
    - **Format conversion**: BGR to RGB
    - **Image Type**: FLOAT32
    - Change channels order from *NHWC* to *NCHW*
    
  - Postprocessing:
    - Extract the result list of arrays for input images, each as a flat list of float numbers.
    - Calculate the cosine similarity value between the two result vectors, to compare two faces.

    
  - To run code，simply use commands below in the terminal:
  
    ``` 
    cd face_recognition/src
    python3 sphereface.py 
    ``` 

