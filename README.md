# OpenVINO with FPGA Hello World Face Detection Exercise

**Note:** This tutorial has been written using OpenVINO™ Toolkit for Linux with FPGA Support version 2018 R3 and is for use with this version only. Using this tutorial with any other version may not be correct.

# Table of Contents

<div class="table-of-contents"><ul>
<li><a href="#openvino-with-fpga-hello-world-face-detection-exercise">OpenVINO with FPGA Hello World Face Detection Exercise</a></li>
<li><a href="#table-of-contents">Table of Contents</a></li>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#getting-started">Getting Started</a><ul>
<li><a href="#prerequisites">Prerequisites</a></li></ul></li>
<li><a href="#download-the-tutorial-from-the-git-repository">Download the Tutorial from the Git Repository</a><ul>
<li><a href="#use-git-clone-to-clone-the-entire-repository">Use Git Clone to Clone the Entire Repository</a></li></ul></li><ul></li></ul>
<li><a href="#tutorial-files">Tutorial Files</a></li>
<li><a href="#openvino-toolkit-overview-and-terminology">OpenVINO™ Toolkit Overview and Terminology</a><ul>
<li><a href="#the-inference-engine">The Inference Engine</a></li></ul><ul>
<li><a href="#face-detection-sample">Face Detection Sample</a></li></ul>
<li><a href="#build">Build</a>
<li><a href="#run-the-application">Run the Application</a></li></ul></div>

## Introduction

The OpenVINO™ toolkit runs inference models using the CPU to process images. As an option, OpenVINO can use GPU and VPU devices, if available. You can use the inference models to process video from an optional USB camera, an existing video file, or still image files. 

This tutorial examines a sample application that was created with the OpenVINO™ toolkit. The tutorial steps will guide you through downloading the latest Face Detection Tutorial from GitHub, walk you through the sample code, and then compile and run the code on the the available hardware. During the process, you will become familiar with key OpenVINO™ concepts.

This tutorial starts with a base application that reads image data and outputs the image to a window. The steps build on each other by adding deep learning models that process image data and make inferences. In the final step, you will use the completed application to detect faces, report the age and gender of the detected face, and draw a 3D axis representing the head pose for each face.

## Getting Started

### Prerequisites
To run the application in this tutorial, the OpenVINO™ toolkit for Linux with FPGA and its dependencies must already be installed and verified using the included demos. Installation instructions may be found at: https://software.intel.com/en-us/articles/openvino-install-test-linux-fpga
If to be used, any optional hardware must also be installed and verified including:
* Intel® Arria® 10 GX FPGA Development Kit or the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA 
* GPU - normally embedded with supported Intel® CPUs and requires drivers and updated Linux kernel to run
* VPU - USB Intel® Movidius™ Neural Compute Stick and what is being referred to as "Myriad"
**GitHub**
Download/clone will include the dx_face_detection zip file

A summary of what is needed:
### Hardware
Target and development platforms meeting the requirements described in the "System Requirements" section of the OpenVINO™ toolkit documentation which may be found at: https://software.intel.com/openvino-toolkit
**Note:** While writing this tutorial, an Intel® i7-7700 (CPU with GPU & Intel® Arria® 10 GX FPGA Development Kit) was used as both the development and target platform.
Optional:
* Intel® Movidius™ Neural Compute Stick
* GPU support
### Software
* OpenVINO™ toolkit supported Linux operating system. This tutorial was run on 64-bit Ubuntu 16.04.3 LTS updated to kernel 4.14.20 following the OpenVINO™ toolkit installation instructions.
* The latest OpenVINO™ toolkit with FPGA support installed and verified. This tutorial was written using version 2018 R3.
* At least one utility for downloading from the GitHub repository: Subversion (svn), Git (git), or both
By now you should have completed the Linux installation guide for the OpenVINO™ toolkit with FPGA support (link), however before continuing, please ensure:
That after installing the OpenVINO™ toolkit with FPGA support you have run the supplied demo samples
* If you have and intend to use an FPGA: You have installed and tested the Quartus® Programmer (link?) and able to program bitstreams 
* If you have and intend to use a GPU: You have installed and tested the GPU drivers
* If you have and intend to use a Myriad: You have connected and tested the USB Intel® Movidius™ Neural Compute Stick
* That your development platform is connected to a network and has Internet access. To download all the files for this tutorial, you will need to access GitHub on the Internet.

### Required Connectivity
* Your development platform is connected to a network and has Internet access 
* You have Github access

## Download the Tutorial from the Git Repository
In these steps you create a directory to put the files you download from the “OpenVINO FPGA Hello World Face Detection” GitHub repository. 
### Use Git Clone to Clone the Entire Repository
1.	Bring up a command shell prompt by opening a terminal (such as xterm) or selecting a terminal that is already open.
2. if you have not installed Git, then install it.
```
sudo apt-get update
sudo apt-get install git
```
3.	Create a directory named "tutorials"
```
mkdir tutorials
```
4. Go to the tutorials directory
```
cd tutorials
```
5.	Clone the repository
```
git clone https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
```

6.	Change to the face detection tutorial folder:
```
cd openvino-with-fpga-hello-world-face-detection
```

You have the Face Detection Tutorial files. The next section shows the directory structure of the files you extracted.

# Tutorial Files
The "tutorial" directory contains:
* Images\ - directory of images
* Videos\ - directory of videos
* cmake\ - directory of common CMake files
* dx_face_detection\  - directory of code to help run the scripts
* Readme.md - This document

# OpenVINO™ Toolkit Overview and Terminology
The OpenVINO™ toolkit enables the quick deployment of Convolutional Neural Networks (CNN) for heterogeneous execution on Intel® hardware while maximizing performance. Deployment is accomplished through the the Intel® Deep Learning Deployment Toolkit (Intel® DL Deployment Toolkit), included within the OpenVINO™ toolkit.
![OV Overview](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/OV%20Overview.png)

The CNN workflow is:
1.	Create and train the CNN inference model in a framework, such as Caffe*.
2.	Use the Model Optimizer on the trained model to produce an optimized Intermediate Representation (IR), stored .bin and .xml files, for use with the Inference Engine.
3.	Use the Inference Engine with your application to load and run the model on your devices.

This tutorial focuses on the last step in the workflow: using the user application the Inference Engine to run models on a CPU, GPU, FPGA, and Movidius™ Neural Compute Stick.
## The Inference Engine
Below is a more detailed view of the User Application and Inference Engine:
![IE Graphic](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/IE%20graphic.png)

The Inference Engine includes a plugin library for each supported device that has been optimized for the Intel® hardware device CPU, GPU, FPGA and Myriad. From here, we will use the terms "device" and “plugin” with the assumption that one infers the other (e.g. CPU device infers the CPU plugin and vice versa). As part of loading the model, the User Application tells the Inference Engine which device to target which in turn loads the associated plugin library to later run on the associated device. The Inference Engine uses “blobs” for all data exchanges, basically arrays in memory arranged according the input and output data of the model.
![face_detection](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/face_detection.png)

## Face Detection Sample
In this Face Detection sample, the model estimates the head pose based on the face image it is given. The face detection model estimates the age, gender and head pose estimation. After the head pose model has processed the face, the application will draw a set of axes over the face, indicating the Yaw, Pitch, and Roll orientation of the head. A sample output showing the results where the three axes appears below. The metrics reported also include the time to run the head pose model.

![face_detection_overlay](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/face_detection_overlay.png)

In the image above, the three axes intersect in the center of the head. The blue line represents Roll, and it extends from the center of the head to the front and the back of the head. The red line represents Pitch, and is drawn from the center of the head to the left ear. The green line represents Yaw, and is drawn from the center of the head to the top of the head.

For details about the models see the Full Tutorial: https://github.com/intel-iot-devkit/inference-tutorials-generic/blob/openvino_toolkit_r3_0/face_detection_tutorial/step_4/Readme.md#introduction

In the next section, you build and run the application, and see how it runs the three analysis models.

# Build
1.	Use a terminal window to access a command shell prompt.

2.	Go the directory containing the Hello World files
```
cd dx_face_detection
```
3.	Source the variables
```
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
4.	Create a directory to build the tutorial
```
mkdir build
```
5.	Go to the build directory
```
cd build
```
6.	Run CMake to set the build target and file locations

```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
7.	Build the executable
```
make
```
**Alternative make command:** Run make across multiple pieces of hardware to speed the process
```
make -j $(nproc)
```
8.	Load a bitstream that works well for object detection. The OpenVINO toolkit with support for FPGA includes bitstreams for Both Arria 10 FPGA cards. 

* For the Arria 10 GX Development Kit, use this command:
```
aocl program acl0 /opt/intel/computer_vision_sdk_fpga_2018.3.343/a10_devkit_bitstreams/2-0-1_A10DK_FP11_ResNet50-101.aocx
```
* For the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA (IEI Mustang-F100-A10), use this command:
```
aocl program acl0 /opt/intel/computer_vision_sdk_2018.4.420/bitstreams/a10_vision_design_bitstreams/4-0_PL1_FP11_MobileNet_ResNet_VGG_Clamp.aocx
```


# Run the Application
1.	Go to the main level directory
```
cd ..
```
2.	Turn the script into an executable file
```
chmod +x run_fd.sh
```
## Script Description
This tutorial includes a script to select the media file, models and hardware. The script commands are provided under each step to provide the opportunity to explore other possibilities.
`run_fd.sh` requires at least one hardware target, and supports up to 3

### How To Use:
```
./run_fd.sh (face detection hardware) (age/gender hardware) (head pose hardware)
```
EXAMPLE: `./run_fd.sh fpga fpga gpu`

**Choose a hardware component for each argument you use**
1. cpu 
2. gpu 
3. fpga 

**Targets (in order on command line):**
1. 1st argument is required, for face detection
2. 2nd argument, optional, for age & gender recognition
3. 3rd argument, optional, requires face detection + age/gender recognition, for head pose

You will see rectangles and the head pose axes that follow the faces around the image (if the faces move), accompanied by age and gender results for the faces, and the timing statistics for processing each frame of the video.

When the video finishes, press any key to close the window and see statistics of the trial.


## Example 1 - Run face detection on targeted hardware (CPU):
```
./run_fd.sh cpu
```

**Example 1 Full command**
```
build/intel64/Release/face_detection_tutorial -i ../Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -d CPU
```


## Example 2 - Run face detection on targeted hardware (FPGA):
 ```
 ./run_fd.sh fpga
 ```

**Example 2 Full command**
```
build/intel64/Release/face_detection_tutorial -i ../Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -d HETERO:FPGA,CPU
```

## Example 3 - Run face detection on FPGA with age/gender recognition on GPU
```
./run_fd.sh fpga gpu
```

**Example 3 Full command**
```
build/intel64/Release/face_detection_tutorial -i ../Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -m_ag /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml -d HETERO:FPGA,CPU -d_ag GPU
```

## Example 4 - Run face detection on FPGA, age/gender recognition on a GPU, and head pose estimation on a CPU
```
./run_fd.sh fpga gpu cpu
```

**Example 4 Full command**
```
build/intel64/Release/face_detection_tutorial -i ../Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -m_ag /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml -m_hp /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -d HETERO:FPGA,CPU -d_ag GPU -d_hp CPU
```


## Example 5 - Run everything on cpu
```
./run_fd.sh cpu cpu cpu
```

**Example 5 Full command**
```
build/intel64/Release/face_detection_tutorial -i ../Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -m_ag /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml -m_hp /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -d CPU -d_ag CPU -d_hp CPU
```

**NOTE:** The FPGA plugin does NOT support the head pose model.  If specified, it will be replaced with CPU.
