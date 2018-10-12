# OpenVINO with FPGA Hello World Face Detection Exercise

**Note:** This tutorial is for the Open Visual Inference & Neural Network Optimization (OpenVINO™) toolkit for Linux with FPGA Support version 2018 R3.1. Do not use this tutorial for other versions of the OpenVINO™ toolkit.

# Table of Contents

<div class="table-of-contents"><ul>
<li><a href="#openvino-with-fpga-hello-world-face-detection-exercise">OpenVINO with FPGA Hello World Face Detection Exercise</a></li>
<li><a href="#table-of-contents">Table of Contents</a></li>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#getting-started">Gather Everything You Need</a><ul>
<li><a href="#prerequisites">Prerequisites</a></li></ul></li>
<li><a href="#downloading-the-tutorial-from-the-git-repository">Downloading the Tutorial from the Git Repository</a><ul>
<li><a href="#option-1-using-git-clone-to-clone-the-entire-repository">Option #1: Using Git Clone to Clone the Entire Repository</a></li>
<li><a href="#option-2-using-svn-export-to-download-only-this-tutorial">Option #2: Using SVN Export to Download Only This Tutorial</a></li></ul></li><ul></li></ul>
<li><a href="#extract-the-dx_face_detection-file">Extract the dx Face Detection Files<a></li>
<li><a href="#tutorial-files">Tutorial Files</a></li>
<li><a href="#openvino-toolkit-overview-and-terminology">OpenVINO™ Toolkit Overview and Terminology</a><ul>
<li><a href="#using-the-inference-engine">The Inference Engine</a></li></ul><ul>
<li><a href="#face-detection-sample">The Face Detection Sample</a></li></ul>
<li><a href="#build">Build the Application</a>
<li><a href="#running-the-app">Run the Application</a></li></ul></div>

## Introduction

The OpenVINO™ toolkit runs inference models using the CPU to process images. As an option, OpenVINO can use GPU and VPU devices, if available. You can use the inference models to process video from an optional USB camera, an existing video file, or still image files. 

This tutorial examines a sample application that was created with the OpenVINO™ toolkit. The tutorial steps will guide you through downloading the latest Face Detection Tutorial from GitHub, walk you through the sample code, and then compile and run the code on the the available hardware. During the process, you will become familiar with key OpenVINO™ concepts.

This tutorial starts with a base application that reads image data and outputs the image to a window. The steps build on each other by adding deep learning models that process image data and make inferences. In the final step, you will use the completed application to detect faces, report the age and gender of the detected face, and draw a 3D axis representing the head pose for each face.

## Gather Everything You Need

### Required and Optional Hardware
* The target and development platforms must meet the requirements described in the "System Requirements" section of the OpenVINO™ toolkit documentation at https://software.intel.com/openvino-toolkit. For both the development and target platform, this tutorial was tested with an Intel® i7-7700 CPU with GPU, and an Intel® Arria® 10 GX FPGA Development Kit
* Intel® Arria® 10 GX FPGA Development Kit or the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA
* Optional hardware must be installed/connected. Optional hardware includes:
	* USB camera: A standard USB Video Class (UVC) camera. This tutorial used the Logitech, Inc. HD Pro Webcam C920.
	* Intel® Arria® 10 GX FPGA Development Kit or the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA 
	* GPU: A GPU is usually embedded in supported Intel® CPUs. The GPU requires drivers and an updated Linux kernel
	* VPU: A USB Movidius™ Neural Compute Stick

### Required and Optional Software
* The latest OpenVINO™ toolkit with FPGA support installed and verified by the demos. For instructions, see https://software.intel.com/en-us/articles/openvino-install-linux-fpga#. This tutorial was tested against OpenVINO™ version 2018 R3.0. 
* A Linux operating system supported by the OpenVINO™ toolkit. This tutorial was tested on 64-bit Ubuntu 16.04.3 LTS that was updated to kernel 4.14.20 according to the OpenVINO™ toolkit installation instructions.
* At least one utility for downloading from the GitHub repository. Examples include Subversion (svn), Git (git), or both
* You installed and tested the Quartus® Programmer (link?) and can program bitstreams 
* Software associated with optional hardware must be installed and tested.
	* If you intend to use a USB camera: You tested your USB camera
	* If you intend to use a GPU: You installed and tested the GPU drivers. The GPU requires an updated Linux kernel
	* If you intend to use a VPU: Movidius™ Neural Compute Stick: You tested the USB Movidius™ Neural Compute Stick

### Required Connectivity
* Your development platform is connected to a network and has Internet access 
* You have GitBub access

## Download the Tutorial from the Git Repository
In these steps you create a directory to put the files you download from the “OpenVINO FPGA Hello World Face Detection” GitHub repository, choose an option to download the tutorial, and then complete the download. 

To begin, choose between two download options
* Option 1: Use "git clone" to download the tutorial files as part of the entire repository
* Option 2: Use “svn export” to download only the tutorial

Use the steps below that correspond to your choice.

### Option #1: Use Git Clone to Clone the Entire Repository
Use these steps if you want to download the full “OpenVINO FPGA Hello World Face Detection” GitHub repository. If you want to download only the tutorial, not the full repository, skip this section and go to Option #2: Use SVN Export to Download Only This Tutorial

1.	Use a terminal window to access a command shell prompt. 
2.	Create a directory named "tutorials"
```
mkdir tutorials
```
3.	Go to the tutorials directory
```
cd tutorials
```
4.	Clone the repository
```
git clone https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
```
A face detection tutorial directory is created, named "openvino-with-fpga-hello-world-face-detection". The tutorial files are under this directory.

You have completed the tutorial download. Do not follow the steps under Option #2. Go to the Tutorial Files section to see the file directory structure.

### Option #2: Use SVN Export to Download Only This Tutorial

Do not use these steps if you completed Option #1. 

1.	Use a terminal window to access a command shell prompt. 
2.	Create a top-level tutorial directory
```
mkdir -p tutorials/openvino-with-fpga-hello-world-face-detection
```
3.	Go to the directory
```
cd tutorials/openvino-with-fpga-hello-world-face-detection
```
4.	Download the tutorial subdirectory
```
svn export https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
```

## Extract the dx Face Detection Files

1.	Go to the face detection tutorial directory
```
cd openvino-with-fpga-hello-world-face-detection
```
2.	Extract the tutorial files
```
tar xvzf dx_face_detection.tgz
```
You have the Face Detection Tutorial files. The next section shows the directory structure of the files you extracted.

# Tutorial Files
The "tutorial" directory contains:
* Images\ - directory of images
* Videos\ - directory of videos
* cmake\ - directory of common CMake files
* dx_face_detection\  - directory of code to help run the scripts
* Readme.md - This document

* Others? **additional explanations?**

# About the OpenVINO™ Toolkit
The OpenVINO™ toolkit enables the quick deployment of convolutional neural networks (CNN) for heterogeneous execution on Intel® hardware while maximizing performance. Deployment is accomplished through the the Intel® Deep Learning Deployment Toolkit (Intel® DL Deployment Toolkit), included within the OpenVINO™ toolkit.
![OV Overview](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/OV%20Overview.png)

The CNN workflow is:
1.	Create and train the CNN inference model in a framework, such as Caffe*.
2.	Use the Model Optimizer on the trained model to produce an optimized Intermediate Representation (IR), stored .bin and .xml files, for use with the Inference Engine.
3.	Use the Inference Engine with your application to load and run the model on your devices.

This tutorial focuses on the last step in the workflow: using the user application the Inference Engine to run models on a CPU, GPU, FPGA, and Movidius™ Neural Compute Stick.

## The Inference Engine
The relationship between the user application and the Inference engine is:
![IE Graphic](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/IE%20graphic.png)

The Inference Engine includes a plugin library for each supported device that is optimized for the Intel® hardware device CPU, GPU, FPGA, and Movidius™ Neural Compute Stick. The terms "device" and “plugin” assume that one infers the other. For example, a CPU device infers the CPU plugin and vice versa. 

When loading a model, the User Application tells the Inference Engine which device to target. The target then loads the associated plugin library, which later runs on the associated device. The Inference Engine uses “blobs” for all data exchanges, basically arrays in memory arranged according the input and output data of the model.
![face_detection](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/face_detection.png)

# The Face Detection Sample
In this sample, the model uses a face image to estimate an individual's head pose. The face detection model estimates the age, gender, and head pose estimation. After the head pose model has processed the face, the application draws a set of axes over the face, indicating the yaw, pitch, and roll orientation of the head. A sample output, below, shows the results in which the three axes appears. The metrics include the time it took to run the head pose model.

![face_detection_overlay](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/face_detection_overlay.png)

In the image above, the three axes intersect at the center of the head. The blue line, roll, extends from the center to the front and the back of the head. The red line, pitch, is drawn from the center to the left ear. The green line, yaw, is drawn from the to the top of the head.

For details about the models, see the full tutorial: https://github.com/intel-iot-devkit/inference-tutorials-generic/blob/openvino_toolkit_r3_0/face_detection_tutorial/step_4/Readme.md#introduction

In the next section, you build and run the application, and see how it runs the three analysis models.

# Build the Application
1.	Use a terminal window to access a command shell prompt.

2.	Go the directory containing the Hello World files
```
cd dx_face_detection
```
3.	Source the variables
```
source /home/<user>/setup_env.sh
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
8.	Load a bitstream that works well for object detection. The OpenVINO toolkit with support for FPGA includes bitstreams.
9.	Program the bitstream
```
aocl program acl0 /opt/intel/computer_vision_sdk_fpga_2018.3.343/a10_devkit_bitstreams/2-0-1_A10DK_FP11_ResNet50-101.aocx
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

## Use the Application Script 
This tutorial includes a script to select the media file, models and hardware. The script commands are provided under each step to provide the opportunity to explore other possibilities.

`run_fd.sh` requires at least one hardware target, and supports up to three

### How to Use the Script
```
./run_fd.sh (face detection hardware) (age/gender hardware) (head pose hardware)
```

**Argument requirements**
* face detection hardware argument: Required for face detection
* age/gender hardware argument: Optional. Use this for age & gender recognition
* head pose hardware argument: Optional. This option requires the face detection + age/gender recognition arguments. Use the head pose hardware argument when you want to see output that includes the head pose

**Choose a hardware component for each argument you use**
* cpu 
* gpu 
* fpga 

The output shows rectangles with head pose axes that follow the face paths around the image if the faces move in a video. Textual output includes age and gender results, and timing statistics for processing each video frame.

## Application Examples
### Example 1 - Run face detection on a CPU
```
./run_fd.sh cpu
```

**Example 1 full command**
```
build/intel64/Release/face_detection_tutorial -i /home/<user>/Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -d CPU
```

### Example 2 - Run face detection on FPGA
```
./run_fd.sh fpga
```

**Example 2 full command**
```
build/intel64/Release/face_detection_tutorial -i /home/<user>/Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -d HETERO:FPGA,CPU
```

### Example 3 - Run face detection on FPGA with age/gender recognition on a GPU
```
./run_fd.sh fpga gpu
```

**Example 3 full command**
```build/intel64/Release/face_detection_tutorial -i /home/<user>/Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -m_ag /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml -d HETERO:FPGA,CPU -d_ag GPU
```

### Example 4 - Run face detection on FPGA, age/gender recognition on a GPU, and head pose estimation on a CPU
```
./run_fd.sh fpga gpu cpu
```

**Example 4 full command**
```
build/intel64/Release/face_detection_tutorial -i /home/<user>/Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -m_ag /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml -m_hp /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -d HETERO:FPGA,CPU -d_ag GPU -d_hp CPU
```

### Example 5 - Run face detection, age/gender recognition, and head pose estimation, all on a CPU
```
./run_fd.sh cpu cpu cpu
```

**Example 5 full command**
```
build/intel64/Release/face_detection_tutorial -i /home/<user>/Videos/head-pose-face-detection-female-and-male.mp4 -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -m_ag /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml -m_hp /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -d CPU -d_ag CPU -d_hp CPU
```

**NOTE:** The FPGA plugin does NOT support the head pose model.  If you try to use this as an option, it will be replaced with cpu.
