# OpenVINO with FPGA Hello World Face Detection Exercise

**Note:** This tutorial has been written using OpenVINO™ Toolkit for Linux with FPGA Support version 2018 R3.0 and is for use with this version only. Using this tutorial with any other version may not be correct.

# Table of Contents

<div class="table-of-contents"><ul><li><a href="#openvino-with-fpga-hello-world-face-detection-exercise">OpenVINO with FPGA Hello World Face Detection Exercise</a></li><li><a href="#table-of-contents">Table of Contents</a></li><li><a href="#introduction">Introduction</a></li><li><a href="#getting-started">Getting Started</a><ul><li><a href="#prerequisites">Prerequisites</a></li></ul></li><li><a href="#downloading-the-tutorial-from-the-git-repository">Downloading the Tutorial from the Git Repository</a><ul><li><a href="#option-1-using-git-clone-to-clone-the-entire-repository">Option #1: Using Git Clone to Clone the Entire Repository</a></li><li><a href="#option-2-using-svn-export-to-download-only-this-tutorial">Option #2: Using SVN Export to Download Only This Tutorial</a></li></ul></li><ul></li></ul><li><a href="#tutorial-files">Tutorial Files</a></li></ul><li><a href="#openvino-toolkit-overview-and-terminology">OpenVINO™ Toolkit Overview and Terminology</a><ul><li><a href="#using-the-inference-engine">Using the Inference Engine</a><ul><li><a href="#face-detection-sample">Face Detection Sample</a></li></ul></li></ul><li><a href="build">Build</a></li></div>

## Introduction

The purpose of this tutorial is to examine a sample application that was created using the Open Visual Inference & Neural Network Optimization (OpenVINO™) toolkit. The application is able to run inference models on the CPU, and optionally (must be available), GPU and VPU devices to process images. The models can be used to process video from an optional USB camera, an existing video file, or still image files. To do that, we will download the latest Face Detection Tutorial from GitHub and then walk through the sample code for each step before compiling and running on the the available hardware.

This tutorial will start from a base application that can read in image data and output the image to a window. From there, each step adds deep learning models that will process the image data and make inferences. In the final step, the complete application will be able to detect a face, report age and gender for the face, and draw a 3D axis representing the head pose for each face. Before that, some key concepts related to using the OpenVINO™ toolkit will be introduced and later seen along the way within the steps.

## Getting Started

### Prerequisites
To run the application in this tutorial, the OpenVINO™ toolkit for Linux with FPGA and its dependencies must already be installed and verified using the included demos. Installation instructions may be found at: https://software.intel.com/en-us/articles/openvino-install-linux-fpga#
If to be used, any optional hardware must also be installed and verified including:
*USB camera - Standard USB Video Class (UVC) camera. The Logitech, Inc. HD Pro Webcam C920 was used when writing this tutorial.
* Intel® Arria® 10 GX FPGA Development Kit or the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA 
* GPU - normally embedded with supported Intel® CPUs and requires drivers and updated Linux kernel to run
* VPU - USB Intel® Movidius™ Neural Compute Stick and what is being referred to as "Myriad"
**GitHub**
Download/clone will include the dx_face_detection zip file

A summary of what is needed:
### Hardware
Target and development platforms meeting the requirements described in the "System Requirements" section of the OpenVINO™ toolkit documentation which may be found at: https://software.intel.com/openvino-toolkit
**Note:** While writing this tutorial, an Intel® i7-7700 (CPU with GPU) was used as both the development and target platform.
Optional:
* Intel® Movidius™ Neural Compute Stick
* USB UVC camera
* GPU support
### Software
* OpenVINO™ toolkit supported Linux operating system. This tutorial was run on 64-bit Ubuntu 16.04.3 LTS updated to kernel 4.14.20 following the OpenVINO™ toolkit installation instructions.
* The latest OpenVINO™ toolkit with FPGA support installed and verified. This tutorial was written using version 2018 R3.0.
* At least one utility for downloading from the GitHub repository: Subversion (svn), Git (git), or both
By now you should have completed the Linux installation guide for the OpenVINO™ toolkit with FPGA support (link), however before continuing, please ensure:
That after installing the OpenVINO™ toolkit with FPGA support you have run the supplied demo samples
* If you have and intend to use an FPGA: You have installed and tested the Quartus® Programmer (link?) and able to program bitstreams 
* If you have and intend to use a GPU: You have installed and tested the GPU drivers
* If you have and intend to use a USB camera: You have connected and tested the USB camera
* If you have and intend to use a Myriad: You have connected and tested the USB Intel® Movidius™ Neural Compute Stick
* That your development platform is connected to a network and has Internet access. To download all the files for this tutorial, you will need to access GitHub on the Internet.

## Downloading the Tutorial from the Git Repository
The first thing we need to do is create a place for the Face Detection tutorial and then download it. To do this, we will create a directory called "tutorials" and use it to store the files that are downloaded from the “OpenVINO FPGA  Hello World Face Detection” GitHub repository. There are two options to download this tutorial: 1) Download as part of the entire repository using “git clone”, or 2) Use “svn export” to download just this tutorial (smaller)
### Option #1: Using Git Clone to Clone the Entire Repository
1.	Bring up a command shell prompt by opening a terminal (such as xterm) or selecting a terminal that is already open.
2.	Create a "tutorials" directory where we can download the Face Detection tutorial and then change to it:
```
mkdir tutorials
cd tutorials
```
3.	Clone the repository:
git clone https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
4.	Change to the top git repository and check out correct version:
cd **TBD**
5.	Change to the face detection tutorial folder:
```
cd openvino-with-fpga-hello-world-face-detection
```
### Option #2: Using SVN Export to Download Only This Tutorial

1.	Bring up a command shell prompt by opening a terminal (such as xterm) or selecting a terminal that is already open.
2.	Create a "tutorials" directory where we can download the Face Detection tutorial and then change to it:
```
mkdir -p tutorials/openvino-with-fpga-hello-world-face-detection
cd tutorials/openvino-with-fpga-hello-world-face-detection
```
3.	Download the subdirectory for just this tutorial for the specific version from the repository:
```
svn export https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
```
4.	Change to the face detection tutorial folder:
```
cd openvino-with-fpga-hello-world-face-detection
```
Now that we have all the files for the Face Detection Tutorial, we can take some time to look through them to see what each part of the tutorial will demonstrate.
# Tutorial Files
In the "Tutorial" directory you will see:
* cmake\ - Common CMake files
* data\ - Image, video, model, etc. data files used with this tutorial
* doc_support\ - Supporting documentation files including images, etc.
* scripts\ - Common helper scripts
* Readme.md - The top level of this tutorial (this page)
* Images/  - images for the Readme.md
* Others? **FROM DAVID’S FILE**

# OpenVINO™ Toolkit Overview and Terminology
Let us begin with a brief overview of the OpenVINO™ toolkit and what this tutorial will be covering. The OpenVINO™ toolkit enables the quick deployment of convolutional neural networks (CNN) for heterogeneous execution on Intel® hardware while maximizing performance. This is done using the Intel® Deep Learning Deployment Toolkit (Intel® DL Deployment Toolkit) included within the OpenVINO™ toolkit with its main components shown below.
![OV Overview](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/OV%20Overview.png)

The basic flow is:
1.	Use a tool, such as Caffe, to create and train a CNN inference model
2.	Run the created model through Model Optimizer to produce an optimized Intermediate Representation (IR) stored in files (.bin and .xml) for use with the Inference Engine
3.	The User Application then loads and runs models onto devices using the Inference Engine and the IR files

This tutorial will focus on the last step, the User Application and using the Inference Engine to run models on CPU, GPU, FPGA and Myriad.
## Using the Inference Engine
Below is a more detailed view of the User Application and Inference Engine:
![IE Graphic](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/IE%20graphic.png)

The Inference Engine includes a plugin library for each supported device that has been optimized for the Intel® hardware device CPU, GPU, FPGA and Myriad. From here, we will use the terms "device" and “plugin” with the assumption that one infers the other (e.g. CPU device infers the CPU plugin and vice versa). As part of loading the model, the User Application tells the Inference Engine which device to target which in turn loads the associated plugin library to later run on the associated device. The Inference Engine uses “blobs” for all data exchanges, basically arrays in memory arranged according the input and output data of the model.
![face_detection](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/face_detection.png)

## Face Detection Sample
In this Face Detection sample, the model estimates the head pose based on the face image it is given. The face detection model estimates the age, gender and head pose estimation. After the head pose model has processed the face, the application will draw a set of axes over the face, indicating the Yaw, Pitch, and Roll orientation of the head. A sample output showing the results where the three axes appears below. The metrics reported also include the time to run the head pose model.

![face_detection_overlay](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection/blob/master/Images/face_detection_overlay.png)

In the image above, the three axes intersect in the center of the head. The blue line represents Roll, and it extends from the center of the head to the front and the back of the head. The red line represents Pitch, and is drawn from the center of the head to the left ear. The green line represents Yaw, and is drawn from the center of the head to the top of the head.

For details about the models see the Full Tutorial: https://github.com/intel-iot-devkit/inference-tutorials-generic/blob/openvino_toolkit_r3_0/face_detection_tutorial/step_4/Readme.md#introduction

Now let us build and run the complete application and see how it runs all three analysis models.

# Build
Open up a terminal (such as xterm) or use an existing terminal to get to a command shell prompt.
Change to the directory containing the Hello World files:
```
cd Downloads/tutorial
```

First, we source variables in case it wasn’t done since the last reboot
```
source /home/<user>/setup_env.sh
```

Now we need to create a directory to build the tutorial in and change to it.
```
mkdir build
cd build
```

The last thing we need to do before compiling is to configure the build settings and build the executable. We do this by running CMake to set the build target and file locations. Then, we run Make to build the executable.

```
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

**Optional:**We could instead run the make this using the following command to run across multiple pieces of hardware and speed up the process
```
make -j $(nproc)
```

To prepare for running on the FPGA, we should load a bitstream that works well for object detection. OpenVINO Toolkit with support for FPGA includes some that we can use.

We will program the bitstream with the command:
```
aocl program acl0 /opt/intel/computer_vision_sdk_fpga_2018.3.343/a10_devkit_bitstreams/2-0-1_A10DK_FP11_ResNet50-101.aocx
```
