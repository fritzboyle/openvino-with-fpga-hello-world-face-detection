# openvino-with-fpga-hello-world-face-detection

**Note:** This tutorial has been written using OpenVINO™ Toolkit for Linux with FPGA Support version 2018 R3.0 and is for use with this version only. Using this tutorial with any other version may not be correct.
##Introduction

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
## Option #1: Using Git Clone to Clone the Entire Repository
1.	Bring up a command shell prompt by opening a terminal (such as xterm) or selecting a terminal that is already open.
2.	Create a "tutorials" directory where we can download the Face Detection tutorial and then change to it:
mkdir tutorials
cd tutorials
3.	Clone the repository:
git clone https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
4.	Change to the top git repository and check out correct version:
cd ********TBD
5.	Change to the face detection tutorial folder:
cd openvino-with-fpga-hello-world-face-detection
## Option #2: Using SVN Export to Download Only This Tutorial

1.	Bring up a command shell prompt by opening a terminal (such as xterm) or selecting a terminal that is already open.
2.	Create a "tutorials" directory where we can download the Face Detection tutorial and then change to it:
mkdir -p tutorials/openvino-with-fpga-hello-world-face-detection
cd tutorials/openvino-with-fpga-hello-world-face-detection
3.	Download the subdirectory for just this tutorial for the specific version from the repository:
svn export https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection
4.	Change to the face detection tutorial folder:
cd openvino-with-fpga-hello-world-face-detection
Now that we have all the files for the Face Detection Tutorial, we can take some time to look through them to see what each part of the tutorial will demonstrate.
## Tutorial FIles
In the " openvino-with-fpga-hello-world-face-detection " directory you will see:
* cmake\ - Common CMake files
* data\ - Image, video, model, etc. data files used with this tutorial
* doc_support\ - Supporting documentation files including images, etc.
* scripts\ - Common helper scripts
* Readme.md - The top level of this tutorial (this page)
* Others? **FROM DAVID’S FILE**
