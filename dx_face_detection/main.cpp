/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine interactive_face_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "face_detection.hpp"
//#include "mkldnn/mkldnn_extension_ptr.hpp"		// deprecated 4.20
#include <ext_list.hpp>


#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio_c.h>				// added for 4.20, defs 
#include <ie_precision.hpp>							// added for 4.20
#include <ie_cnn_net_reader.h>						// added for 4.20

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ExecutableNetwork net;
    InferenceEngine::InferencePlugin * plugin = NULL;
    InferRequest::Ptr request;
    std::string & commandLineFlag;
    std::string topoName;
    const int maxBatch;

    BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch)
        : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch) {}

    virtual ~BaseDetection() {}

    ExecutableNetwork* operator ->() {
        return &net;
    }
    virtual InferenceEngine::CNNNetwork read()  = 0;

    virtual void submitRequest() {
        if (!enabled() || request == nullptr) return;
        request->StartAsync();
    }

    virtual void wait() {
        if (!enabled()|| !request) return;
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
    void printPerformanceCounts() {
        if (!enabled()) {
            return;
        }
        slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
        ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
    }
};

struct FaceDetectionClass : BaseDetection {
    std::string input;
    std::string output;
    int maxProposalCount = 0;
    int objectSize = 0;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;
    std::vector<std::string> labels;
    using BaseDetection::operator=;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        if (!enquedFrames) return;
        enquedFrames = 0;
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void enqueue(const cv::Mat &frame) {
        if (!enabled()) return;

        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        width = frame.cols;
        height = frame.rows;

        auto  inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t >(frame, inputBlob);
		enquedFrames = 1;
    }


    FaceDetectionClass() : BaseDetection(FLAGS_m, "Face Detection", 1) {}
    InferenceEngine::CNNNetwork read() override {
        slog::info << "Loading network files for Face Detection" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is set to  "<< maxBatch << slog::endl;
        netReader.getNetwork().setBatchSize(maxBatch);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";

        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Face Detection inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Face Detection outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one output");
        }
        auto& _output = outputInfo.begin()->second;
        output = outputInfo.begin()->first;

        const auto outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
        if (outputLayer->type != "DetectionOutput") {
            throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
                ") should be DetectionOutput, but was " +  outputLayer->type);
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            throw std::logic_error("Face Detection network output layer (" +
                output + ") should have num_classes integer attribute");
        }

        const int num_classes = outputLayer->GetParamAsInt("num_classes");
        if (labels.size() != num_classes) {
            if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
        const InferenceEngine::SizeVector outputDims = _output->dims;
        maxProposalCount = outputDims[1];
        objectSize = outputDims[0];
        if (objectSize != 7) {
            throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                                           std::to_string(outputDims.size()));
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        slog::info << "Loading Face Detection model to the "<< FLAGS_d << " plugin" << slog::endl;
        input = inputInfo.begin()->first;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request->GetBlob(output)->buffer().as<float *>();

        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= FLAGS_t) {
                continue;
            }

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;

            if ((image_id < 0) || (image_id >= maxBatch)) {  // indicates end of detections
                break;
            }
            if (FLAGS_r) {
                std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }

            results.push_back(r);
        }
    }
};

struct AgeGenderDetection : BaseDetection {
    std::string input;
    std::string outputAge;
    std::string outputGender;
    int enquedFaces = 0;


    using BaseDetection::operator=;
    AgeGenderDetection() : BaseDetection(FLAGS_m_ag, "Age Gender", FLAGS_n_ag) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces >= maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Age Gender detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        auto  inputBlob = request->GetBlob(input);

        matU8ToBlob<float>(face, inputBlob, enquedFaces);
        enquedFaces++;
    }

    struct Result { float age; float maleProb;};
    Result operator[] (int idx) const {
        auto  genderBlob = request->GetBlob(outputGender);
        auto  ageBlob    = request->GetBlob(outputAge);

        return {ageBlob->buffer().as<float*>()[idx] * 100,
                genderBlob->buffer().as<float*>()[idx * 2 + 1]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for AgeGender" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_ag);

        /** Set batch size **/
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Age Gender" << slog::endl;


        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_ag) + ".bin";
        netReader.ReadWeights(binFileName);

        // -----------------------------------------------------------------------------------------------------

        /** Age Gender network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Age Gender inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Age gender topology should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Age Gender outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Age Gender network should have two output layers");
        }
        auto it = outputInfo.begin();
        auto ageOutput = (it++)->second;
        auto genderOutput = (it++)->second;

        // if gender output is convolution, it can be swapped with age
        if (genderOutput->getCreatorLayer().lock()->type == "Convolution") {
            std::swap(ageOutput, genderOutput);
        }

        if (ageOutput->getCreatorLayer().lock()->type != "Convolution") {
            throw std::logic_error("In Age Gender network, age layer (" + ageOutput->getCreatorLayer().lock()->name +
                ") should be a Convolution, but was: " + ageOutput->getCreatorLayer().lock()->type);
        }

        if (genderOutput->getCreatorLayer().lock()->type != "SoftMax") {
            throw std::logic_error("In Age Gender network, gender layer (" + genderOutput->getCreatorLayer().lock()->name +
                ") should be a SoftMax, but was: " + genderOutput->getCreatorLayer().lock()->type);
        }
        slog::info << "Age layer: " << ageOutput->getCreatorLayer().lock()->name<< slog::endl;
        slog::info << "Gender layer: " << genderOutput->getCreatorLayer().lock()->name<< slog::endl;

        outputAge = ageOutput->name;
        outputGender = genderOutput->name;

        slog::info << "Loading Age Gender model to the "<< FLAGS_d_ag << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct HeadPoseDetection : BaseDetection {
    std::string input;
    std::string outputAngleR = "angle_r_fc";
    std::string outputAngleP = "angle_p_fc";
    std::string outputAngleY = "angle_y_fc";
    int enquedFaces = 0;
    cv::Mat cameraMatrix;
    HeadPoseDetection() : BaseDetection(FLAGS_m_hp, "Head Pose", FLAGS_n_hp) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Head Pose detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        auto  inputBlob = request->GetBlob(input);

        matU8ToBlob<float>(face, inputBlob, enquedFaces);
        enquedFaces++;
    }

    struct Results {
        float angle_r;
        float angle_p;
        float angle_y;
    };

    Results operator[] (int idx) const {
        auto  angleR = request->GetBlob(outputAngleR);
        auto  angleP = request->GetBlob(outputAngleP);
        auto  angleY = request->GetBlob(outputAngleY);

        return {angleR->buffer().as<float*>()[idx],
                angleP->buffer().as<float*>()[idx],
                angleY->buffer().as<float*>()[idx]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for Head Pose detection " << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_hp);
        /** Set batch size to maximum currently set to one provided from command line **/
        netReader.getNetwork().setBatchSize(maxBatch);
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to  " << netReader.getNetwork().getBatchSize() << " for Head Pose Network" << slog::endl;
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_hp) + ".bin";
        netReader.ReadWeights(binFileName);


        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Head Pose Network inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Head Pose topology should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Head Pose network outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 3) {
            throw std::logic_error("Head Pose network should have 3 outputs");
        }
        std::map<std::string, bool> layerNames = {
            {outputAngleR, false},
            {outputAngleP, false},
            {outputAngleY, false}
        };

        for (auto && output : outputInfo) {
            auto layer = output.second->getCreatorLayer().lock();
            if (layerNames.find(layer->name) == layerNames.end()) {
                throw std::logic_error("Head Pose network output layer unknown: " + layer->name + ", should be " +
                    outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
            }
            if (layer->type != "FullyConnected") {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid type: " +
                    layer->type + ", should be FullyConnected");
            }
            auto fc = dynamic_cast<FullyConnectedLayer*>(layer.get());
            if (fc->_out_num != 1) {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid out-size=" +
                    std::to_string(fc->_out_num) + ", should be 1");
            }
            layerNames[layer->name] = true;
        }

        slog::info << "Loading Head Pose model to the "<< FLAGS_d_hp << " plugin" << slog::endl;

        _enabled = true;
        return netReader.getNetwork();
    }

    void buildCameraMatrix(int cx, int cy, float focalLength) {
        if (!cameraMatrix.empty()) return;
        cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
        cameraMatrix.at<float>(0) = focalLength;
        cameraMatrix.at<float>(2) = static_cast<float>(cx);
        cameraMatrix.at<float>(4) = focalLength;
        cameraMatrix.at<float>(5) = static_cast<float>(cy);
        cameraMatrix.at<float>(8) = 1;
    }

    void drawAxes(cv::Mat& frame, cv::Point3f cpoint, Results headPose, float scale) {
        double yaw   = headPose.angle_y;
        double pitch = headPose.angle_p;
        double roll  = headPose.angle_r;

        if (FLAGS_r) {
            std::cout << "Head pose results: yaw, pitch, roll = " << yaw << ";" << pitch << ";" << roll << std::endl;
        }

        pitch *= CV_PI / 180.0;
        yaw   *= CV_PI / 180.0;
        roll  *= CV_PI / 180.0;

        cv::Matx33f        Rx(1,           0,            0,
                              0,  cos(pitch),  -sin(pitch),
                              0,  sin(pitch),  cos(pitch));
        cv::Matx33f Ry(cos(yaw),           0,    -sin(yaw),
                              0,           1,            0,
                       sin(yaw),           0,    cos(yaw));
        cv::Matx33f Rz(cos(roll), -sin(roll),            0,
                       sin(roll),  cos(roll),            0,
                              0,           0,            1);


        auto r = cv::Mat(Rz*Ry*Rx);
        buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

        cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

        xAxis.at<float>(0) = 1 * scale;
        xAxis.at<float>(1) = 0;
        xAxis.at<float>(2) = 0;

        yAxis.at<float>(0) = 0;
        yAxis.at<float>(1) = -1 * scale;
        yAxis.at<float>(2) = 0;

        zAxis.at<float>(0) = 0;
        zAxis.at<float>(1) = 0;
        zAxis.at<float>(2) = -1 * scale;

        zAxis1.at<float>(0) = 0;
        zAxis1.at<float>(1) = 0;
        zAxis1.at<float>(2) = 1 * scale;

        cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
        o.at<float>(2) = cameraMatrix.at<float>(0);

        xAxis = r * xAxis + o;
        yAxis = r * yAxis + o;
        zAxis = r * zAxis + o;
        zAxis1 = r * zAxis1 + o;

        cv::Point p1, p2;

        p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

        p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

        p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

        p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);

        cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferenceEngine::InferencePlugin & plg) const {
        if (detector.enabled()) {
            detector.net = plg.LoadNetwork(detector.read(), {});
            detector.plugin = &plg;
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        /** This sample covers 3 certain topologies and cannot be generalized **/
        std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        // ---------------------------Parsing and validation of input args--------------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // -----------------------------Read input -----------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        const bool isCamera = FLAGS_i == "cam";
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);

        // read input (video) frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }

        // ---------------------Load plugins for inference engine------------------------------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp}
        };

        FaceDetectionClass FaceDetection;
        AgeGenderDetection AgeGender;
        HeadPoseDetection HeadPose;


        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Load extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    //auto extension_ptr = make_so_pointer<InferenceEngine::MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);		// 4.20
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);												// 4.20
                    plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }

            pluginsForDevices[deviceName] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }


        // --------------------Load networks (Generated xml/bin files)-------------------------------------------

        Load(FaceDetection).into(pluginsForDevices[FLAGS_d]);
        Load(AgeGender).into(pluginsForDevices[FLAGS_d_ag]);
        Load(HeadPose).into(pluginsForDevices[FLAGS_d_hp]);


        // ----------------------------Do inference-------------------------------------------------------------
        slog::info << "Start inference " << slog::endl;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        std::chrono::high_resolution_clock::time_point wallclockStart, wallclockEnd;

        int totalFrames = 1;  // cap.read() above
        double ocv_decode_time = 0, ocv_render_time = 0;
		float fdFpsTot = 0.0; 
		float otherTotFps = 0.0; 

		double ocv_ttl_render = 0;
		double ocv_ttl_decode = 0;

		wallclockStart = std::chrono::high_resolution_clock::now();
        /** Start inference & calc performance **/
        while (true) {
        	double secondDetection = 0;

            /** requesting new frame if any*/
            cap.grab();

            auto t0 = std::chrono::high_resolution_clock::now();
            FaceDetection.enqueue(frame);
            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // ----------------------------Run face detection inference------------------------------------------
            FaceDetection.submitRequest();
            FaceDetection.wait();

            t1 = std::chrono::high_resolution_clock::now();
            ms detection = std::chrono::duration_cast<ms>(t1 - t0);

            // fetch all face results
            FaceDetection.fetchResults();

            // track and store age and gender results for all faces
            std::vector<AgeGenderDetection::Result> ageGenderResults;
            int ageGenderFaceIdx = 0;
            int ageGenderNumFacesInferred = 0;
            int ageGenderNumFacesToInfer = AgeGender.enabled() ? FaceDetection.results.size() : 0;

            // track and store head pose results for all faces
            std::vector<HeadPoseDetection::Results> headPoseResults;
            int headPoseFaceIdx = 0;
            int headPoseNumFacesInferred = 0;
            int headPoseNumFacesToInfer = HeadPose.enabled() ? FaceDetection.results.size() : 0;


            while((ageGenderFaceIdx < ageGenderNumFacesToInfer)
        		   || (headPoseFaceIdx < headPoseNumFacesToInfer)) {
            	// enqueue input batch
            	while ((ageGenderFaceIdx < ageGenderNumFacesToInfer) && (AgeGender.enquedFaces < AgeGender.maxBatch)) {
					FaceDetectionClass::Result faceResult = FaceDetection.results[ageGenderFaceIdx];
					auto clippedRect = faceResult.location & cv::Rect(0, 0, width, height);
					auto face = frame(clippedRect);
					AgeGender.enqueue(face);
					ageGenderFaceIdx++;
            	}

            	while ((headPoseFaceIdx < headPoseNumFacesToInfer) && (HeadPose.enquedFaces < HeadPose.maxBatch)) {
					FaceDetectionClass::Result faceResult = FaceDetection.results[headPoseFaceIdx];
					auto clippedRect = faceResult.location & cv::Rect(0, 0, width, height);
					auto face = frame(clippedRect);
					HeadPose.enqueue(face);
					headPoseFaceIdx++;
            	}

				t0 = std::chrono::high_resolution_clock::now();

            	// if faces are enqueued, then start inference
            	if (AgeGender.enquedFaces > 0) {
					AgeGender.submitRequest();
            	}
            	if (HeadPose.enquedFaces > 0) {
            		HeadPose.submitRequest();
            	}

            	// if there are outstanding results, then wait for inference to complete
            	if (ageGenderNumFacesInferred < ageGenderFaceIdx) {
					AgeGender.wait();
            	}
            	if (headPoseNumFacesInferred < headPoseFaceIdx) {
            		HeadPose.wait();
            	}

				t1 = std::chrono::high_resolution_clock::now();
				secondDetection += std::chrono::duration_cast<ms>(t1 - t0).count();

				// process results if there are any
				if (ageGenderNumFacesInferred < ageGenderFaceIdx) {
					for(int ri = 0; ri < AgeGender.maxBatch; ri++) {
						ageGenderResults.push_back(AgeGender[ri]);
						ageGenderNumFacesInferred++;
					}
            	}
				if (headPoseNumFacesInferred < headPoseFaceIdx) {
					for(int ri = 0; ri < HeadPose.maxBatch; ri++) {
						headPoseResults.push_back(HeadPose[ri]);
						headPoseNumFacesInferred++;
					}
            	}
            }

            // ----------------------------Processing outputs-----------------------------------------------------
			ocv_ttl_render += ocv_render_time;
			ocv_ttl_decode += ocv_decode_time;

            std::ostringstream out;
            out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                << (ocv_decode_time + ocv_render_time) << " ms";
            cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
						float currFdFps = 1000.f / detection.count();
						fdFpsTot += currFdFps;

            out.str("");
            out << "Face detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                << " ms ("
                << currFdFps << " fps)";
            cv::putText(frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                        cv::Scalar(255, 0, 0));

            if (HeadPose.enabled() || AgeGender.enabled()) {
                out.str("");
                out << (AgeGender.enabled() ? "Age Gender"  : "")
                    << (AgeGender.enabled() && HeadPose.enabled() ? "+"  : "")
                    << (HeadPose.enabled() ? "Head Pose "  : "")
                    << "time: "<< std::fixed << std::setprecision(2) << secondDetection
                    << " ms ";
                if (!FaceDetection.results.empty()) {
                    float otherFps = 1000.f / secondDetection;
					otherTotFps += otherFps;
                    out << "(" << otherFps << " fps)";
                }
                cv::putText(frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
            }

            // render results
            for(int ri = 0; ri < FaceDetection.results.size(); ri++) {
            	FaceDetectionClass::Result faceResult = FaceDetection.results[ri];
                cv::Rect rect = faceResult.location;

                out.str("");

                if (AgeGender.enabled()) {
                    out << (ageGenderResults[ri].maleProb > 0.5 ? "M" : "F");
                    out << std::fixed << std::setprecision(0) << "," << ageGenderResults[ri].age;
                } else {
                    out << (faceResult.label < FaceDetection.labels.size() ? FaceDetection.labels[faceResult.label] :
                             std::string("label #") + std::to_string(faceResult.label))
                        << ": " << std::fixed << std::setprecision(3) << faceResult.confidence;
                }

                cv::putText(frame,
                            out.str(),
                            cv::Point2f(faceResult.location.x, faceResult.location.y - 15),
                            cv::FONT_HERSHEY_COMPLEX_SMALL,
                            0.8,
                            cv::Scalar(0, 0, 255));

                if (FLAGS_r) {
                    std::cout << "Predicted gender, age = " << out.str() << std::endl;
                }

                if (HeadPose.enabled()) {
                    cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
                    HeadPose.drawAxes(frame, center, headPoseResults[ri], 50);
                }

                auto genderColor =
                		(AgeGender.enabled()) ?
                              ((ageGenderResults[ri].maleProb < 0.5) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0)) :
                              cv::Scalar(0, 255, 0);
                cv::rectangle(frame, faceResult.location, genderColor, 2);
            }

            int keyPressed;
            if (-1 != (keyPressed = cv::waitKey(1))) {
            	// done processing, save time
            	wallclockEnd = std::chrono::high_resolution_clock::now();

            	if ('s' == keyPressed) {
            		// save screen to output file
            		slog::info << "Saving screenshot" << slog::endl;
            		cv::imwrite("snapshot.bmp", frame);
            	} else {
            		break;
            	}
            }

            t0 = std::chrono::high_resolution_clock::now();
            if (!FLAGS_no_show)
                cv::imshow("Detection results", frame);

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            // end of file, for single frame file, like image we just keep it displayed to let user check what was shown
            cv::Mat newFrame;
            if (!cap.retrieve(newFrame)) {
            	// done processing, save time
            	wallclockEnd = std::chrono::high_resolution_clock::now();

				if (!FLAGS_no_wait && !FLAGS_no_show) {
                    slog::info << "Press 's' key to save a screenshot, press any other key to exit" << slog::endl;
                    while (cv::waitKey(0) == 's') {
                		// save screen to output file
                		slog::info << "Saving screenshot of image" << slog::endl;
                		cv::imwrite("screenshot.bmp", frame);
                    }
                }
                break;
            }
            frame = newFrame;  // shallow copy
			totalFrames++;
        }

		float avgFdFps = fdFpsTot/totalFrames;
		float avgAGHpFps = otherTotFps/totalFrames;

        // calculate total run time
        ms total_wallclock_time = std::chrono::duration_cast<ms>(wallclockEnd - wallclockStart);

        // report loop time
		float avgTimePerFrameMs = total_wallclock_time.count() / (float)totalFrames;

		std::string na(80, '=');
		std::string nb(80, '-');
		std::cout << na << std::endl;

		slog::info << "   Total main-loop time: " << std::fixed << std::setprecision(2)
				<< total_wallclock_time.count() << " ms " <<  slog::endl;
		slog::info << "     Total number of frames: " << totalFrames <<  slog::endl;

		std::cout << nb << std::endl;

		// Debug - OCV Times
		slog::info << "     Total OpenCV Render Time: " << ocv_ttl_render << " (" << (ocv_ttl_render / total_wallclock_time.count()) * 100 << "%)" << slog::endl;
		slog::info << "     Total OpenCV Decode Time: " << ocv_ttl_decode <<  " (" << (ocv_ttl_decode / total_wallclock_time.count()) * 100 << "%)" << slog::endl;

		slog::info << "     Avg OpenCV Render Time: " << std::fixed << std::setprecision(2) << ocv_ttl_render/totalFrames <<  "ms" << slog::endl;
		slog::info << "     Avg OpenCV Decode Time: " << std::fixed << std::setprecision(2) << ocv_ttl_decode/totalFrames <<  "ms" << slog::endl;

		std::cout << nb << std::endl;

		slog::info << "   Average time per frame:           " << std::fixed << std::setprecision(2)
					<< avgTimePerFrameMs << " ms "
					<< "(" << 1000.0f / avgTimePerFrameMs << " fps)" << slog::endl;

		slog::info << "   Average Face Detection FPS:       " << std::fixed << std::setprecision(2)
					<< avgFdFps << " fps" << slog::endl;

		slog::info << "   Average Age/Gender/Head Pose FPS: " << std::fixed << std::setprecision(2)
					<< avgAGHpFps << " fps" << slog::endl;

		std::cout << nb << std::endl;

        // ---------------------------Some perf data--------------------------------------------------
        if (FLAGS_pc) {
            FaceDetection.printPerformanceCounts();
            AgeGender.printPerformanceCounts();
            HeadPose.printPerformanceCounts();
        }

    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
