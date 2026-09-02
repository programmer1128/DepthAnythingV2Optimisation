#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "DepthModel.hpp"
#include <thread>
#include <iostream>


class DepthModel 
{
    public:
        explicit DepthModel(const std::string& model_path);
        ~DepthModel() = default;

        // Runs inference on the preprocessed tensor buffer
        const float* infer(const std::vector<float>& input_tensor_data);

    private:
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        Ort::MemoryInfo memory_info;

        std::vector<const char*> input_names = {"input"};
        std::vector<const char*> output_names = {"depth"};
        std::vector<int64_t> input_shape;
};