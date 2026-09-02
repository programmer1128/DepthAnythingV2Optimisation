#include "DepthModel.hpp"
#include <thread>
#include <iostream>
#include <unordered_map>

DepthModel::DepthModel(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "DepthAnythingV2Engine"),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_shape({1, 3, 518, 518}) 
{

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //Generic Execution Provider API
    std::unordered_map<std::string, std::string> ov_options;
    
    // Target the Intel Integrated Graphics (Iris Xe)
    ov_options["device_type"] = "CPU"; 
    
    // Use the generic Append API. This tells ONNX to dynamically load 
    // the OpenVINO .so files we linked from the Python folder!
    session_options.AppendExecutionProvider("OpenVINO", ov_options);

    std::cout << "Compiling model for Intel iGPU... (This may take a minute on first run)\n";
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
}

const float* DepthModel::infer(const std::vector<float>& input_tensor_data) 
{
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor_data.data()),
        input_tensor_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );

    return output_tensors[0].GetTensorMutableData<float>();
}