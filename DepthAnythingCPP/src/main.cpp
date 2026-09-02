#include <iostream>
#include <chrono>
#include "ImageProcessor.hpp"
#include "DepthModel.hpp"

int main() 
{
     std::string model_path = "/home/aritra/Desktop/OptimisingDepthAnythingV2/Depth-Anything-V2/DepthAnythingCPP/depth_anything_v2_vitl.onnx";
 
     DepthModel model(model_path);
     std::cout << "Model Ready! Entering interactive loop.\n\n";

     std::vector<float> input_tensor;

     while (true) 
     {
         std::cout << "Enter image path (or 'exit'): ";
         std::string path;
         std::getline(std::cin, path);

        if (path == "exit" || path == "quit") break;

        cv::Mat raw = cv::imread(path);
        if (raw.empty()) {
            std::cerr << "Invalid image path!\n\n";
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();

        //Multithreaded AVX2 Preprocessing
        ImageProcessor::preprocess(raw, input_tensor);

        //Model Inference 
        const float* raw_depth = model.infer(input_tensor);

        //Save Raw Depth to Binary (.bin)
        // 518 x 518 = 268,324 floats
        size_t total_elements = ImageProcessor::TARGET_WIDTH * ImageProcessor::TARGET_HEIGHT;
        std::string bin_name = "depth_matrix_518x518.bin";
        ImageProcessor::saveResizedDepthToBin(raw_depth, raw.cols, raw.rows, bin_name);

        //Postprocessing & Colormapping (for visual debugging)
        cv::Mat result = ImageProcessor::postprocess(raw_depth, raw.cols, raw.rows);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::string out_name = "depth_result_" + std::to_string(duration) + "ms.jpg";
        cv::imwrite(out_name, result);

        std::cout << ">> Total latency: " << duration << " ms\n";
        std::cout << ">> Saved visual depth map to: " << out_name << "\n\n";
    }

    return 0;
}