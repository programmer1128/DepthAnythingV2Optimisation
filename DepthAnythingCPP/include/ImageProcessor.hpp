#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <immintrin.h>

class ImageProcessor {
public:
    static constexpr int TARGET_WIDTH = 518;
    static constexpr int TARGET_HEIGHT = 518;
    static constexpr int CHANNELS = 3;

    // Normalizes, transposes HWC->CHW, and loads into output buffer using AVX2 + multithreading
    static void preprocess(const cv::Mat& src, std::vector<float>& dst_tensor);

    // Converts raw float depth map back into visual colorized depth map
    static cv::Mat postprocess(const float* raw_depth, int orig_w, int orig_h);
   static void saveResizedDepthToBin(const float* raw_depth, int orig_w, int orig_h, const std::string& filepath);

private:
    // Helper function: Applies CLAHE to the lightness channel in CIE L*a*b* space
    static cv::Mat applyCLAHE(const cv::Mat& src);
    // Worker function: processes a slice of rows on a dedicated thread
    static void processRowSlice(const cv::Mat& rgb_img, float* r_plane, float* g_plane, float* b_plane, int start_row, int end_row);
};
