#include "ImageProcessor.hpp"
#include <thread>
#include <vector>
#include <fstream>
#include <vector>
#include <iostream>

cv::Mat ImageProcessor::applyCLAHE(const cv::Mat& src) 
{
    // 1. Convert BGR to CIE L*a*b* color space
    cv::Mat lab_img;
    cv::cvtColor(src, lab_img, cv::COLOR_BGR2LAB);

    // 2. Split L*a*b* image into 3 individual channels (L, a, b)
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_img, lab_planes);

    // --- GEOTIFF ADAPTIVE CONFIGURATION ---
    // Target ~64x64 pixels per tile for fine terrain detail
    const int target_tile_pixels = 64;
    
    // Dynamically calculate grid dimension (ensuring a minimum 8x8 grid)
    int grid_x = std::max(8, src.cols / target_tile_pixels);
    int grid_y = std::max(8, src.rows / target_tile_pixels);

    // Use a conservative clip limit (1.2) to prevent amplifying satellite/aerial noise
    const double clip_limit = 1.2;
    // -------------------------------------

    // 3. Create CLAHE with dynamic grid size & adjusted clip limit
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(grid_x, grid_y));
    clahe->apply(lab_planes[0], lab_planes[0]);

    // 4. Merge updated L channel back with original a & b channels
    cv::merge(lab_planes, lab_img);

    // 5. Convert back to BGR color space for downstream processing
    cv::Mat enhanced_bgr;
    cv::cvtColor(lab_img, enhanced_bgr, cv::COLOR_LAB2BGR);

    return enhanced_bgr;
}

void ImageProcessor::saveResizedDepthToBin(const float* raw_depth, int orig_w, int orig_h, const std::string& filepath) 
{
     //Wrap the 518x518 raw float array into an OpenCV float matrix
     cv::Mat depth_518(TARGET_HEIGHT, TARGET_WIDTH, CV_32FC1, const_cast<float*>(raw_depth));

     //Bilinear interpolation back to the original GeoTIFF / Image resolution
     cv::Mat resized_depth;
     cv::resize(depth_518, resized_depth, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

     //Write the exact (orig_h x orig_w) binary float matrix to disk
     std::ofstream out_file(filepath, std::ios::binary | std::ios::trunc);
     if (!out_file.is_open()) 
     {
         std::cerr << "error opening binary file: " << filepath << "\n";
         return;
     }

     size_t total_bytes = orig_w * orig_h * sizeof(float);
     out_file.write(reinterpret_cast<const char*>(resized_depth.data), total_bytes);
     out_file.close();

     std::cout << "Saved resized depth matrix (" << orig_w << "x" << orig_h << ", " << 
         total_bytes << " bytes) to: " << filepath << "\n";
}


void ImageProcessor::processRowSlice(const cv::Mat& rgb_img, float* r_plane, float* g_plane, float* b_plane, int start_row, int end_row) 
{
    // Normalization constants: val = (pixel / 255.0 - mean) / std = pixel * scale + bias
    const float scale_r = 1.0f / (255.0f * 0.229f);
    const float scale_g = 1.0f / (255.0f * 0.224f);
    const float scale_b = 1.0f / (255.0f * 0.225f);

    const float bias_r = -0.485f / 0.229f;
    const float bias_g = -0.456f / 0.224f;
    const float bias_b = -0.406f / 0.225f;

    const __m256 v_scale_r = _mm256_set1_ps(scale_r);
    const __m256 v_scale_g = _mm256_set1_ps(scale_g);
    const __m256 v_scale_b = _mm256_set1_ps(scale_b);

    const __m256 v_bias_r = _mm256_set1_ps(bias_r);
    const __m256 v_bias_g = _mm256_set1_ps(bias_g);
    const __m256 v_bias_b = _mm256_set1_ps(bias_b);

    for (int h = start_row; h < end_row; ++h) 
    {
        const uchar* row_ptr = rgb_img.ptr<uchar>(h);
        int row_offset = h * TARGET_WIDTH;

        int w = 0;
        // Vectorized SIMD loop: 8 pixels at a time
        for (; w <= TARGET_WIDTH - 8; w += 8) 
        {
            float r_temp[8], g_temp[8], b_temp[8];

            for (int i = 0; i < 8; ++i) 
            {
                r_temp[i] = static_cast<float>(row_ptr[(w + i) * 3 + 0]);
                g_temp[i] = static_cast<float>(row_ptr[(w + i) * 3 + 1]);
                b_temp[i] = static_cast<float>(row_ptr[(w + i) * 3 + 2]);
            }

            // Load into 256-bit vector registers
            __m256 vr = _mm256_loadu_ps(r_temp);
            __m256 vg = _mm256_loadu_ps(g_temp);
            __m256 vb = _mm256_loadu_ps(b_temp);

            // Fused Multiply-Add: (pixel * scale) + bias
            vr = _mm256_fmadd_ps(vr, v_scale_r, v_bias_r);
            vg = _mm256_fmadd_ps(vg, v_scale_g, v_bias_g);
            vb = _mm256_fmadd_ps(vb, v_scale_b, v_bias_b);

            // Store directly into separate CHW planar channels
            _mm256_storeu_ps(&r_plane[row_offset + w], vr);
            _mm256_storeu_ps(&g_plane[row_offset + w], vg);
            _mm256_storeu_ps(&b_plane[row_offset + w], vb);
        }

         // Scalar fallback for remaining tail pixels (518 is not a multiple of 8)
         for (; w < TARGET_WIDTH; ++w) 
         {
             r_plane[row_offset + w] = static_cast<float>(row_ptr[w * 3 + 0]) * scale_r + bias_r;
             g_plane[row_offset + w] = static_cast<float>(row_ptr[w * 3 + 1]) * scale_g + bias_g;
             b_plane[row_offset + w] = static_cast<float>(row_ptr[w * 3 + 2]) * scale_b + bias_b;
         }
    }
}

void ImageProcessor::preprocess(const cv::Mat& src, std::vector<float>& dst_tensor) 
{
     // High-res GeoTIFF is passed here -> applyCLAHE calculates dynamic grid size based on full resolution
    cv::Mat enhanced_src = applyCLAHE(src);
     
    cv::Mat resized, rgb;
    cv::resize(src, resized, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    dst_tensor.resize(CHANNELS * TARGET_HEIGHT * TARGET_WIDTH);
    float* r_plane = dst_tensor.data();
    float* g_plane = r_plane + (TARGET_HEIGHT * TARGET_WIDTH);
    float* b_plane = g_plane + (TARGET_HEIGHT * TARGET_WIDTH);

    // Multithreaded Row-wise Execution across all 8 CPU threads
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    int rows_per_thread = TARGET_HEIGHT / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) 
    {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? TARGET_HEIGHT : start_row + rows_per_thread;
        workers.emplace_back(processRowSlice, std::cref(rgb), r_plane, g_plane, b_plane, start_row, end_row);
    }

    for (auto& worker : workers) 
    {
        if (worker.joinable()) worker.join();
    }
}

cv::Mat ImageProcessor::postprocess(const float* raw_depth, int orig_w, int orig_h) 
{
    cv::Mat depth_mat(TARGET_HEIGHT, TARGET_WIDTH, CV_32FC1, const_cast<float*>(raw_depth));

    double min_val, max_val;
    cv::minMaxLoc(depth_mat, &min_val, &max_val);

    cv::Mat depth_norm;
    depth_mat.convertTo(depth_norm, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));

    cv::Mat color_depth;
    cv::applyColorMap(depth_norm, color_depth, cv::COLORMAP_INFERNO);

    cv::Mat final_output;
    cv::resize(color_depth, final_output, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_CUBIC);
    return final_output;
}
