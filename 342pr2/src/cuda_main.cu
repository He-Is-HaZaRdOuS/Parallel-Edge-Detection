/**
 *
 * CENG342 Project-2
 *
 * Edge Detection
 *
 * Usage:  executable <input.jpg> <output.jpg> <threadsPerBlock> <sequential_output.jpg>
 *
 * @group_id 07
 * @author  Yousif
 * @author  Türker
 * @author  Eren
 * @author  Aysara
 *
 * @version 1.0, 18 May 2024
 */

// ReSharper disable CppUseAuto
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define KERNEL_DIMENSION 3
#define THRESHOLD 40
#define USE_THRESHOLD 0

__constant__ float sobelX[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {-1, 0, 1},{-2, 0, 2},{-1, 0, 1} };
__constant__ float sobelY[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {-1, -2, -1},{0, 0, 0},{1, 2, 1} };

//Do not use global variables

/* CUDA Kernel that applies Sobel's filter */
__global__ void CUDAedgeDetection(const uint8_t* img, uint8_t* buffer, const uint64_t width, const uint64_t height) {
    /* Sobel Convolution Kernels */
    float slider[KERNEL_DIMENSION][KERNEL_DIMENSION] = {0.0};

    /* Find X and Y indices of current thread */
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    for(int wy = 0; wy < KERNEL_DIMENSION; ++wy) {
        for(int wx = 0; wx < KERNEL_DIMENSION; ++wx) {
            int xIndex = (x + wx - 1);
            int yIndex = (y + wy - 1);

            /* Clamp */
            if(xIndex <= 0)
                xIndex = -xIndex;
            if(yIndex <= 0)
                yIndex = -yIndex;
            if(xIndex >= width)
                xIndex = width - 1;
            if(yIndex >= height)
                yIndex = height - 1;

            /* Build up sliding window */
            slider[wy][wx] = img[xIndex + yIndex * width];
        }
    }

    float sumX = 0;
    float sumY = 0;

    /* Convolve */
    for(int ky = 0; ky < KERNEL_DIMENSION; ++ky) {
        for(int kx = 0; kx < KERNEL_DIMENSION; ++kx) {
            sumX = sumX + slider[kx][ky] * sobelX[kx][ky];
            sumY = sumY + slider[kx][ky] * sobelY[kx][ky];
        }
    }
    float magnitude = sqrtf((sumX*sumX)+(sumY*sumY));

#if USE_THRESHOLD
    /* Clamp down color values if below THRESHOLD */
    buffer[y * width + x] = magnitude > THRESHOLD ? 255 : 0;
#else
    /* Otherwise use whatever value outputted from square root */
    buffer[y * width + x] = static_cast<uint8_t>(magnitude);
#endif
}

class Timer {
public:
    Timer() {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
    }

    ~Timer() = default;

    double Stop() {
        m_EndTimepoint = std::chrono::high_resolution_clock::now();

        const uint64_t start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        const uint64_t end = std::chrono::time_point_cast<std::chrono::microseconds>(m_EndTimepoint).time_since_epoch().count();

        const std::chrono::duration<uint64_t, std::ratio<1, 1000000>>::rep duration = (end - start);
        return static_cast<double>(duration) * 0.000001;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_EndTimepoint;
};

int main(int argc, char* argv[]) {

    /* Abort if # of CLA is invalid */
    if(argc != 5){
        std::cerr << "Invalid number of arguments, aborting...\n";
        std::cerr << "Try ./cuda <input.jpg> <output.jpg> <threadsPerBlock> <sequential.jpg>\n";
        exit(1);
    }
    int width, height, bpp;

    /* Prepend path to input and output filenames */
    std::string inputPath = RESOURCES_PATH;
    std::string outputPath = CUDA_OUTPUT_PATH;
    inputPath = inputPath + argv[1];
    outputPath = outputPath + argv[2];
    const int threadsPerBlock = std::stoi(argv[3]);
    if(threadsPerBlock <= 0) {
        std::cerr << "Invalid argument provided, aborting...\n";
        std::cerr << "Argument <threadsPerBlock> should be a positive integer bigger than 0\n";
        exit(1);
    }
    const int threadsPerDimension = sqrt(threadsPerBlock);

    /* Read image in grayscale */
    uint8_t *input_image = stbi_load(inputPath.c_str(), &width, &height, &bpp, CHANNEL_NUM);

    if(stbi_failure_reason()) {
        std::cerr << stbi_failure_reason() << " \"" + inputPath + "\"\n";
        std::cerr << "Aborting...\n";
        exit(1);
    }

    printf("Width: %d  Height: %d  BPP: %d \n",width, height, bpp);
    printf("Input: %s , Output: %s  \n",inputPath.c_str(), outputPath.c_str());

    uint8_t *cuda_input, *cuda_output;

    /* allocate vram for buffers */
    cudaMalloc(reinterpret_cast<void **>(&cuda_input), (width*height));
    cudaMalloc(reinterpret_cast<void **>(&cuda_output), (width*height));

    /* memcpy input buffer from system memory into vram */
    cudaMemcpy(cuda_input, input_image, width * height, cudaMemcpyHostToDevice);
    /* memset output buffer inside vram */
    cudaMemset(cuda_output, 0, width * height);

    /* Launch ((width*height)/GRID_SIZE^2) many SM blocks */
    dim3 blkCnt(ceil(width/threadsPerDimension), ceil(height/threadsPerDimension), 1);
    /* Assign GRID_SIZE size chunks for each SP */
    dim3 thPerBlk(threadsPerDimension, threadsPerDimension, 1);

    std::cout << "SM blocks queued up: " << blkCnt.x*blkCnt.y << ". SP threads queued up: " << thPerBlk.x*thPerBlk.y << ". Total thrads queued up: " << blkCnt.x*blkCnt.y*thPerBlk.x*thPerBlk.y << std::endl;

    /* Start the timer */
    Timer t;

    /* Call CUDA Kernel */
    CUDAedgeDetection<<<blkCnt, thPerBlk>>>(cuda_input, cuda_output, width, height);
    /* Capture errors, if any */
    cudaError_t cuda_err = cudaDeviceSynchronize();

    /* Stop the timer */
    double elapsedTime = t.Stop();

    /* Write error to stdout */
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA Synchronization failed!: %s\n", cudaGetErrorName(cuda_err));
        exit(1);
    }

    printf("Elapsed time: %lf seconds (%lf ms) \n",elapsedTime, elapsedTime*1000);

    /* memcpy output buffer from vram back to input buffer in system memory */
    cudaMemcpy(input_image, cuda_output, width * height, cudaMemcpyDeviceToHost);

    /* free vram buffers */
    cudaFree(cuda_input);
    cudaFree(cuda_output);

    /* Write input buffer to disk */
    stbi_write_jpg(outputPath.c_str(), width, height, CHANNEL_NUM, input_image, 100);
    stbi_image_free(input_image);

    /* Check if the two image outputs are identical */
    {
        /* Prepend path to input and output filenames */
        std::string alt_input = SEQUENTIAL_OUTPUT_PATH;
        std::string par_input = CUDA_OUTPUT_PATH;
        alt_input = alt_input + argv[4];
        par_input = par_input + argv[2];
        uint8_t *alt_img, *par_img;
        int seq_width, seq_height, seq_bpp;
        int par_width, par_height, par_bpp;

        /* Read image in grayscale */
        alt_img = stbi_load(alt_input.c_str(), &seq_width, &seq_height, &seq_bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + alt_input + "\"\n";
            std::cerr << "Aborting...\n";
            exit(1);
        }

        par_img = stbi_load(par_input.c_str(), &par_width, &par_height, &par_bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + par_input + "\"\n";
            std::cerr << "Aborting...\n";
            stbi_image_free(alt_img);
            exit(1);
        }

        std::cout << "Comparing " << alt_input << " and " << par_input << std::endl;

        /* Make sure Local and Alternate outputs are the same */
        int err_cnt = 0;
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                if(par_img[x + y * width] != alt_img[x + y * width]) {
                    ++err_cnt;
                }
            }
        }
        if(err_cnt == 0)
            std::cout << "CUDA and Sequential images are identical\n";
        else
            std::cout << err_cnt << " pixels are mismatched\n";

        /* Let go of STB image buffers */
        stbi_image_free(alt_img);
        stbi_image_free(par_img);
    }

    return 0;
}
