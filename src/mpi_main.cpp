#include "mpi.h"
#include "timer.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define KERNEL_DIMENSION 3
#define THRESHOLD 40
#define USE_THRESHOLD 0
#define USE_LOAD_BALANCING 0

// Do not use global variables

/* Function prototypes */
void par_edgeDetection(uint8_t *input_image, int width, int height, int rank,
                       int comm_sz, int s_offset);
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION],
               float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]);

int main(int argc, char *argv[]) {
  /* Abort if # of CLA is invalid */
  if (argc != 4) {
    std::cerr << "Invalid number of arguments, aborting...";
    exit(1);
  }

  MPI_Init(&argc, &argv);
  int m_rank, comm_sz, width, height, bpp;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
  std::string inputPath, outputPath;
  uint8_t *input_image = nullptr;

  if (m_rank == 0) {
    /* Prepend path to input and output filenames */
    inputPath = RESOURCES_PATH;
    outputPath = PARALLEL_OUTPUT_PATH;
    inputPath = inputPath + argv[1];
    outputPath = outputPath + argv[2];

    /* Read image in grayscale */
    input_image =
        stbi_load(inputPath.c_str(), &width, &height, &bpp, CHANNEL_NUM);

    /* If image could not be opened, Abort */
    if (stbi_failure_reason()) {
      std::cerr << stbi_failure_reason() << " \"" + inputPath + "\"\n";
      std::cerr << "Aborting...\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
    }

    printf("Width: %d  Height: %d  BPP: %d \n", width, height, bpp);
    printf("Input: %s , Output: %s  \n", inputPath.c_str(), outputPath.c_str());
  }

  /* Broadcast updated variables to other processes */
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Partition work among processes */
  const int zoneHeight = height / comm_sz;
  const int zoneSize = zoneHeight * width;
  const int offset = (comm_sz == 1) ? 0 : 2;

  /* Allocate a temporary output buffer for each process */
  uint8_t *temp_out = static_cast<uint8_t *>(malloc(
      width * (zoneHeight + offset) * sizeof(uint8_t))); // NOLINT(*-use-auto)

#if !USE_LOAD_BALANCING
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(m_rank, &mask);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
    perror("sched_setaffinity");
    assert(false);
  }
  if (m_rank == 0)
    std::cout << "Load Balancing: OFF" << std::endl;
#else
  if (m_rank == 0)
    std::cout << "Load Balancing: ON" << std::endl;
#endif

  /* Start the timer */
  Timer t;

  /* Map zones to processes */
  if (m_rank == 0) {
    /* memcpy 0 - zone+2 to process 0 from input */
    memcpy(temp_out, input_image,
           ((zoneHeight + offset) * width) * sizeof(uint8_t));

    for (int dest = 1; dest < comm_sz; ++dest) {
      if (dest != comm_sz - 1) {
        /* Send zone-1 - zone+1 to intermediate processes */
        MPI_Send(&input_image[(zoneSize * dest) - width],
                 (zoneHeight + offset) * width, MPI_UINT8_T, dest, 1,
                 MPI_COMM_WORLD);
      } else {
        /* Send zone-2 - zone to last process */
        MPI_Send(&input_image[(zoneSize * dest) - (width * 2)],
                 (zoneHeight + offset) * width, MPI_UINT8_T, dest, 1,
                 MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Recv(temp_out, (zoneHeight + offset) * width, MPI_UINT8_T, 0, 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /* Apply Sobel's operator */
  par_edgeDetection(temp_out, width, zoneHeight + offset, m_rank, comm_sz,
                    offset);

  /* Collect sub-solutions into one buffer on process 0 */
  MPI_Gather(temp_out, zoneSize, MPI_UINT8_T, input_image, zoneSize,
             MPI_UINT8_T, 0, MPI_COMM_WORLD);

  /* Synchronize and stop timer */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsedTime = t.Stop();

  if (m_rank == 0) {
    printf("Elapsed time: %lf seconds (%lf ms) \n", elapsedTime,
           elapsedTime * 1000);
    /* Write image to disk */
    stbi_write_jpg(outputPath.c_str(), width, height, CHANNEL_NUM, input_image,
                   100);
    stbi_image_free(input_image);
  }

  /* Let go of heap memory */
  free(temp_out);

  /* Verify sequential and parallel images are identical */
  if (m_rank == 0) {
    /* Prepend path to input and output filenames */
    std::string seq_input = SEQUENTIAL_OUTPUT_PATH;
    std::string par_input = PARALLEL_OUTPUT_PATH;
    seq_input = seq_input + argv[3];
    par_input = par_input + argv[2];
    uint8_t *seq_img, *par_img;
    int seq_width, seq_height, seq_bpp;
    int par_width, par_height, par_bpp;

    /* Read image in grayscale */
    seq_img = stbi_load(seq_input.c_str(), &seq_width, &seq_height, &seq_bpp,
                        CHANNEL_NUM);

    /* If image could not be opened, Abort */
    if (stbi_failure_reason()) {
      std::cerr << stbi_failure_reason() << " \"" + seq_input + "\"\n";
      std::cerr << "Aborting...\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
    }

    par_img = stbi_load(par_input.c_str(), &par_width, &par_height, &par_bpp,
                        CHANNEL_NUM);

    /* If image could not be opened, Abort */
    if (stbi_failure_reason()) {
      std::cerr << stbi_failure_reason() << " \"" + par_input + "\"\n";
      std::cerr << "Aborting...\n";
      stbi_image_free(seq_img);
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
    }

    std::cout << "Comparing " << seq_input << " and " << par_input << std::endl;

    /* Make sure sequential and parallel outputs are the same */
    int err_cnt = 0;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (par_img[x + y * width] != seq_img[x + y * width]) {
          ++err_cnt;
        }
      }
    }
    if (err_cnt == 0)
      std::cout << "Sequential and Parallel images are identical\n";
    else
      std::cout << err_cnt << " pixels are mismatched\n";

    /* Let go of STB image buffers */
    stbi_image_free(seq_img);
    stbi_image_free(par_img);
  }

  MPI_Finalize();
  return 0;
}

/* Apply Sobel's Operator  */
void par_edgeDetection(uint8_t *input_image, const int width, const int height,
                       const int rank, const int comm_sz, const int s_offset) {

  /* Declare Kernels */
  float sobelX[KERNEL_DIMENSION][KERNEL_DIMENSION] = {
      {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  float sobelY[KERNEL_DIMENSION][KERNEL_DIMENSION] = {
      {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  float slider[KERNEL_DIMENSION][KERNEL_DIMENSION];

  /* Allocate temporary memory to construct final sub-image */
  uint8_t *output_image = static_cast<uint8_t *>(
      malloc(width * height * sizeof(uint8_t))); // NOLINT(*-use-auto)

  /* Iterate through all pixels */
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int wy = 0; wy < KERNEL_DIMENSION; ++wy) {
        for (int wx = 0; wx < KERNEL_DIMENSION; ++wx) {
          int xIndex = (x + wx - 1);
          int yIndex = (y + wy - 1);
          /* Duplicate opposite edge values if on barrier pixels */
          if (xIndex < 0)
            xIndex = -xIndex;
          if (yIndex < 0)
            yIndex = -yIndex;
          if (xIndex >= width)
            xIndex = width - 1;
          if (yIndex >= height)
            yIndex = height - 1;

          /* Build up sliding window */
          slider[wy][wx] = input_image[xIndex + yIndex * width];
        }
      }

      /* Convolve sliding window with kernels (Sobel X and Y gradient) */
      const float gx = convolve(slider, sobelX);
      const float gy = convolve(slider, sobelY);
      float magnitude = sqrt(gx * gx + gy * gy);

#if USE_THRESHOLD
      /* Clamp down color values if below THRESHOLD */
      output_image[x + (y)*width] = magnitude > THRESHOLD ? 255 : 0;
#else
      /* Otherwise use whatever value outputted from square root */
      output_image[x + y * width] = static_cast<uint8_t>(magnitude);
#endif
    }
  }

  /* Shift around zones based on their rank */
  int offset;
  if (rank == 0) {
    offset = 0;
  } else if (rank != comm_sz - 1) {
    offset = 1;
  } else {
    offset = 2;
  }

  /* copy zone to input buffer */
  for (int y = 0; y < height - s_offset; ++y) {
    for (int x = 0; x < width; ++x) {
      input_image[x + (y)*width] = output_image[x + (y + offset) * width];
    }
  }

  /* De-allocate temporary memory */
  free(output_image);
}

/* Convolve slider across kernel (multiply and sum values of two arrays) */
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION],
               float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]) {
  float sum = 0;
  for (int y = 0; y < KERNEL_DIMENSION; ++y) {
    for (int x = 0; x < KERNEL_DIMENSION; ++x) {
      sum = sum + slider[x][y] * kernel[x][y];
    }
  }

  return sum;
}
