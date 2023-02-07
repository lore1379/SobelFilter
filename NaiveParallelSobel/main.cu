#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include "cmath"

#include "image.h"

#define SOBEL_OP_DIM 3
#define SOBEL_OP_RADIUS (SOBEL_OP_DIM / 2)

#define THREADS_x 32
#define THREADS_y 32

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

__global__ void sobelFilterGPU(uint8_t* hGradientImgData, uint8_t* vGradientImgData, uint8_t* resultData, int w, int h) {
	
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t Idx = y * blockDim.x * gridDim.x + x;

	if (x < w && y < h)
		resultData[Idx] = (uint8_t)sqrt(pow(hGradientImgData[Idx], 2) + pow(vGradientImgData[Idx], 2));
}

__global__ void gradientConvolution(uint8_t* inputImgData, int* mask, uint8_t* gradientImgData, int w, int h) {
	
	// Calculate the global thread positions
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Starting index for calculation
	int start_r = row - SOBEL_OP_RADIUS;
	int start_c = col - SOBEL_OP_RADIUS;

	// Temp value for calculation
	int temp = 0;

	// Iterate over all the rows
	for (int i = 0; i < SOBEL_OP_DIM; i++) {
		// Go over each column
		for (int j = 0; j < SOBEL_OP_DIM; j++) {
			// Range check for rows
			if ((start_r + i) >= 0 && (start_r + i) < h) {
				// Range check for columns
				if ((start_c + j) >= 0 && (start_c + j) < w) {
					// Accumulate result
					temp += inputImgData[(start_r + i) * w + (start_c + j)] *
						mask[i * SOBEL_OP_DIM + j];
				}
			}
		}

	}
	// Write back the result
	gradientImgData[row * w + col] = (uint8_t)abs(temp);
}

__global__ void ConvertImageToGrayGpu(uint8_t* inputImgData, uint8_t* grayImgData, int channels, int w, int h) {

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t destIdx = y * blockDim.x * gridDim.x + x;
	uint32_t sourceIdx = (y * blockDim.x * gridDim.x + x) * channels;

	if (x < w && y < h)
		grayImgData[destIdx] = 0.2126 * inputImgData[sourceIdx] + 0.7152 * inputImgData[sourceIdx + 1] + 0.0722 * inputImgData[sourceIdx + 2];
}

int main(int argc, char** argv) {

	// definisco i timers 
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::duration<double, std::milli> duration;

	// read input image
	std::cout << "Loading image file...";
	
	Image inputImg("imgs_in/imageFHD.jpg");
	
	std::cout << "DONE" << std::endl;

	// create grayImg with 1 channel
	Image grayImg(inputImg.w, inputImg.h, 1);

	// Copy data to the gpu
	std::cout << "Copy data to GPU...";
	uint8_t* d_inputImgData;
	uint8_t* d_grayImgData;
	HANDLE_ERROR(cudaMalloc(&d_inputImgData, inputImg.size));
	HANDLE_ERROR(cudaMalloc(&d_grayImgData, grayImg.size));

	HANDLE_ERROR(cudaMemcpy(d_inputImgData, inputImg.data, inputImg.size, cudaMemcpyHostToDevice));
	std::cout << " DONE" << std::endl;

	// Process image on gpu
	std::cout << "Running CUDA Kernel, converting input image to grayscale...";

	dim3 dimBlock(THREADS_x, THREADS_y);
	dim3 dimGrid((inputImg.w + THREADS_x - 1) / THREADS_x, (inputImg.h + THREADS_y -1 ) / THREADS_y);
	
	ConvertImageToGrayGpu << <dimGrid, dimBlock >> > (d_inputImgData, d_grayImgData, inputImg.channels, inputImg.w, inputImg.h);
	
	std::cout << " DONE" << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU...";
	HANDLE_ERROR(cudaMemcpy(grayImg.data, d_grayImgData, grayImg.size, cudaMemcpyDeviceToHost));
	std::cout << " DONE" << std::endl;

	// write intermediate result
	// grayImg.write("imgs_out/imageFHD_gray.png");

	// STEP 2 ##### calcolo del gradiente orizzontale e verticale

	// horizontal mask on host
	int sobel_h[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	// horizontal gradient image on host
	Image hGradientImg(grayImg.w, grayImg.h, 1);

	// Copy data to the gpu
	std::cout << "Copy data to GPU...";

	// allocate in device memory space for mask and result
	int* d_hSobelMask;
	uint8_t* d_hGradientImgData;
	HANDLE_ERROR(cudaMalloc(&d_hSobelMask, SOBEL_OP_DIM * SOBEL_OP_DIM * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_hGradientImgData, hGradientImg.size));

	// copy mask on device
	HANDLE_ERROR(cudaMemcpy(d_hSobelMask, sobel_h, SOBEL_OP_DIM * SOBEL_OP_DIM * sizeof(int), cudaMemcpyHostToDevice));
	std::cout << " DONE" << std::endl;

	// Process image on gpu
	std::cout << "Running CUDA Kernel, computing horizontal gradient...";

	gradientConvolution << < dimGrid, dimBlock >> > (d_grayImgData, d_hSobelMask, d_hGradientImgData, grayImg.w, grayImg.h);

	std::cout << " DONE" << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU...";
	HANDLE_ERROR(cudaMemcpy(hGradientImg.data, d_hGradientImgData, hGradientImg.size, cudaMemcpyDeviceToHost));
	std::cout << " DONE" << std::endl;

	// hGradientImg.write("imgs_out/imageFHD_sobel_h.png");

	// vertical mask on host
	int sobel_v[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	// vertical gradient image on host
	Image vGradientImg(grayImg.w, grayImg.h, 1);

	// Copy data to the gpu
	std::cout << "Copy data to GPU...";

	int* d_vSobelMask;
	uint8_t* d_vGradientImgData;
	HANDLE_ERROR(cudaMalloc(&d_vSobelMask, SOBEL_OP_DIM * SOBEL_OP_DIM * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_vGradientImgData, vGradientImg.size));

	// copy mask on device
	HANDLE_ERROR(cudaMemcpy(d_vSobelMask, sobel_v, SOBEL_OP_DIM * SOBEL_OP_DIM * sizeof(int), cudaMemcpyHostToDevice));
	std::cout << " DONE" << std::endl;

	// Process image on gpu
	std::cout << "Running CUDA Kernel, computing horizontal gradient...";

	gradientConvolution << < dimGrid, dimBlock >> > (d_grayImgData, d_vSobelMask, d_vGradientImgData, grayImg.w, grayImg.h);

	std::cout << " DONE" << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU...";
	HANDLE_ERROR(cudaMemcpy(vGradientImg.data, d_vGradientImgData, vGradientImg.size, cudaMemcpyDeviceToHost));
	std::cout << " DONE" << std::endl;

	// vGradientImg.write("imgs_out/imageFHD_sobel_v.png");
	
	// final result
	Image finalResult(grayImg.w, grayImg.h, 1);

	uint8_t* d_finalResultData;
	HANDLE_ERROR(cudaMalloc(&d_finalResultData, finalResult.size));

	std::cout << "Running CUDA Kernel, combining horizontal and vertical gradient ... ";
	
	sobelFilterGPU << < dimGrid, dimBlock >> > (d_hGradientImgData, d_vGradientImgData, d_finalResultData, grayImg.w, grayImg.h);

	std::cout << " DONE" << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU...";
	HANDLE_ERROR(cudaMemcpy(finalResult.data, d_finalResultData, finalResult.size, cudaMemcpyDeviceToHost));
	std::cout << " DONE" << std::endl;

	finalResult.write("imgs_out/imageFHD_final.png");
	

	// Free allocated memory on the device and host
	cudaFree(d_inputImgData);
	cudaFree(d_grayImgData);
	cudaFree(d_hSobelMask);
	cudaFree(d_hGradientImgData);
	cudaFree(d_vSobelMask);
	cudaFree(d_vGradientImgData);
	cudaFree(d_finalResultData);

	return 0;
}