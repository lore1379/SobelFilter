#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include "cmath"

#include "image.h"

#define SOBEL_OP_DIM 3
#define SOBEL_OP_RADIUS (SOBEL_OP_DIM / 2)

// tile dimension
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

// Allocate masks in constant memory
__constant__ int constSobelH[SOBEL_OP_DIM * SOBEL_OP_DIM];
__constant__ int constSobelV[SOBEL_OP_DIM * SOBEL_OP_DIM];

// Shared Memory Elements needed to be loaded as per Mask Size
#define SharedDim_x (TILE_WIDTH + SOBEL_OP_DIM - 1)
#define SharedDim_y (TILE_HEIGHT + SOBEL_OP_DIM - 1)

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

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int Idx = y * blockDim.x * gridDim.x + x;
	
	if (x < w && y < h)
		resultData[Idx] = (uint8_t)sqrt(pow(hGradientImgData[Idx], 2) + pow(vGradientImgData[Idx], 2));
}

__global__ void gradientConvolutionH(uint8_t* inputImgData, uint8_t* gradientImgData, int width, int height) {

	__shared__ uint8_t N_ds[SharedDim_y][SharedDim_x];

	// First batch loading
	int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
		destY = dest / SharedDim_x, destX = dest % SharedDim_x,
		srcY = blockIdx.y * TILE_HEIGHT + destY - SOBEL_OP_RADIUS,
		srcX = blockIdx.x * TILE_WIDTH + destX - SOBEL_OP_RADIUS,
		src = (srcY * width + srcX);
	if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
		N_ds[destY][destX] = inputImgData[src];
	else
		N_ds[destY][destX] = 0;

	// Second batch loading
	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + (TILE_WIDTH * TILE_HEIGHT);
	destY = dest / SharedDim_x, destX = dest % SharedDim_x;
	srcY = blockIdx.y * TILE_HEIGHT + destY - SOBEL_OP_RADIUS;
	srcX = blockIdx.x * TILE_WIDTH + destX - SOBEL_OP_RADIUS;
	src = (srcY * width + srcX);
	if (destY < SharedDim_y && destX < SharedDim_x)
	{
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = inputImgData[src];
		else
			N_ds[destY][destX] = 0;
	}
	__syncthreads();

	int temp = 0;
	int y, x;
	for (y = 0; y < SOBEL_OP_DIM; y++)
		for (x = 0; x < SOBEL_OP_DIM; x++)
			temp += N_ds[threadIdx.y + y][threadIdx.x + x] * constSobelH[y * SOBEL_OP_DIM + x];
	y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
	x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	if (y < height && x < width){
		gradientImgData[y * width + x] = (uint8_t)abs(temp);
	}
}

__global__ void gradientConvolutionV(uint8_t* inputImgData, uint8_t* gradientImgData, int width, int height) {

	__shared__ uint8_t N_ds[SharedDim_y][SharedDim_x];

	// First batch loading
	int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
		destY = dest / SharedDim_x, destX = dest % SharedDim_x,
		srcY = blockIdx.y * TILE_HEIGHT + destY - SOBEL_OP_RADIUS,
		srcX = blockIdx.x * TILE_WIDTH + destX - SOBEL_OP_RADIUS,
		src = (srcY * width + srcX);
	if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
		N_ds[destY][destX] = inputImgData[src];
	else
		N_ds[destY][destX] = 0;

	// Second batch loading
	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + (TILE_WIDTH * TILE_HEIGHT);
	destY = dest / SharedDim_x, destX = dest % SharedDim_x;
	srcY = blockIdx.y * TILE_HEIGHT + destY - SOBEL_OP_RADIUS;
	srcX = blockIdx.x * TILE_WIDTH + destX - SOBEL_OP_RADIUS;
	src = (srcY * width + srcX);
	if (destY < SharedDim_y && destX < SharedDim_x)
	{
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = inputImgData[src];
		else
			N_ds[destY][destX] = 0;
	}
	__syncthreads();

	int temp = 0;
	int y, x;
	for (y = 0; y < SOBEL_OP_DIM; y++)
		for (x = 0; x < SOBEL_OP_DIM; x++)
			temp += N_ds[threadIdx.y + y][threadIdx.x + x] * constSobelV[y * SOBEL_OP_DIM + x];
	y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
	x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	if (y < height && x < width) {
		gradientImgData[y * width + x] = (uint8_t)abs(temp);
	}
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

	// timers 
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::duration<double, std::milli> duration;

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

	dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
	dim3 dimGrid((inputImg.w + TILE_WIDTH - 1) / TILE_WIDTH, (inputImg.h + TILE_HEIGHT - 1) / TILE_HEIGHT);

	ConvertImageToGrayGpu << <dimGrid, dimBlock >> > (d_inputImgData, d_grayImgData, inputImg.channels, inputImg.w, inputImg.h);
	
	std::cout << " DONE" << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU...";
	HANDLE_ERROR(cudaMemcpy(grayImg.data, d_grayImgData, grayImg.size, cudaMemcpyDeviceToHost));
	std::cout << " DONE" << std::endl;

	// grayImg.write("imgs_out/imageFHD_gray.png");

	// STEP 2 ##### calcolo del gradiente orizzontale e verticale

	// horizontal mask on host
	int sobelH[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	// horizontal gradient image on host
	Image hGradientImg(grayImg.w, grayImg.h, 1);

	// Copy data to the gpu
	std::cout << "Copy data to GPU...";

	// allocate in device memory space for result
	uint8_t* d_hGradientImgData;
	HANDLE_ERROR(cudaMalloc(&d_hGradientImgData, hGradientImg.size));

	// copy mask on device
	HANDLE_ERROR(cudaMemcpyToSymbol(constSobelH, sobelH, SOBEL_OP_DIM * SOBEL_OP_DIM * sizeof(int)));
	std::cout << " DONE" << std::endl;

	// Process image on gpu
	std::cout << "Running CUDA Kernel, computing horizontal gradient...";

	gradientConvolutionH << < dimGrid, dimBlock >> > (d_grayImgData, d_hGradientImgData, grayImg.w, grayImg.h);	
	std::cout << " DONE" << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU...";
	HANDLE_ERROR(cudaMemcpy(hGradientImg.data, d_hGradientImgData, hGradientImg.size, cudaMemcpyDeviceToHost));
	std::cout << " DONE" << std::endl;

	// hGradientImg.write("imgs_out/imageFHD_sobel_h.png");

	// vertical mask on host
	int sobelV[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	// vertical gradient image on host
	Image vGradientImg(grayImg.w, grayImg.h, 1);

	// Copy data to the gpu
	std::cout << "Copy data to GPU...";

	uint8_t* d_vGradientImgData;
	HANDLE_ERROR(cudaMalloc(&d_vGradientImgData, vGradientImg.size));

	// copy mask on device
	HANDLE_ERROR(cudaMemcpyToSymbol(constSobelV, sobelV, SOBEL_OP_DIM* SOBEL_OP_DIM * sizeof(int)));
	std::cout << " DONE" << std::endl;

	// Process image on gpu
	std::cout << "Running CUDA Kernel, computing horizontal gradient...";

	gradientConvolutionV << < dimGrid, dimBlock >> > (d_grayImgData, d_vGradientImgData, grayImg.w, grayImg.h);
	
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
	cudaFree(d_hGradientImgData);
	cudaFree(d_vGradientImgData);
	cudaFree(d_finalResultData);

	cudaDeviceReset();
	return 0;
}