#include "image.h"
#include <chrono>
#include <iostream>
#include <fstream>

#define SOBEL_OP_DIM 3


int main(int argc, char** argv) {

	// timers 
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::duration<double, std::milli> duration;

	// read input image
	Image inputImg("imgs_in/imageFHD.jpg");

	std::cout << " Converting input image to grayscale ... ";

	// create gray Img with 1 channel
	Image grayImg(inputImg.w, inputImg.h, 1);

	static clock::time_point CPU_grayConversion_start = clock::now();

	CPUconvertImageToGray(inputImg, grayImg);

	duration CPU_grayConversion_elapsed = clock::now() - CPU_grayConversion_start;
	std::cout << " DONE: processing image on CPU took " <<
		CPU_grayConversion_elapsed.count() << " ms\n";

	// grayImg.write("imgs_out/imageFHD_gray.png");

	// horizontal gradient 
	Image hGradientImg(grayImg.w, grayImg.h, 1);

	double sobel_h[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	std::cout << " Computing horizontal gradient ... ";

	static clock::time_point CPU_hGradient_start = clock::now();

	cpu2DConvolution(grayImg,hGradientImg, sobel_h, grayImg.w, grayImg.h, SOBEL_OP_DIM);
 
	duration CPU_hGradient_elapsed = clock::now() - CPU_hGradient_start;
	std::cout << " DONE: processing image on CPU took " <<
		CPU_hGradient_elapsed.count() << " ms\n";

	// hGradientImg.write("imgs_out/imageFHD_sobel_h.png");

	// vertical gradient
	Image vGradientImg(grayImg.w, grayImg.h, 1);

	double sobel_v[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	std::cout << " Computing vertical gradient ... ";

	static clock::time_point CPU_vGradient_start = clock::now();

	cpu2DConvolution(grayImg, vGradientImg, sobel_v, grayImg.w, grayImg.h, SOBEL_OP_DIM);

	duration CPU_vGradient_elapsed = clock::now() - CPU_vGradient_start;
	std::cout << " DONE: processing image on CPU took " <<
		CPU_vGradient_elapsed.count() << " ms\n";

	// vGradientImg.write("imgs_out/imageFHD_sobel_v.png");

	// final result
	Image finalResult(inputImg.w, inputImg.h, 1);

	std::cout << " Combining horizontal and vertical gradient ... ";

	static clock::time_point CPU_finalResult_start = clock::now();

	sobelFilterCPU(hGradientImg, vGradientImg, finalResult.size, finalResult);

	duration CPU_finalResult_elapsed = clock::now() - CPU_finalResult_start;
	std::cout << " DONE: processing image on CPU took " <<
		CPU_finalResult_elapsed.count() << " ms\n";

	finalResult.write("imgs_out/imageFHD_final.png");	

	return 0;
}