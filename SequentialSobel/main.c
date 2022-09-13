#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "fileUtils.h"
#include "imageUtils.h"
#include "math.h"
#include "string.h"
#include <windows.h>
#include <inttypes.h>


typedef unsigned char byte;

#define STRING_BUFFER_SIZE 1024

int main(int argc, char** argv)
{

	//performance timers
	LARGE_INTEGER StartingTime, EndingTime, computation_time_load_img, io_time_load_img, computation_img_processing,
	io_write_gray_img, io_png_conversion;
	LARGE_INTEGER Frequency;

	if(!QueryPerformanceFrequency(&Frequency))
	    printf("QueryPerformanceFrequency failed!\n");

	QueryPerformanceCounter(&StartingTime);

	if (argc < 2)
	{
		printf("No input image was found\n");
		return -1;
	}

	// STEP 1 - carico l'immagine (altezza,larghezza) e poi la converto in RGB


	char * file_output_RGB = "imgs_out/image.rgb";
	char * png_strings[4] = { "convert ", argv[1], " ", file_output_RGB };
	char * str_PNG_to_RGB = array_strings_to_string(png_strings, 4,	STRING_BUFFER_SIZE);

	printf("Input image loaded \n");

	QueryPerformanceCounter(&EndingTime);
	computation_time_load_img.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	// scala e converte in us
	computation_time_load_img.QuadPart *= 1000000;
	computation_time_load_img.QuadPart /= Frequency.QuadPart;

	// converto l'immagine caricata in RGB
	QueryPerformanceCounter(&StartingTime);
	int status_conversion = system(str_PNG_to_RGB);
	QueryPerformanceCounter(&EndingTime);
	io_time_load_img.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	io_time_load_img.QuadPart *= 1000000;
	io_time_load_img.QuadPart /= Frequency.QuadPart;

	QueryPerformanceCounter(&StartingTime);
	if (status_conversion != 0)
	{
		printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
		return -1;
	}

	int width = 0;
	int height = 0;

	// ottengo altezza e larghezza immagine
	get_image_size(argv[1], &width, &height);

	printf("Size of the loaded image: width=%d height=%d \n", width, height);

	// la size sarà moltiplicata per 3 perchè ho una RGB
	int rgb_size = width * height * 3;

	// mi serve un buffer per tutti i pixel dell'immagine
	byte * rgb_image;

	// carica l'immagine input in RGB in un singolo array a 1 dimensione
	read_file(file_output_RGB, &rgb_image, rgb_size);

	// STEP 2 - convertire l'immagine RGB in scala di grigi

	// converto altezza e larghezza in char
	char str_width[100];
	sprintf(str_width, "%d", width);

	char str_height[100];
	sprintf(str_height, "%d", height);

	// buffer per l'immagine in scala di grigi
	byte * grayImage;

	// converto il vettore RGB in Grey-scale
	int gray_size = rgb_to_gray(rgb_image, &grayImage, rgb_size);

	// se INTERMEDIATE_OUTPUT é true allora salvo l'immagine in PNG
	output_gray_scale_image(grayImage, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/img_gray.png");

	// STEP 3 - calcolo del gradiente verticale e orizzontale

	byte* sobel_h_res;
	byte* sobel_v_res;

	// kernel asse orizzontale
	int sobel_h[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	itConv(grayImage, gray_size, width, sobel_h, &sobel_h_res);

	// output del gradiente orizzontale in PNG se INTERMEDIATE_OUTPUT == true
	output_gradient(sobel_h_res, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_horiz_grad.png");

	// kernel per l'asse verticale
	int sobel_v[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	itConv(grayImage, gray_size, width, sobel_v, &sobel_v_res);

	//output del gradiente orizzontale in PNG se if INTERMEDIATE_OUTPUT == true
	output_gradient(sobel_v_res, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_vert_grad.png");

	// STEP 4 - combina i gradienti per ottenere il contorno


	byte* countour_img;
	contour(sobel_h_res, sobel_v_res, gray_size, &countour_img);
	QueryPerformanceCounter(&EndingTime);
	computation_img_processing.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	computation_img_processing.QuadPart *= 1000000;
	computation_img_processing.QuadPart /= Frequency.QuadPart;

	QueryPerformanceCounter(&StartingTime);
	write_file("imgs_out/sobel_countour.gray", countour_img, gray_size);
	QueryPerformanceCounter(&EndingTime);
	io_write_gray_img.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	io_write_gray_img.QuadPart *= 1000000;
	io_write_gray_img.QuadPart /= Frequency.QuadPart;

	// output del contorno finale
	QueryPerformanceCounter(&StartingTime);
	output_gradient(countour_img, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_countour.png");
	QueryPerformanceCounter(&EndingTime);
	io_png_conversion.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	io_png_conversion.QuadPart *= 1000000;
	io_png_conversion.QuadPart /= Frequency.QuadPart;

	// Step 5 - Mostra i risultati
	printf("\n");
	// tempi I/O
	printf("Time spent on I/O operations from/to disk:\n");
	printf("Time to load the image: %"PRId64" us\n", io_time_load_img);
	printf("Time to write gray image: %"PRId64" us\n", io_write_gray_img);
	printf("Time to convert to png image: %"PRId64" us\n", io_png_conversion);
	printf("\n");


	// tempi di computazione
	printf("Time spent on computation:\n");
	printf("Time spent in computation to load image: %"PRId64" us\n", computation_time_load_img);
	printf("Time spent in processing the image: %"PRId64" us\n", computation_img_processing);

	// deallocare la memoria
	free(grayImage);
	free(sobel_h_res);
	free(sobel_v_res);
	free(countour_img);
	return 0;
}
