#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <inttypes.h>

#include "utils.c"
#include "kernel.cu"

#define SOBEL_OP_SIZE 9
#define STRING_BUFFER_SIZE 1024

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char** argv)
{
	//performance timers
	LARGE_INTEGER StartingTime, EndingTime, computation_time_load_img, io_time_load_img, computation_img_processing,
		io_write_gray_img, io_png_conversion;
	LARGE_INTEGER Frequency;

	if (!QueryPerformanceFrequency(&Frequency))
		printf("QueryPerformanceFrequency failed!\n");

	QueryPerformanceCounter(&StartingTime);
	if (argc < 2)
	{
		printf("No input image was found\n");
		return -1;
	}

	// STEP 1 - carico l'immagine (altezza,larghezza) e poi la converto in RGB


	const char* file_output_rgb = "imgs_out/image.rgb";
	const char* png_strings[4] = { "convert ", argv[1], " ", file_output_rgb };
	const char* str_PNG_to_RGB = array_strings_to_string(png_strings, 4, STRING_BUFFER_SIZE);

	printf("Input image loaded \n");

	QueryPerformanceCounter(&EndingTime);
	computation_time_load_img.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	computation_time_load_img.QuadPart *= 1000000;
	computation_time_load_img.QuadPart /= Frequency.QuadPart;

	QueryPerformanceCounter(&StartingTime);
	// converto l'immagine caricata in RGB	
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
	// ottengo altezza e larghezza immagine
	int width = 0;
	int height = 0;

	get_image_size(argv[1], &width, &height);

	printf("Size of the loaded image: width=%d height=%d \n", width, height);

	// la size sarà moltiplicata per 3 perchè ho una RGB
	int rgb_size = width * height * 3;

	// mi serve un buffer per tutti i pixel dell'immagine
	byte* rgb_image;

	// carica l'immagine input in RGB in un singolo array a 1 dimensione
	read_file(file_output_rgb, &rgb_image, rgb_size);


	// STEP 2 - convertire l'immagine RGB in scala di grigi


	int gray_size = rgb_size / 3;
	byte* r_vector, * g_vector, * b_vector;

	// prendo il vettore RGB e creo 3 array separati, uno per dimensione
	get_dimension_from_RGB_vec(0, rgb_image, &r_vector, gray_size);
	get_dimension_from_RGB_vec(1, rgb_image, &g_vector, gray_size);
	get_dimension_from_RGB_vec(2, rgb_image, &b_vector, gray_size);

	// alloco memoria sul device per i vettori R,G,B
	byte* dev_r_vec, * dev_g_vec, * dev_b_vec;
	byte* dev_gray_image;

	HANDLE_ERROR(cudaMalloc((void**)&dev_r_vec, gray_size * sizeof(byte)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_g_vec, gray_size * sizeof(byte)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b_vec, gray_size * sizeof(byte)));

	// copio il contenuto dei vettori dall'host al device
	HANDLE_ERROR(cudaMemcpy(dev_r_vec, r_vector, gray_size * sizeof(byte), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_g_vec, g_vector, gray_size * sizeof(byte), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b_vec, b_vector, gray_size * sizeof(byte), cudaMemcpyHostToDevice));
	
	// alloco la memoria sul device per l'immagine grigia di output
	HANDLE_ERROR(cudaMalloc((void**)&dev_gray_image, gray_size * sizeof(byte)));

	// lancio il kernel per convertire il file RGB in grigio
	rgb_img_to_gray <<< width, height >>> (dev_r_vec, dev_g_vec, dev_b_vec, dev_gray_image, gray_size);
	cudaDeviceSynchronize();

	byte* gray_image = (byte*)malloc(gray_size * sizeof(byte));

	// Copio sull'host il vettore dell'immagine in grigio
	HANDLE_ERROR(cudaMemcpy(gray_image, dev_gray_image, gray_size * sizeof(byte), cudaMemcpyDeviceToHost));
	
	// converto altezza e larghezza in char

	char str_width[100];
	sprintf(str_width, "%d", width);

	char str_height[100];
	sprintf(str_height, "%d", height);

	// se INTERMEDIATE_OUTPUT é true allora salvo l'immagine in PNG
	output_gray_scale_image(gray_image, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/img_gray.png");

	cudaFree(dev_r_vec);
	cudaFree(dev_g_vec);
	cudaFree(dev_b_vec);

	// STEP 3 - calcolo del gradiente verticale e orizzontale

	// kernel orizzontale host
	int sobel_h[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	int* dev_sobel_h;
	byte* dev_sobel_h_res;

	// alloco la memoria sul device
	HANDLE_ERROR(cudaMalloc((void**)&dev_sobel_h, SOBEL_OP_SIZE * sizeof(int)));

	// copio il contenuto del kernel host sul device
	HANDLE_ERROR(cudaMemcpy(dev_sobel_h, sobel_h, SOBEL_OP_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	// alloco la memoria per il risultato sul device
	HANDLE_ERROR(cudaMalloc((void**)&dev_sobel_h_res, gray_size * sizeof(byte)));

	// eseguo il gradiente orizzontale su ogni pixel
	it_conv <<< width, height >>> (dev_gray_image, gray_size, width, dev_sobel_h, dev_sobel_h_res);
	cudaDeviceSynchronize();

	// alloco la memoria per il risultato sull'host
	byte* sobel_h_res = (byte*)malloc(gray_size * sizeof(byte));

	// copio il risultato dal device all'host
	HANDLE_ERROR(cudaMemcpy(sobel_h_res, dev_sobel_h_res, gray_size * sizeof(byte), cudaMemcpyDeviceToHost));

	cudaFree(dev_sobel_h);

	// output del gradiente orizzontale in PNG se INTERMEDIATE_OUTPUT == true
	output_gradient(sobel_h_res, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_horiz_grad.png");

	// kernel verticale host
	int sobel_v[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	int* dev_sobel_v;
	byte* dev_sobel_v_res;

	// alloco la memoria sul device
	HANDLE_ERROR(cudaMalloc((void**)&dev_sobel_v, SOBEL_OP_SIZE * sizeof(int)));

	// copio il contenuto del kernel host sul device
	HANDLE_ERROR(cudaMemcpy(dev_sobel_v, sobel_v, SOBEL_OP_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	// alloco la memoria per il risultato sul device
	HANDLE_ERROR(cudaMalloc((void**)&dev_sobel_v_res, gray_size * sizeof(byte)));

	// eseguo il gradiente verticale su ogni pixel
	it_conv <<<width, height >>> (dev_gray_image, gray_size, width, dev_sobel_v, dev_sobel_v_res);
	cudaDeviceSynchronize();

	// alloco la memoria per il risultato sull'host
	byte* sobel_v_res = (byte*)malloc(gray_size * sizeof(byte));

	// copio il risultato dal device all'host
	HANDLE_ERROR(cudaMemcpy(sobel_v_res, dev_sobel_v_res, gray_size * sizeof(byte), cudaMemcpyDeviceToHost));

	cudaFree(dev_sobel_v);

	// output del gradiente verticale in PNG se INTERMEDIATE_OUTPUT == true
	output_gradient(sobel_v_res, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_vert_grad.png");


	// STEP 4 - combina i gradienti per ottenere il contorno


	byte* dev_countour_img;

	HANDLE_ERROR(cudaMalloc((void**)&dev_countour_img, gray_size * sizeof(byte)));

	contour <<< width, height >>> (dev_sobel_h_res, dev_sobel_v_res, gray_size, dev_countour_img);
	cudaDeviceSynchronize();

	byte* countour_img = (byte*)malloc(gray_size * sizeof(byte));

	HANDLE_ERROR(cudaMemcpy(countour_img, dev_countour_img, gray_size * sizeof(byte), cudaMemcpyDeviceToHost));

	cudaFree(dev_sobel_h_res);
	cudaFree(dev_sobel_v_res);
	cudaFree(dev_countour_img);

	QueryPerformanceCounter(&EndingTime);
	computation_img_processing.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	computation_img_processing.QuadPart *= 1000000;
	computation_img_processing.QuadPart /= Frequency.QuadPart;

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
	printf("Time to load the image: %I64u us\n", io_time_load_img);
	printf("Time to convert to png image: %I64u us\n", io_png_conversion);
	printf("\n");


	// tempi di computazione
	printf("Time spent on computation:\n");
	printf("Time spent in computation to load image: %I64u us\n", computation_time_load_img);
	printf("Time spent in processing the image: %I64u us\n", computation_img_processing);


	free(gray_image);
	free(sobel_h_res);
	free(sobel_v_res);
	free(countour_img);

	return 0;

}