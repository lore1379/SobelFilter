#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include "image.h"
#include "cmath"


Image::Image(const char* filename) {
	if (read(filename)) {
		printf("Read %s\n", filename);
		size = w * h * channels;
	}
	else {
		printf("Failed to read %s\n", filename);
	}
}

Image::Image(int w, int h, int channels) : w(w), h(h), channels(channels) {
	size = w * h * channels;
	data = new uint8_t[size];
}

Image::Image(const Image& img) : Image(img.w, img.h, img.channels) {
	memcpy(data, img.data, size);
}

Image::~Image() {
	stbi_image_free(data);
}

bool Image::read(const char* filename) {
	data = stbi_load(filename, &w, &h, &channels, 0);
	return data != NULL;
}

bool Image::write(const char* filename) {
	ImageType type = getFileType(filename);
	int success;
	switch (type) {
	case PNG:
		success = stbi_write_png(filename, w, h, channels, data, w * channels);
		break;
	case BMP:
		success = stbi_write_bmp(filename, w, h, channels, data);
		break;
	case JPG:
		success = stbi_write_jpg(filename, w, h, channels, data, 100);
		break;
	}

	if (success != 0) {
		printf("Wrote %s, %d, %d, %d, %zu\n", filename, w, h, channels, size);
		return true;
	}
	else {
		printf("Failed to write %s, %d, %d, %d, %zu\n", filename, w, h, channels, size);
		return false;
	}
}

ImageType Image::getFileType(const char* filename) {
	const char* ext = strrchr(filename, '.');
	if (ext != nullptr) {
		if (strcmp(ext, ".png") == 0) {
			return PNG;
		}
		else if (strcmp(ext, ".jpg") == 0) {
			return JPG;
		}
		else if (strcmp(ext, ".bmp") == 0) {
			return PNG;
		}

	}

	return PNG;
}

Image& Image::grayscale() {
	if (channels < 3) {
		printf("Image %p has less than 3 channels, it is assumed to already be grayscale.", this);
	}
	else {
		for (int i = 0; i < size; i += channels) {
			int gray = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
			memset(data + i, gray, 3);
		}
	}
	return *this;
}

void CPUconvertImageToGray(Image& inputImg, Image& grayImg) {

	inputImg.grayscale();

	int imgSize = inputImg.w * inputImg.h;

	for (int k = 0; k < imgSize; k++) {
		grayImg.data[k] = inputImg.data[inputImg.channels * k];
	}

}

void cpu2DConvolution(Image& grayImg, Image& hGradientImg, double* mask, int w, int h, int maskDim) {
	// Temp value for accumulating results
	double temp;
	int radius = maskDim / 2;


	// Intermediate value for more readable code
	int offset_r;
	int offset_c;

	// Go over each row
	for (int i = 0; i < h; i++) {
		// Go over each column
		for (int j = 0; j < w; j++) {
			// Reset the temp variable
			temp = 0;

			// Go over each mask row
			for (int k = 0; k < maskDim; k++) {
				// Update offset value for row
				offset_r = i - radius + k;

				// Go over each mask column
				for (int l = 0; l < maskDim; l++) {
					// Update offset value for column
					offset_c = j - radius + l;

					// Range checks if we are hanging off the matrix
					if (offset_r >= 0 && offset_r < h) {
						if (offset_c >= 0 && offset_c < w) {
							// Accumulate partial results
							temp += grayImg.data[offset_r * w + offset_c] * mask[k * maskDim + l];
						}
					}
				}
			}
			hGradientImg.data[i * w + j] = (uint8_t)abs(temp);
		}
	}
	
}

void sobelFilterCPU(Image& hGradient, Image& vGradient, int size, Image& result) {
	for (int i = 0; i < size; i++)
	{
		result.data[i] = (uint8_t)sqrt(pow(hGradient.data[i], 2) + pow(vGradient.data[i], 2));
	}
}