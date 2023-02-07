#include <cstdio>
#include <stdint.h>


enum ImageType {
	PNG, JPG, BMP, TGA
};

struct Image {
	uint8_t* data = NULL;
	size_t size = 0;
	int w;
	int h;
	int channels;

	// constructor, it takes only file as input
	Image(const char* filename);
	// constructor to create a blank image
	Image(int w, int h, int channels);
	// copy constructor, it takes the image to copy
	Image(const Image& img);
	// distructor
	~Image();

	bool read(const char* filename);
	bool write(const char* filename);

	ImageType getFileType(const char* filename);

};