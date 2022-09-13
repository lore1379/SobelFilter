typedef unsigned char byte;


void read_file(char* file_name, byte** buffer, int buffer_size);
void write_file(char* file_name, byte* buffer, int buffer_size);
int get_image_size(const char* fn, int* x, int* y);
char* array_strings_to_string(char** strings, int stringsAmount, int buffer_size);
