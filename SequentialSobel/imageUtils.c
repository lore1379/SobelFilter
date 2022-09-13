#include "imageUtils.h"

#include "fileUtils.h"

// gray_image: buffer per l'immagine in scala di grigi, sarà convertita in png
// gray_size: la grandezza di gray_image
// str_width: la larghezza dell'immagine di output in formato stringa
// str_height: l'altezza dell'immagine di output in formato stringa
// str_buffer_size: i bytes da allocare per produrre la stringa (per la conversione in PNG)
// png_file_name: l'immagine png in output

void output_gray_scale_image(byte * gray_image, int gray_size, char * str_width, char * str_height, int string_buffer_size, char * png_file_name)
{
		char * file_gray = "imgs_out/img_gray.gray";
		write_file(file_gray, gray_image, gray_size);

		char * pngConvertGray[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, " ", png_file_name};
		char * strGrayToPNG = array_strings_to_string(pngConvertGray, 8, string_buffer_size);
		system(strGrayToPNG);

}


// produce l'output sia per il gradiente orizzontale
// che verticale

void output_gradient(byte* sobel_res, int gray_size, char* str_width, char* str_height, int string_buffer_size, char* png_file_name)
{

       	// output del gradiente
        char* file_out_grad = "imgs_out/sobel_grad.gray";
        write_file(file_out_grad, sobel_res, gray_size);
        // converte il file in PNG
        char* pngConvert[8] = { "convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_grad, " ", png_file_name };
        char* str_grad_to_PNG = array_strings_to_string(pngConvert, 8, string_buffer_size);
        system(str_grad_to_PNG);
}

// in input ho l'immagine RGB contenuta nel buffer RGB
// il buffer che conterrà l'immagine in scala di grigi
// la grandezza del buffer RGB
// in output avrò l'array che contiene l'immagine in scala di grigi

int rgb_to_gray(byte *rgb, byte **grayImage, int buffer_size)
{
    // La grandezza dell'imagine in scala di grigi è RGB/3
    // si alloca poi la memoria
	int gray_size = buffer_size / 3;
	*grayImage = malloc(sizeof(byte) * gray_size);

    // puntatori per iterare
	byte *p_rgb = rgb;
	byte *p_gray = *grayImage;

    // Calcola il valore per ogni pixel in grigio
	for(int i=0; i < gray_size; i++)
	{
        // Formula: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
		*p_gray = 0.30*p_rgb[0] + 0.59*p_rgb[1] + 0.11*p_rgb[2];
		 p_rgb += 3;
		 p_gray++;
	}

    return gray_size;
}

// in input ho tutti i pixel dell'immagine in scala di grigi
// la grandezza dell'immagine
// la larghezza dell'immagine
// l'indice del pixel che sto considerando
// in output ho la regione di pixel 3x3 intorno all'indice

void make_op_mem(byte* buffer, int buffer_size, int width, int index, byte* op_mem)
{
    // è 1 solo nella prima riga
    int bottom = index - width < 0;
    // è 1 solo nell'ultima riga
    int top = index + width >= buffer_size;
    // indica quando la colonna di sinistra è fuori dall'immagine
    int left = index % width == 0;
    // indica quando la colonna di destra è fuori dall'mimmagine
    int right = (index + 1) % width == 0;

    op_mem[0] = !bottom && !left ? buffer[index - width - 1] : 0;
    op_mem[1] = !bottom ? buffer[index - width] : 0;
    op_mem[2] = !bottom && !right ? buffer[index - width + 1] : 0;

    op_mem[3] = !left ? buffer[index - 1] : 0;
    op_mem[4] = buffer[index];
    op_mem[5] = !right ? buffer[index + 1] : 0;

    op_mem[6] = !top && !left ? buffer[index + width - 1] : 0;
    op_mem[7] = !top ? buffer[index + width] : 0;
    op_mem[8] = !top && !right ? buffer[index + width + 1] : 0;
}


// in input il vettore x, la coordinata y e la grandezza dell'area
// su cui fare la convoluzione
// in output il risultato della convoluzione
int convolution(byte* X, int* Y, int c_size)
{
    int sum = 0;

    for (int i = 0; i < c_size; i++) {
        sum += X[i] * Y[c_size - i - 1];
    }

    return sum;
}

// in input ho l'immagine in scala di grigi
// la grandezza dell'immagine
// la larghezza dell'immagine
// il kernel 3x3 per fare la convoluzione sull'immagine
// infine l'output, il gradiente risultate

void itConv(byte* buffer, int buffer_size, int width, int* op, byte** res)
{
    // alloco la memoria per il risultato
    *res = malloc(sizeof(byte) * buffer_size);

    // Memoria temporanea per ogni operazione su pixel
    // sobel op size è sempre 9
    byte op_mem[SOBEL_OP_SIZE];
    // copia il char op_mem (byte è un unsigned char) nei primi 9 charactes della stringa puntata dall'argomento op_mem.
    // essenzialmente mette 9 zeri in op_mem, inizializzandolo.
    memset(op_mem, 0, SOBEL_OP_SIZE);

    // fa la convoluzione su ogni pixel
    for (int i = 0; i < buffer_size; i++)
    {
        // esegue le operazione con il kernel
        make_op_mem(buffer, buffer_size, width, i, op_mem);

        // Convoluzione
        // la funzione abs è utilizzata per evitare in numeri negativi nell'array
        // questo non è un problema perchè tanto quando lo riusiamo il valore
        // è elevato al quadrato
        (*res)[i] = (byte)abs(convolution(op_mem, op, SOBEL_OP_SIZE));
    }
}

// In input ho l'array che contiene il gradiente orizzontale
// l'array che contiene il gradiente verticale
// il numero di pixel dell'immagine
// l'output è l'immagine finale dei contorni combinando i due gradienti

void contour(byte* sobel_h, byte* sobel_v, int gray_size, byte** contour_img)
{
    // alloco la memoria per contour_img
    *contour_img = malloc(sizeof(byte) * gray_size);

    // itero su ogni pixel per calcolare l'immagine contorno
    for (int i = 0; i < gray_size; i++)
    {
        (*contour_img)[i] = (byte)sqrt(pow(sobel_h[i], 2) + pow(sobel_v[i], 2));
    }
}
