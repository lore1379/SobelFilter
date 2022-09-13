#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



typedef unsigned char byte;


// in input ho il nome del file su cui caricare i dati
// il buffer per i pixel
// la size del buffer, quindi quanti elementi RGB da caricare
// in output avrò il buffer con i pixels RGB dell'immagine caricata
// si può leggere come
// buffer[0] = R-pixel
// buffer[1] = G-pixel
// buffer[2] = B-pixel
// buffer += 3 per ottenere il pixel successivo

void read_file(const char* file_name, byte** buffer, int buffer_size)
{
	// apre il file in modalità lettura binaria
	FILE* file = fopen(file_name, "rb");

	// Alloca la memoria per il buffer che contiene il file
	*buffer = (byte*)malloc(sizeof(byte) * buffer_size);

	// Legge ogni char del file uno alla volta
	for (int i = 0; i < buffer_size; i++)
	{
		(*buffer)[i] = fgetc(file);
	}

	// chiude il file
	fclose(file);
}

// in input ho il nome del file dove il contenuto del buffer dev'essere scritto
// il buffer che contiene i pixel da scrivere sul file
// la size del buffer da scrivere

void write_file(const char* file_name, byte* buffer, int buffer_size) //was * buffer
{
	FILE* file = fopen(file_name, "wb");

	for (int i = 0; i < buffer_size; i++) {
		fputc(buffer[i], file);
	}

	fclose(file);
}


// codice: http://www.cplusplus.com/forum/beginner/45217
// in input ho il nome del file che devo caricare
// in output ho 0 se ottengo la size con successo in *x e *y

int get_image_size(const char* fn, int* x, int* y)
{
	FILE* f = fopen(fn, "rb");
	if (f == 0) return -1;
	fseek(f, 0, SEEK_END);
	long len = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (len < 24) {
		fclose(f);
		return -1;
	}

	unsigned char buf[24]; fread(buf, 1, 24, f);

	if (buf[0] == 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF && buf[3] == 0xE0 && buf[6] == 'J' && buf[7] == 'F' && buf[8] == 'I' && buf[9] == 'F')
	{
		long pos = 2;
		while (buf[2] == 0xFF)
		{
			if (buf[3] == 0xC0 || buf[3] == 0xC1 || buf[3] == 0xC2 || buf[3] == 0xC3 || buf[3] == 0xC9 || buf[3] == 0xCA || buf[3] == 0xCB) break;
			pos += 2 + (buf[4] << 8) + buf[5];
			if (pos + 12 > len) break;
			fseek(f, pos, SEEK_SET); fread(buf + 2, 1, 12, f);
		}
	}

	fclose(f);

	if (buf[0] == 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF)
	{
		*y = (buf[7] << 8) + buf[8];
		*x = (buf[9] << 8) + buf[10];
		return 0;
	}

	if (buf[0] == 'G' && buf[1] == 'I' && buf[2] == 'F')
	{
		*x = buf[6] + (buf[7] << 8);
		*y = buf[8] + (buf[9] << 8);
		return 0;
	}

	if (buf[0] == 0x89 && buf[1] == 'P' && buf[2] == 'N' && buf[3] == 'G' && buf[4] == 0x0D && buf[5] == 0x0A && buf[6] == 0x1A && buf[7] == 0x0A
		&& buf[12] == 'I' && buf[13] == 'H' && buf[14] == 'D' && buf[15] == 'R')
	{
		*x = (buf[16] << 24) + (buf[17] << 16) + (buf[18] << 8) + (buf[19] << 0);
		*y = (buf[20] << 24) + (buf[21] << 16) + (buf[22] << 8) + (buf[23] << 0);
		return 0;
	}

	return -1;
}

// in input ho un array di stringhe, quante stringhe sono presenti nell'array
// e la grandezza del buffer di char* da creare (STRING_BUFFER_SIZE)
// l'output è una stringa che contiene tutte le stringhe in input
// concatenate
char* array_strings_to_string(const char** strings, int stringsAmount, int buffer_size)
{
	char* strConvert = (char*)malloc(buffer_size);

	// copio il primo elemento
	strcpy(strConvert, strings[0]);

	for (int i = 1; i < stringsAmount; i++)
	{
		// append dei successivi
		strcat(strConvert, strings[i]);
	}
	return strConvert;
}

// in input ho la dimensione, 0 - R, 1 - G, 2 - B
// in ouput la dimensione del vettore, estratto dal vettore RGB e corrispondente alla dimensione specificata
void get_dimension_from_RGB_vec(int dimension, byte* rgbImage, byte** dim_vector, int gray_size)
{
	// prende la size dell'immagine grigia (1D) e alloca la memoria
	*dim_vector = (byte*)malloc(sizeof(byte) * gray_size);

	// Creo i puntatori per le iterazioni
	byte* p_rgb = rgbImage;
	byte* p_gray = *dim_vector;

	// calcolo il valore per ogni pixel in grigio
	for (int i = 0; i < gray_size; i++)
	{
		*p_gray = p_rgb[dimension];
		p_rgb += 3;
		p_gray++;
	}
}


// FUNZIONI DI OUTPUT SU DISCO



// in input abbiamo 
// gray_image: buffer per l'immagine in scala di grigi, sarà convertita in png
// gray_size: la grandezza di gray_image
// str_width: la larghezza dell'immagine di output in formato stringa
// str_height: l'altezza dell'immagine di output in formato stringa
// str_buffer_size: i bytes da allocare per produrre la stringa (per la conversione in PNG)
// png_file_name: l'immagine png in output
void output_gray_scale_image(byte* gray_image, int gray_size, char* str_width, char* str_height, int string_buffer_size, const char* png_file_name)
{

		const char* file_gray = "imgs_out/img_gray.gray";
		write_file(file_gray, gray_image, gray_size);

		const char* PNG_convert_to_gray[8] = { "convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, " ", png_file_name };
		const char* str_gray_to_PNG = array_strings_to_string(PNG_convert_to_gray, 8, string_buffer_size);
		system(str_gray_to_PNG);

}

// produce l'output sia per il gradiente orizzontale
// che verticale

void output_gradient(byte* sobel_res, int gray_size, char* str_width, char* str_height, int string_buffer_size, const char* png_file_name)
{
		// output del gradiente
		const char* file_out_grad = "imgs_out/sobel_grad.gray";
		write_file(file_out_grad, sobel_res, gray_size);
		// converte il file in PNG
		const char* png_convert[8] = { "convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_grad, " ", png_file_name };
		const char* str_grad_to_PNG = array_strings_to_string(png_convert, 8, string_buffer_size);
		system(str_grad_to_PNG);
}