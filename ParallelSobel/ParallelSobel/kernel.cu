
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

typedef unsigned char byte;

#define SOBEL_OP_SIZE 9


// In input ho l'array che contiene il gradiente orizzontale
// l'array che contiene il gradiente verticale
// il numero di pixel dell'immagine
// l'output è l'immagine finale dei contorni combinando i due gradienti

__global__ void contour(byte* dev_sobel_h, byte* dev_sobel_v, int gray_size, byte* dev_contour_img) 
{
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    // uso abs come strategia di linearizzazione per evitare di avere threads con numeri negativi
    int tid = abs(tid_x - tid_y);


    // Il calcolo dell'immagine è fatto su ogni pixel in parallelo
    while (tid < gray_size)
    {
        //g = (g_x^2 + g_y^2)^0.5
        dev_contour_img[tid] = (byte)sqrt(pow((double)dev_sobel_h[tid], 2.0) + pow((double)dev_sobel_v[tid], 2.0));

        tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;

    }
}



// Convoluzione su una regione in input
__device__ int convolution(byte* X, int* Y, int c_size)
{
    int sum = 0;

    for (int i = 0; i < c_size; i++)
    {
        sum += X[i] * Y[c_size - i - 1];
    }

    return sum;
}

// in input ho tutti i pixel dell'immagine in scala di grigi
// la grandezza dell'immagine
// la larghezza dell'immagine
// l'indice del pixel che sto considerando
// in output ho la regione di pixel 3x3 intorno all'indice

__device__ void make_op_mem(byte* dev_buffer, int buffer_size, int width, int cindex, byte* op_mem)
{
    int bottom = cindex - width < 0;
    int top = cindex + width >= buffer_size;
    int left = cindex % width == 0;
    int right = (cindex + 1) % width == 0;

    op_mem[0] = !bottom && !left ? dev_buffer[cindex - width - 1] : 0;
    op_mem[1] = !bottom ? dev_buffer[cindex - width] : 0;
    op_mem[2] = !bottom && !right ? dev_buffer[cindex - width + 1] : 0;

    op_mem[3] = !left ? dev_buffer[cindex - 1] : 0;
    op_mem[4] = dev_buffer[cindex];
    op_mem[5] = !right ? dev_buffer[cindex + 1] : 0;

    op_mem[6] = !top && !left ? dev_buffer[cindex + width - 1] : 0;
    op_mem[7] = !top ? dev_buffer[cindex + width] : 0;
    op_mem[8] = !top && !right ? dev_buffer[cindex + width + 1] : 0;
}


// in input ho l'immagine in scala di grigi
// la grandezza dell'immagine
// la larghezza dell'immagine
// il kernel 3x3 per fare la convoluzione sull'immagine
// infine l'output, il gradiente risultate

__global__ void it_conv(byte* dev_buffer, int buffer_size, int width, int* dev_op, byte* dev_res)
{
    // Memoria temporanea per ogni operazione su pixel
    byte op_mem[SOBEL_OP_SIZE];
    memset(op_mem, 0, SOBEL_OP_SIZE);
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    // linearizzazione semplice
    int tid = abs(tid_x - tid_y);

    // Esegue una convoluzione per ogni pixel, un thread per pixel
    while (tid < buffer_size)
    {
        // la regione nell'immagine grigia dove effettuare la convoluzione
        make_op_mem(dev_buffer, buffer_size, width, tid, op_mem);

        // effettua la convoluzione
        dev_res[tid] = (byte)abs(convolution(op_mem, dev_op, SOBEL_OP_SIZE));
        
        tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;
    }
}


// in input ho i vettori che contengono le componenti RGB dell'immagine in input
// il numero di pixel nel vettore RGB / 3
// in output avrò l'array che contiene l'immagine in scala di grigi

// CUDA kernel l'immagine in scala di grigi
// l'immagine dev'essere preallocata in memoria
__global__ void rgb_img_to_gray(byte* dev_r_vec, byte* dev_g_vec, byte* dev_b_vec, byte* dev_gray_image, int gray_size)
{
    // Ottengo l'id del thread in un blocco
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    // linearizzazione semplice di una spazio a 2D
    int tid = abs(tid_x - tid_y);

    // operazione pixel per pixel per i vettori RGB
    while (tid < gray_size)
    {
        // r, g, b pixels dell'immagine input
        byte p_r = dev_r_vec[tid];
        byte p_g = dev_g_vec[tid];
        byte p_b = dev_b_vec[tid];

        //Formula: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        dev_gray_image[tid] = 0.30 * p_r + 0.59 * p_g + 0.11 * p_b;

        tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;

    }
}
