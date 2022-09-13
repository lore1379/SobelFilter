#include "fileUtils.h"

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

void read_file(char* file_name, byte** buffer, int buffer_size)
{
    // apre il file in modalità lettura binaria
    FILE* file = fopen(file_name, "rb");

    // Alloca la memoria per il buffer che contiene il file
    *buffer = malloc(sizeof(byte) * buffer_size);

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

void write_file(char* file_name, byte* buffer, int buffer_size)
{
    // apre il file su cui scrivere
    FILE* file = fopen(file_name, "wb");

    // scrive
    for (int i = 0; i < buffer_size; i++) {
        fputc(buffer[i], file);
    }

    // chiude
    fclose(file);
}

// codice: http://www.cplusplus.com/forum/beginner/45217
// in input ho il nome del file che devo caricare
// in output ho 0 se ottengo la size con successo in *x e *y
int get_image_size(const char* fn, int* x, int* y)
{
    // r = apre in lettura in modalità binario (b)
    FILE* f = fopen(fn, "rb");
    if (f == 0) return -1;
    // muove il puntatore di lettura in una posizione all'interno del file
    // il secondo elemento indica quanti byte deve essere spostato il pointer
    // il terzo inca la posizione di partenza, ovvero la fine
    fseek(f, 0, SEEK_END);
    // restituisce la posizione corrente del pointer rispetto all'inizio
    long len = ftell(f);
    // SEEK_SET indica l'inizio del file
    fseek(f, 0, SEEK_SET);
    if (len < 24) {
        fclose(f);
        return -1;
    }

    // Strategia:
    // leggere la dimensione di una GIF richiede i primi 10 bytes del file
    // leggere la dimensione di un PNG richiede i primi 24 bytes del file
    // leggere la dimensione di un JPEG richiede lo scan dei chunks JPEG
    // In tutti i formati comunque il file è grande almeno 24 bytes
    // quindi li leggiamo sempre

    // fread legge i dati del file binario, per leggere a blocchi
    // dobbiamo aprire il file in modalità binaria
    // punt è un vettore, quindi le info lette sono trasferite al suo interno
    // il secondo elemento indica la dimensione in byte del singolo dato
    // il terzo elemento il numero di elementi da leggere
    // f è il file aperto con fopen
    unsigned char buf[24]; fread(buf, 1, 24, f);

    // Per il JPEG dobbiamo leggere il primi 12 bytes di ogni chunk
    // leggiamo quindi i 12 bytes a buf+2 ... buf+13, sovrascrivendo il buffer
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

    // JPEG: i primi 2 bytes di buf sono i primi 2 bytes del file jpeg
    // il resto è DCT frame
    if (buf[0] == 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF)
    {
        *y = (buf[7] << 8) + buf[8];
        *x = (buf[9] << 8) + buf[10];
        //cout << *x << endl;
        return 0;
    }

    // GIF: i primi 3 bytes dicono "GIF", i 3 successivi il numero di versione
    // poi c'è la dimensione
    if (buf[0] == 'G' && buf[1] == 'I' && buf[2] == 'F')
    {
        *x = buf[6] + (buf[7] << 8);
        *y = buf[8] + (buf[9] << 8);
        return 0;
    }

    // PNG: il primo frame è per definizione un IHDR frame,
    // che da le dimensioni.
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
char* array_strings_to_string(char** strings, int stringsAmount, int buffer_size)
{
    char* strConvert = malloc(buffer_size);

    // copio il primo elemento
    strcpy(strConvert, strings[0]);

    for (int i = 1; i < stringsAmount; i++)
    {
        // faccio append dei successivi
        strcat(strConvert, strings[i]);
    }

    // l'output è una stringa che contiene tutte le stringhe in input
    // concatenate
    return strConvert;
}
