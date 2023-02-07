# Sequential and CUDA parallel implementations of the Sobel Filter.

Final assignment for the Parallel Computing course. It compares a sequential approach against three parallel ones implemented with CUDA to observe the benefit of using parallelism in image processing, in particular in the application of a Sobel Filter. The Sobel Filter, also known as Sobel Operator, is an algorithm used in computer vision and image processing for identifying regions characterised by sharp changes in intensity (i.e., edges). For further information, please refer to [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator).

Detailed explaination of the project can be found [here](Report.pdf).

The present implementations work with PNG, JPG and BMP input images which can be stored in imgs_in folder. Results will be found in imgs_out folder.

In order to run the CUDA version, make sure you have a CUDA-compatible GPU and its corresponding CUDA drivers, as well as the nvidia-cuda-toolkit to compile CUDA programs.
