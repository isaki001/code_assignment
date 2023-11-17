* create build directory: mkdir build
* place vancouver.data inside build directory
* cd build
* cmake ..
* make
./mipmap

You must have a c++17 compatible compiler
I used gcc 8.5, nvcc 11.6, cmake 3.20.2,

"make" should create an executable called mipmap.
I have placed the filename in string. Must change in code to use with other files.


# Assumptions: I have not set configurable bounds, routine assumes square image and width and height being multiples of 32. 
The code would never stay that way, but i focused on trying to find a better access pattern.
I didn't get a new pattern to work better, so I have included a straightforward implementation with not a good access pattern that i was using as baseline.
I was experimenting but I didn't beat my baseline implementation. Given more time, i think that i could. 

# Approach

* each thread computes a 2x2 pixel average and stores the results on the downscaled output image
* launch 2D blocks on a 2D grid
* the blocks are 32x32 
* the number of blocks is such that the whole image is covered
* this was the naive/baseline implementation to compare against
* I attempted other versions wiht shared memory caching but I run out of time
* current naive implementation has good memory throughput but uncoalesced accesses
* i tried coalesced flobal load and storing to shared memory and then having each thread compute the average from the shared memory data.
* unfortunately, that approach was slower than the naive implementation 
* on v100, naive impl. takes 1786.35 ms after having read the file. The largest kernel takes 345.25 micro-seconds



# NVPROF-output Singularity> nvprof --print-gpu-trace ./mipmap


read file:201326592
image read
==4007528== NVPROF is profiling process 4007528, command: ./mipmap
iterative mipmap:1320.48 ms
==4007528== Profiling application: ./mipmap
==4007528== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
437.30ms  24.438ms                    -               -         -         -         -  192.00MB  7.6725GB/s    Pageable      Device  Tesla V100-SXM2         1         7  [CUDA memcpy HtoD]
462.06ms  345.25us          (128 128 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [114]
476.88ms  5.8912ms                    -               -         -         -         -  48.000MB  7.9568GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
982.40ms  98.304us            (64 64 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [119]
988.00ms  2.2987ms                    -               -         -         -         -  12.000MB  5.0980GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.12617s  31.200us            (32 32 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [124]
1.12696s  378.97us                    -               -         -         -         -  3.0000MB  7.7306GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.16994s  12.864us            (16 16 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [129]
1.17016s  61.056us                    -               -         -         -         -  768.00KB  11.996GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.18760s  4.6720us              (8 8 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [134]
1.18768s  16.160us                    -               -         -         -         -  192.00KB  11.331GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.19297s  4.2240us              (4 4 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [139]
1.19301s  5.0560us                    -               -         -         -         -  48.000KB  9.0539GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.19619s  3.9360us              (2 2 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [144]
1.19623s  1.9840us                    -               -         -         -         -  12.000KB  5.7682GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.19870s  3.9040us              (1 1 1)       (32 32 1)        26        0B        0B         -           -           -           -  Tesla V100-SXM2         1         7  even_distribution_improved(char3*, char3*, unsigned long, unsigned long) [149]
1.19874s  1.6640us                    -               -         -         -         -  3.0000KB  1.7194GB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.20131s  1.5360us                    -               -         -         -         -      768B  476.84MB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.20345s  1.5040us                    -               -         -         -         -      192B  121.75MB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.20587s  1.5050us                    -               -         -         -         -       48B  30.416MB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.20797s  1.5040us                    -               -         -         -         -       12B  7.6091MB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]
1.21027s  1.5360us                    -               -         -         -         -        3B  1.8626MB/s      Device    Pageable  Tesla V100-SXM2         1         7  [CUDA memcpy DtoH]



It took some time to get the naive impl. working and to figure out how to read the file in binary format.
I experimented but did not get to a satisfactory implementation.
I attempted another approach that pipes the output image to shared memory and uses it as input for another downscaling in the same kernel. I wanted to load to shared memory and re-use as much as possible. i didn't finish this and the temporary results seemed slower than the naive impl.

