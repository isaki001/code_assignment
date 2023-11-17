#include "utils.cuh"
#include <chrono>

std::vector<char>
readfile(const std::string& filename){
  std::ifstream imageFile(filename.c_str(), std::ios::binary);

  if (!imageFile) {
        std::cerr << "Could not open the file!" << std::endl;
        abort();
  }

  std::vector<char> buffer(
    (std::istreambuf_iterator<char>(imageFile)),
    (std::istreambuf_iterator<char>()));

  imageFile.close();
  std::cout << "read file:" << buffer.size() << std::endl;
  return buffer;
}

void store_image(char* d_img, int w, int h){
  std::vector<char> out(w*h*3, {});
  cuda_memcpy_to_host<char>(out.data(), d_img, w*h*3); CudaCheckError();

  std::string filename = "output_" + std::to_string(h) + "x" + std::to_string(w) + ".data";
  std::ofstream outFile(filename.c_str(), std::ios::binary);

  if (!outFile) {
    std::cerr << "Could not open the file for writing!" << std::endl;
  }

  outFile.write(out.data(), out.size());
  outFile.close();
}

__global__ 
void
even_distribution_improved(char3* out, char3* img, const size_t w, const size_t h){

  const size_t row = 2*(blockIdx.y * blockDim.y + threadIdx.y);
  const size_t col = 2*(blockIdx.x * blockDim.x + threadIdx.x);
  const size_t o_index = (blockIdx.y * blockDim.y + threadIdx.y) * w/2 + (blockIdx.x * blockDim.x + threadIdx.x);

  const int x = static_cast<int>(static_cast<unsigned char>(img[index(row, col, w)].x)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row + 1, col, w)].x)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row, col + 1, w)].x)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row + 1, col + 1, w)].x));

  const int y = static_cast<int>(static_cast<unsigned char>(img[index(row, col, w)].y)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row + 1, col, w)].y)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row, col + 1, w)].y)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row + 1, col + 1, w)].y));

  const int z = static_cast<int>(static_cast<unsigned char>(img[index(row, col, w)].z)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row + 1, col, w)].z)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row, col + 1, w)].z)) + 
          static_cast<int>(static_cast<unsigned char>(img[index(row + 1, col + 1, w)].z));

  out[o_index] = {static_cast<char>(x/4), static_cast<char>(y/4), static_cast<char>(z/4)};
  
}

void
even_distribution_improved(){
  std::vector<char> img = readfile("vancouver.data");
  const size_t w_pixels = sqrt(img.size()/3);
  const size_t h_pixels = w_pixels;
  const size_t num_images = log2(w_pixels) + 1;

  char* d_img = cuda_malloc<char>(img.size());
  cuda_memcpy_to_device<char>(d_img, img.data(), img.size());

  char* d_out = cuda_malloc<char>(img.size()/4);
  std::vector<char> out(img.size()/4, {});

  dim3 blockDim(32, 32); // 16x16 threads per block
  dim3 gridDim((w_pixels/2)/blockDim.x, (h_pixels/2)/blockDim.y); // Enough blocks to cover the whole data

  even_distribution_improved<<<gridDim, blockDim>>>((char3*)d_out, (char3*)d_img, w_pixels, h_pixels);
  cudaDeviceSynchronize();
  CudaCheckError();
  
  cuda_memcpy_to_host<char>(out.data(), d_out, img.size()/4);
  std::ofstream outFile("output.data", std::ios::binary);
  if (!outFile) {
    std::cerr << "Could not open the file for writing!" << std::endl;
  }
  outFile.write(out.data(), out.size());
  outFile.close();

  cudaFree(d_img);
  cudaFree(d_out);
}

char* 
downscale(char* d_img, int w, int h){
  //allocate for output
  char* d_out = cuda_malloc<char>((w/2)*(h/2)*3);CudaCheckError();

  dim3 blockDim(32, 32); // 16x16 threads per block
  dim3 gridDim((w/2)/blockDim.x, (h/2)/blockDim.y); // Enough blocks to cover the whole data

  even_distribution_improved<<<gridDim, blockDim>>>((char3*)d_out, (char3*)d_img, w, h);
  cudaDeviceSynchronize();
  CudaCheckError();
  return d_out;
}

void 
compute_iterative_mipmap(std::vector<char> img, int sw, int sh){
  //input: vector of characters for the three pixels, source image width, source image height
  int w = sw;
  int h  = sh;
  char* d_img = cuda_malloc<char>(img.size());
  cuda_memcpy_to_device<char>(d_img, img.data(), img.size());

  while(w > 1 && h > 1){
    //downscale the w x h image to w/2 x h/2 and store in d_out
    char* d_out = downscale(d_img, w, h);  

    store_image(d_out, w/2, h/2);
    //we are done with the original, free it 
    cudaFree(d_img);

    //output of this iteration is the input of the next
    d_img = d_out;
    
    //update dimensions
    w/=2;
    h/=2;
  }
  cudaFree(d_img);
}

void iterative_mipmap(){

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  std::vector<char> img = readfile("vancouver.data");
  const size_t w_pixels = sqrt(img.size()/3);
  const size_t h_pixels = w_pixels;

  std::cout<<"image read\n";
  auto const t0 = std::chrono::high_resolution_clock::now();
  compute_iterative_mipmap(img, w_pixels, h_pixels);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  std::cout << "iterative mipmap:" << dt.count() << " ms" << std::endl;
}


int main(){
    iterative_mipmap();
}