#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <c10/cuda/CUDAException.h>
/*
Lets try to create a fast GPU normal dist sampling algorithm:
It will require this kernels:
Sample: Input = mu,sigma 1d-vectors, N where N is number of vectors to sample. Output: d*N matrix of vectors
We can parallelize over each element in the mu,sigma vector and then have each thread write N numbers
*/

// Threads parallelize over N, blocks parallelize over d
__global__ void sample(float *d_mu, float *d_sigma, float *d_rand, float *d_output, int d, int N)
{
    int dimension_id = blockIdx.x;
    int sample_id = threadIdx.x;

    d_output[dimension_id * N + sample_id] = d_mu[dimension_id] + d_sigma[dimension_id] * d_rand[sample_id];
}

void sample_gpu(int d, int num_samples, float *d_mu, float *d_sigma, float *d_output, float *d_rand)
{
    sample<<<d, num_samples>>>(d_mu, d_sigma, d_rand, d_output, d, num_samples);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/*This commented out block of code is what tests the kernel to see if
it ouptuts reasonable code*/

// void printArray(float *arr, int d, int N)
// {
//     for (int i = 0; i < N; i++)
//     {
//         std::cout << "Vector " << i << ": ";
//         for (int j = 0; j < d; j++)
//         {
//             std::cout << arr[i * d + j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// int main()
// {
//     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//     // curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
//     const int num_samples = 10;
//     const int d = 10;
//     float mu[d];
//     for (int i = 0; i < d; i += 1)
//     {
//         mu[i] = 1.0f;
//     }
//     float sigma[d];
//     float output[d * num_samples];
//     for (int i = 0; i < d; i += 1)
//     {
//         sigma[i] = 0.5f;
//     }
//     float *d_mu;
//     float *d_sigma;
//     float *d_output;
//     float *d_rand;
//     cudaMalloc((void **)&d_mu, d * sizeof(float));
//     cudaMalloc((void **)&d_sigma, d * sizeof(float));
//     cudaMalloc((void **)&d_output, d * num_samples * sizeof(float));
//     cudaMalloc((void **)&d_rand, num_samples * sizeof(float));
//     curandGenerateNormal(gen, d_rand, num_samples, 0, 1);
//     cudaMemcpy(d_mu, mu, d * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_sigma, sigma, d * sizeof(float), cudaMemcpyHostToDevice);

//     sample<<<d, num_samples>>>(d_mu, d_sigma, d_rand, d_output, d, num_samples);
//     cudaMemcpy(output, d_output, num_samples * d * sizeof(float), cudaMemcpyDeviceToHost);
//     // for (int i = 0; i < num_elements; i += 1)
//     // {
//     //     printf("Element %d = %f\n", i, arr[i]);
//     // }
//     cudaFree(d_output);
//     cudaFree(d_output);
//     cudaFree(d_output);
//     printArray(output, d, num_samples);
//     return 0;
// }