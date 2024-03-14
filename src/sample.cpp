#include <torch/extension.h>
#include "ATen/ATen.h"
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
void sample_gpu(int d, int num_samples, float *d_mu, float *d_sigma, float *d_output, float *d_rand);

torch::Tensor sample(torch::Tensor mu, torch::Tensor sigma, int num_samples)
{
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    int d = mu.size(0);
    auto output = torch::zeros({d, num_samples}, mu.options());
    auto rand = torch::randn({num_samples}, mu.options());
    sample_gpu(d, num_samples, mu.data_ptr<float>(), sigma.data_ptr<float>(), output.data_ptr<float>(), rand.data_ptr<float>());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sample", &sample, "gpu normal dist sampling");
}
