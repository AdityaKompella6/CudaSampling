import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt

sample_lib = load(name="sample", sources=["sample.cu", f"sample.cpp"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"])

def sample_torch(num_samples,mu,sigma):
    return mu + sigma * torch.randn((1,num_samples),device="cuda")
compiled_sample_torch = torch.compile(sample_torch)

torch_times = []
my_kernel_times = []
compiled_times = []
vector_dimensions = []
for d in range(10,5000, 100):
    num_samples = 1024
    vector_dim = d
    vector_dimensions.append(vector_dim)
    warmup_steps = 50
    mu = torch.randn((vector_dim,1),device="cuda")
    sigma = torch.randn((vector_dim,1),device="cuda")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    torch.cuda.empty_cache()
    for i in range(1000):
        start.record()
        samples = sample_lib.sample(mu,sigma,num_samples)
        end.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if i > warmup_steps:
            times.append(start.elapsed_time(end) / 1000)
    my_time = torch.mean(torch.tensor(times))
    my_kernel_times.append(my_time)
    ##These commented lines are a good check that you are sampling correctly since the average of the samples should be close to mu
    # print(samples.mean(1))
    # print(mu)

    times = []
    torch.cuda.empty_cache()
    for i in range(1000):
        start.record()
        # samples = torch.randn((num_samples,vector_dim),device="cuda") * sigma + mu
        samples = sample_torch(num_samples,mu,sigma)
        end.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if i > warmup_steps:
            times.append(start.elapsed_time(end) / 1000)
    torch_time = torch.mean(torch.tensor(times))
    torch_times.append(torch_time)

    times = []
    torch.cuda.empty_cache()
    for i in range(1000):
        start.record()
        # samples = torch.randn((num_samples,vector_dim),device="cuda") * sigma + mu
        samples = compiled_sample_torch(num_samples,mu,sigma)
        end.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if i > warmup_steps:
            times.append(start.elapsed_time(end) / 1000)
    compiled_torch_time = torch.mean(torch.tensor(times))
    compiled_times.append(compiled_torch_time)

# ##Timing Plot
# plt.plot(vector_dimensions, torch_times, label="Torch")
# plt.plot(vector_dimensions, my_kernel_times, label="My Kernel") 
# plt.plot(vector_dimensions, compiled_times, label="Torch Compiled")
# plt.title("Timing Comparison")
# plt.xlabel("Vector Dimension")
# plt.ylabel("Time (s)")
# plt.legend()
# plt.show()

##Speedup Plot
plt.plot(vector_dimensions, torch.tensor(torch_times)/torch.tensor(my_kernel_times))
plt.title("Speedup Comparison")
plt.xlabel("Vector Dimension")
plt.ylabel("Speedup")
plt.show()
print(f"Average Speedup over all Vector Sizes: {torch.mean(torch.tensor(torch_times)/torch.tensor(my_kernel_times))}")




