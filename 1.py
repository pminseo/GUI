import numpy as np

def gaussianFiltering(*args):
    gaussianfilter = getGaussianfilter(kernel_size = 3, sigma = 1)
    print(gaussianfilter)
    
    pass

def getGaussianfilter(**kargs):
    kernel_size = kargs["kernel_size"] if "kernel_size" in kargs else 3
    sigma = kargs["sigma"] if "sigma" in kargs else 1
    array = np.arange((kernel_size // 2) * (-1), (kernel_size // 2) + 1, dtype=np.float32)
    arr = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            arr[i, j] = array[i]**2 + array[j]**2

    gaussianFilter = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussianFilter[i, j] = np.exp(-arr[i, j] / (2 * sigma**2))
    gaussianFilter /= gaussianFilter.sum()
    return gaussianFilter


if __name__ == "__main__":
    gaussianFiltering()