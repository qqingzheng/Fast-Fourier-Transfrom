import numpy as np
import matplotlib.pyplot as plt
import time
def similar_equal(array1, array2):
    array1 = array1.reshape(-1)
    array2 = array2.reshape(-1)
    equal = True
    for i in range(array1.shape[0]):
        if np.abs(array1[i]/(array2[i]+1e-16)) > 1e8:
            print(array1[i], array2[i])
            equal = False
            break
    return equal
def complex_similar_equal(array1, array2):
    equal = similar_equal(np.real(array1), np.real(array2)) and similar_equal(np.imag(array1), np.imag(array2))
    return equal

def generate_dft_matrix(n):
    # n means number of samples.
    dft_matrix = np.ones((n,n), dtype=np.complex128)
    omega_n = np.exp(-1j*2*np.pi/n)
    order = np.arange(1, n)
    for i in range(1,n):
        dft_matrix[i,1:] = np.power(omega_n, order*i) # n-1
    return dft_matrix

def generate_dft_matrix(n):
    # n means number of samples.
    dft_matrix = np.ones((n,n), dtype=np.complex128)
    omega_n = np.exp(-1j*2*np.pi/n)
    order = np.arange(1, n)
    for i in range(1,n):
        dft_matrix[i,1:] = np.power(omega_n, order*i) # n-1
    return dft_matrix

def regular_dft(data):
    # only support one input data vector. And input data type must be numpy.darray
    data = data.reshape(-1)
    length_of_data = data.shape[0]
    return np.matmul(generate_dft_matrix(length_of_data), data)

def fast_dft(data):
    data = data.reshape(-1)
    length_of_data = int(np.power(2,np.ceil(np.log2(data.shape[0]))))
    padding_data = np.zeros(length_of_data)
    padding_data[0:data.shape[0]] = data
    if(length_of_data == 2):
        dft_matrix = generate_dft_matrix(2)
        return (dft_matrix@padding_data[:,None]).reshape(-1)
    even_dft = fast_dft(padding_data[list(range(0,length_of_data,2))])
    odd_dft = fast_dft(padding_data[list(range(1,length_of_data,2))])
    omega_n = np.exp(-1j*2*np.pi/length_of_data)
    order = np.arange(0, length_of_data/2)
    omega_order = np.diag(np.power(omega_n, order))
    top_half = even_dft+np.matmul(omega_order,odd_dft)
    bottom_half = even_dft-np.matmul(omega_order,odd_dft)
    result = np.concatenate((top_half, bottom_half), axis=-1)
    # if data.shape[-1] != length_of_data:
    #     recover_matrix = 
    return result[0:data.shape[-1]]


if __name__ == "__main__":
    # print(generate_dft_matrix(80))
    # Make a function
    x = np.arange(-10.24, 10.24, 0.01) # make sample points's number is odd.\
    y = np.zeros_like(x)
    length_of_data = x.shape[0]
    center_index = (length_of_data-1)//2
    y[center_index-1:center_index+51] = 1

    
    # regular dft
    regular_dft_figure = plt.figure()
    regular_dft_figure.subplots_adjust(hspace=0.5)
    raxes = regular_dft_figure.add_subplot(3,1,1)
    raxes.set_title("Input Function")
    raxes.plot(x, y)
    rdaxes = regular_dft_figure.add_subplot(3,1,2)
    dft_result = regular_dft(y)
    rdaxes.set_title("DFT Result")
    rdaxes.plot(abs(dft_result))
    irdaxes = regular_dft_figure.add_subplot(3,1,3)
    irdaxes.set_title("Numpy Lib IDFT Result")
    irdaxes.plot(x, np.fft.ifft(dft_result))
    regular_dft_figure.savefig("dft.png")
    # verify regular dft
    if complex_similar_equal(dft_result, np.fft.fft(y)):
        print("regular dft equal")
    else:
        print("regular dft inequal")
        
    # fast dft
    fast_dft_figure = plt.figure()
    fast_dft_figure.subplots_adjust(hspace=0.5)
    fraxes = fast_dft_figure.add_subplot(3,1,1)
    fraxes.set_title("Input Function")
    fraxes.plot(x, y)
    frdaxes = fast_dft_figure.add_subplot(3,1,2)
    frdaxes.set_title("FFT Result")
    fdft_result = fast_dft(y)
    frdaxes.plot(abs(fdft_result))
    firdaxes = fast_dft_figure.add_subplot(3,1,3)
    firdaxes.set_title("Numpy Lib IFFT Result")
    firdaxes.plot(x, np.fft.ifft(fdft_result))
    fast_dft_figure.savefig("fft.png")
    
    # verify fast dft
    if complex_similar_equal(fdft_result, np.fft.fft(y)):
        print("fast dft equal")
    else:
        print("fast dft inequal")
        
    # time cosuming test
    start = time.time()
    for i in range(10):
        regular_dft(y)
    end = time.time()
    print(f"DFT time:{(end - start)/10}")
    start = time.time()
    for i in range(10):
        fast_dft(y)
    end = time.time()
    print(f"FFT time:{(end - start)/10}")