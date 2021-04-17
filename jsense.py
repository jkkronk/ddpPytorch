import numpy as np
import torch
import torch.fft
from utils import fftshift

def normalize_basis(basis_funct):
    """
    We return a list of orthonormal bases using Gram Schmidt algorithm
    :param X: list of basis we would liek to orthonormalize
    :return:
    """
    matrix = np.array(basis_funct).T

    orthonormal_basis, _ = np.linalg.qr(matrix)

    orthonormal_basis = orthonormal_basis.T
    return orthonormal_basis
    
def create_basis_functions(num_rows, num_cols, max_order, orthonormalize=True, show_plot=False):
    """
    We are creating 2D polynomial basis functions
    (ie: a0 + a1*x + a2*y + a3*xy + a4*x^2 + a5*x^2*y + a6*y^2x +....)
    :param num_rows:
    :param num_cols:
    :param image: x estimate of the reconstruction image (if we want the base
    to be multiplied: E^H E bx)
    :return: a list with all basis [(m x n)] arrays
    """
    # We take all the values between 0 and 1, in the x and y axis
    y_axis = np.linspace(-1, 1, num_rows)
    x_axis = np.linspace(-1, 1, num_cols)

    X, Y = np.meshgrid(x_axis, y_axis, copy=False)
    X = X.flatten().T
    Y = Y.flatten().T

    basis_funct = np.zeros(((max_order+1)**2, num_cols*num_rows))
    i = 0
    for power_x in range(0, max_order + 1):
        for power_y in range(0, max_order + 1):
            current_basis = X ** power_x * Y ** power_y
            basis_funct[i,:] = current_basis
            i += 1

    # We normalize the basis function
    if orthonormalize:
        basis_funct = normalize_basis(basis_funct)

    return basis_funct.reshape((max_order+1)**2, num_rows, num_cols)

def sense_reconstruction(coeffs_array, basis_funct, exponential, num_rows,
                             num_cols, num_coils):
    """
    We reconstruct a bias free image with the coefficient estimates we computed.
    Essentially, we are computing B = sum_i (c_i * Phi_i)
    or B = exp(sum_i (c_i * Phi_i))

    :param coeffs_array: array of size [num_coeffs]
    :param basis_funct: a list is size [num_basis_funct] where each entry is a basis function array of size [(m x n)]
    :param exponential: whether we are using the exponentiated basis functions
    :param num_rows
    :param num_cols

    :return: bias field ([n x m] array)
    """
    d_sense = np.zeros((num_rows, num_cols, num_coils), dtype=complex)
    num_coeffs = len(coeffs_array)
    coeffs_array = coeffs_array.reshape(num_coeffs//num_coils, num_coils)
    num_coeffs_ci = (num_coeffs//num_coils)//2

    for ci in range(num_coils):
        for i in range(len(basis_funct)):
            C_coeffs = coeffs_array[i,ci] + 1j * coeffs_array[num_coeffs_ci+i,ci]
            d_sense[:,:,ci] += C_coeffs * basis_funct[i].reshape(num_rows, num_cols)

        # We are using the exponentiated version of the basis function
        if exponential:
            d_sense[:,:,ci] = np.exp(d_sense[:,:,ci])

    return d_sense

def recon_error(coeffs_array, **kwargs):
    """
    This is the function f(x), used in F(x) = 0.5 * sum(rho(f_i(x)**2)
    , where F(x) is the function that we want to minimize to find the
    bias coefficients.
    (U*FFT(s*B*x) - y), where B = sum(c_i * Phi_i)

    :param coeffs: (array) basis coefficients that we want to estimate
    """
    y = kwargs.get('y')
    basis_funct = kwargs.get('basis_funct')
    exponential = kwargs.get('exponential')
    x_estimate = kwargs.get('x_estimate')
    uspat = kwargs.get('uspat')
    sensmaps = kwargs.get('sensmaps')
    error_funct = kwargs.get('error_funct')

    num_rows, num_cols = x_estimate.shape

    d_sense = sense_reconstruction(coeffs_array, basis_funct,
                    exponential, num_rows, num_cols, sensmaps.shape[-1])

    sensmaps = sensmaps * d_sense

    us_estimate_fourier = UFT(x_estimate, uspat, sensmaps)

    #if error_funct == 'norm':
    #    df_funct = np.linalg.norm(us_estimate_fourier - y)
    #elif error_funct == 'abs':
    #df_funct_real = np.abs(us_estimate_fourier.real - y.real)
    #df_funct_imag = np.abs(us_estimate_fourier.imag - y.imag)

    df_funct = np.abs((us_estimate_fourier - y).flatten())

    #    diff.real**2 + diff.imag**2
    #print('df_funct {}'.format(df_funct))
    #print('np.linalg.norm(us_estimate_fourier - y) {}'.format(np.linalg.norm(us_estimate_fourier - y)))
    #print('\ndf_funct shape: {}\n'.format(df_funct.shape))

    return df_funct

def complex_inverse(ctensor, ntry=5) -> "ComplexTensor":
    # Code from https://github.com/kamo-naoyuki/pytorch_complex/blob/b6f82d076f8e6ad035e8573a007c467391d646ff/torch_complex/tensor.py
    # m x n x n
    in_size = ctensor.size()
    a = ctensor.view(-1, ctensor.size(-1), ctensor.size(-1))
    # see "The Matrix Cookbook" (http://www2.imm.dtu.dk/pubdb/p.php?3274)
    # "Section 4.3"
    for i in range(ntry):
        t = i * 0.1

        e = a.real + t * a.imag
        f = a.imag - t * a.real

        try:
            x = torch.matmul(f, e.inverse())
            z = (e + torch.matmul(x, f)).inverse()
        except Exception:
            if i == ntry - 1:
                raise
            continue

        if t != 0.0:
            eye = torch.eye(
                a.real.size(-1), dtype=a.real.dtype, device=a.real.device
            )[None]
            o_real = torch.matmul(z, (eye - t * x))
            o_imag = -torch.matmul(z, (t * eye + x))
        else:
            o_real = z
            o_imag = -torch.matmul(z, x)
            
        o = torch.complex(o_real, o_imag).to(a.real.device)
        return o.view(*in_size)


def sense_estimation_ls(y, X, basis_funct, uspat):
    """
    We estimate the bias field with a polynomial basis of the given order, using least squares method
    :param data: data (Fourier) [n x m, c]
    :param x_estimate: predicted reconstruction estimate [n x m, c] (needed to compute recon_error!)
    :param max_basis_order:
    :param ls_threshold:
    :param max_bias_eval:
    :return:
    """
    num_coils, sizex, sizey = y.shape
    num_coeffs = basis_funct.shape[1]

    coeff_coils = torch.zeros((num_coils, num_coeffs), dtype=torch.cfloat, device=y.real.device)
    # XA - Y = 0
    for i in range(num_coils):
        Y = y[i,:,:].reshape(sizex*sizey) 
        A = UFT(X, uspat, basis_funct[i,:,:,:]).reshape(num_coeffs, sizex*sizey) 
        coeff = torch.matmul(torch.matmul(Y, torch.transpose(torch.conj(A), 0, 1)), complex_inverse(torch.matmul(A, torch.transpose(torch.conj(A), 0, 1))))
        coeff_coils[i,:] = coeff.clone()
        del Y
        del A
        del coeff
        
    return coeff_coils

def UFT(x, uspat, sensmaps):
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny, ns]
    return uspat[:, :].unsqueeze(0) * fftshift(torch.fft.fftn(sensmaps * x[:, :].unsqueeze(0), dim=(1, 2)), dim=(1, 2))












