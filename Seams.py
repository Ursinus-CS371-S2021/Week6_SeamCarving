import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import filters
import skimage.io
from skimage.color import rgb2gray
from scipy.signal import convolve2d


def grad_energy(img, sigma = 3):
    """
    Compute the gradient magnitude of an image by doing
    1D convolutions with the derivative of a Gaussian
    
    Parameters
    ----------
    img: ndarray(M, N, 3)
        A color image
    sigma: float
        Width of Gaussian to use for filter
    
    Returns
    -------
    ndarray(M, N): Gradient Image
    """
    I = rgb2gray(img)
    N = int(sigma*6+1)
    t = np.linspace(-3*sigma, 3*sigma, N)
    dgauss = -t*np.exp(-t**2/(2*sigma**2))
    IDx = convolve2d(I, dgauss[None, :], mode='same')
    IDy = convolve2d(I, dgauss[:, None], mode='same')
    Grad = np.sqrt(IDx**2 + IDy**2)
    return Grad

def energy_img_seam(energy):
    """
    Compute a vertical seam from an energy image using
    dynamic programming
    
    Parameters
    ----------
    energy: ndarray(M, N)
        The energy image
    
    Returns
    -------
    list: A list of length M with the column index of each
          element in the optimal seam, from row 0 to row M-1
    """
    M, N = energy.shape
    # total_energy is a MxN array that holds the minimum energy over 
    # all seams that start at the top and reach this pixel
    total_energy = np.zeros_like(energy)
    # choices is A MxN array that holds:
    # a 0 if the optimal seam came from [i-1, j-1]
    # a 1 if the optimal seam came from [i-1, j]
    # a 2 if the optimal seam came from [i-1, j+1]
    choices = np.zeros(energy.shape, dtype=int)
    
    ## Step 1: Perform dynamic programming to determine 
    ## the cost of the minimum energy seam up to each pixel
    # Base Case: First row
    total_energy[0, :] = energy[0, :]
    # Perform dynamic programming for the rest of the rows
    for i in range(1, M):
        # Figure out the sum of the energy
        # from left, up, and right
        # (A fast numpy version that knocks a whole row out at once)
        L = np.zeros(N)
        L[1::] = total_energy[i-1, 0:-1]
        L[0] = np.inf
        U = total_energy[i-1, :]
        R = np.zeros(N)
        R[0:-1] = total_energy[i-1, 1::]
        R[-1] = np.inf
        choices[i, :] = np.argmin([L, U, R], axis=0)
        total_energy[i, :] = energy[i, :] + np.min([L, U, R], axis=0)
    
    ## Step 2: Backtrace from the bottom row to find the seam
    # Find the column index of the minimum cost seam at the bottom
    # row and start there
    j = np.argmin(total_energy[-1, :])
    seam = [0]*M
    ## TODO: Complete the backtracing to fill in the column
    ## indices in seam, starting at row M-1
    return seam


def remove_vertical_seam(img, seam):
    """
    Remove a vertical seam from an image

    Parameters
    ----------
    I: ndarray(nrows, ncols, 3)
        An RGB image
    seam: ndarray(nrows, dtype=int)
        A list of column indices of the seam from
        top to bottom

    Returns
    ----------
    ndarray(nrows, ncols-1, 3)
        An RGB image with the seam removed
    """
    for row in range(img.shape[0]):
        idx = seam[row]
        if len(img.shape) > 2:
            img[row, idx:-1, :] = img[row, idx+1::, :]
        else:
            img[row, idx:-1] = img[row, idx+1::]
    if len(img.shape) > 2:
        return img[:, 0:-1, :]
    else:
        return img[:, 0:-1]

def plot_seam(img, seam):
    """
    Plot a seam on top of the image
    Parameters
    ----------
    I: ndarray(nrows, ncols, 3)
        An RGB image
    seam: ndarray(nrows, dtype=int)
        A list of column indices of the seam from
        top to bottom
    """
    plt.imshow(img)
    X = np.zeros((len(seam), 2))
    X[:, 0] = np.arange(len(seam))
    X[:, 1] = seam
    plt.plot(X[:, 1], X[:, 0], 'r')


def seam_carve(impath, num_seams, mask_path = "", show_progress = True):
    """
    Perform seam carving on an image using vertical seams
    
    Parameters
    ----------
    impath: string
        Path to image file
    num_seams: int
        Number of seams to remove
    mask_path: string
        Path to a file to use as a mask, or empty string if none
    show_progress: boolean
        Whether to draw images showing each seam that's removed
    """
    img = skimage.io.imread(impath)
    mask = np.ones((img.shape[0], img.shape[1]))
    if len(mask_path) > 0:
        mask_img = skimage.io.imread(mask_path)
        mask *= mask_img[:, :, 0] > 0
    if show_progress:
        aspect = img.shape[1]/img.shape[0]
        plt.figure(figsize=(aspect*10, 10))
    for i in range(num_seams):
        print(i)
        energy = grad_energy(img)*mask
        seam = energy_img_seam(energy)
        if show_progress:
            plt.clf()
            plot_seam(img, seam)
            plt.savefig("{}.png".format(i))
        img = remove_vertical_seam(img, seam)
        mask = remove_vertical_seam(mask, seam)
    return img

img = seam_carve("LivingRoom.jpg", 50, show_progress=True)
skimage.io.imsave("Result.png", img)