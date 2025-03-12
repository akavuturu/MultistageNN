import warnings
warnings.simplefilter("ignore")
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import time
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fftshift, fftn
import matplotlib.pyplot as plt
from pylab import *

from utils import NeuralNet, create_ds, poisson, calculate_N, normalize
import argparse

parser = argparse.ArgumentParser(description="Initialize and train the network.")
# Network initialization arguments
parser.add_argument(
    "--num_stages", type=int, default=4, help="Upper limit on number of stages in the network."
)
parser.add_argument(
    "--precision", type=float, default=1e-16, help="Upper limit on required error of final stage."
)
parser.add_argument(
    "--num_hidden_layers", type=int, default=3, help="Number of hidden layers per stage."
)
parser.add_argument(
    "--num_hidden_nodes", type=int, default=20, help="Number of hidden nodes per layer."
)
parser.add_argument(
    "--dim", type=int, required=True, help="Dimensionality of the input data."
)
parser.add_argument(
    "--L", type=float, default=2.04, help="Size of domain [-L/2, L/2]"
)
parser.add_argument(
    "--plot", type=bool, default=False, help="Save loss plot"
)
args = parser.parse_args()
logging.basicConfig(
    filename=f"logs/MSNN_DIM_{args.dim}.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)   

logging.info(f"Parsed Arguments: {args}")

class MultistageNeuralNetwork:
    """
    MultistageNeuralNetwork is a multi-stage model used for predicting
    high-dimensional function outputs through regression. This class encapsulates
    the functionality for constructing, training, and predicting using a sequence
    of neural networks, where each stage of the network can focus on different
    aspects of the data.

    Attributes:
        dim (int): Dimensionality of the input data.
        N (int): Number of points per dimension in the dataset.
        stages (list): List of neural networks, one for each stage.
        layers (list): Architecture of the neural network (input layer, hidden layers, output layer).
        lt (list): Minimum values for each dimension in the input data.
        ut (list): Maximum values for each dimension in the input data.
    """

    def __init__(self, x_train, num_hidden_layers, num_hidden_nodes):
        """
        Initialize the MultistageNeuralNetwork instance.

        Args:
            x_train (tf.Tensor): Input training data.
            num_stages (int): Number of stages in the multi-stage neural network.
            num_hidden_layers (int): Number of hidden layers in each stage.
            num_hidden_nodes (int): Number of nodes in each hidden layer.
        """
        self.dim = x_train.shape[-1]                                 # Number of dimensions in the input data.
        self.N = int(round(x_train.shape[0] ** (1 / self.dim)))      # Number of points per dimension.
        # self.stages = [None] * num_stages                            # List to store each stage's neural network.
        self.stages = []
        self.layers = [self.dim] + ([num_hidden_nodes] * num_hidden_layers) + [1]  # Neural network architecture.
        self.lt = [tf.math.reduce_min(x_train[:, i]) for i in range(x_train.shape[-1])]  # Min values for each dimension.
        self.ut = [tf.math.reduce_max(x_train[:, i]) for i in range(x_train.shape[-1])]  # Max values for each dimension.

    def train(self, x_train, y_train, stage, kappa, iters):
        """
        Train a specific stage of the neural network.

        Args:
            x_train (tf.Tensor): Input training data.
            y_train (tf.Tensor): Corresponding labels for training.
            stage (int): The stage index to train.
            kappa (float): Scaling factor for activation.
            iters (list): Number of iterations for [Adam optimizer, L-BFGS optimizer].
        """
        act = 0 if stage == 0 else 1  # Use different activation for first stage.
        lt = [tf.cast(tf.math.reduce_min(x_train[:, i]).numpy(), dtype=tf.float64) for i in range(x_train.shape[-1])]
        ut = [tf.cast(tf.math.reduce_max(x_train[:, i]).numpy(), dtype=tf.float64) for i in range(x_train.shape[-1])]

        self.stages.append(NeuralNet(x_train, y_train, self.layers, kappa=kappa, lt=lt, ut=ut, acts=act))
        # self.stages[stage] = NeuralNet(x_train, y_train, self.layers, kappa=kappa, lt=lt, ut=ut, acts=act)
        self.stages[stage].train(iters[0], 1)  # Train using Adam optimizer.
        self.stages[stage].train(iters[1], 2)  # Train using L-BFGS optimizer.

    @staticmethod
    def fftn_(x_train, residue):
        """
        Perform a Fast Fourier Transform (FFT) to analyze the frequency domain of the residue.

        Args:
            x_train (tf.Tensor): Input training data.
            residue (tf.Tensor): Residual errors between predictions and true values.

        Returns:
            float: Adjusted scaling factor (kappa) based on the dominant frequency.
        """
        dim = x_train.shape[-1]
        N_train = int(round(x_train.shape[0] ** (1 / dim)))
        g = residue.numpy()

        GG = g.reshape([N_train] * dim)  # Reshape residue into a grid.
        G = fftn(GG)                    # Perform FFT.
        G_shifted = fftshift(G)         # Shift zero-frequency component to the center.

        N = len(G)
        total_time_range = 2  # Time range from -1 to 1.
        sample_rate = N / total_time_range  # Sampling rate.

        half_N = N // 2
        T = 1.0 / sample_rate
        idxs = tuple(slice(half_N, N, 1) for _ in range(dim))
        G_pos = G_shifted[idxs]  # Extract positive frequencies.

        freqs = [fftshift(fftfreq(GG.shape[i], d=T)) for i in range(len(GG.shape))]
        freq_pos = [freqs[i][half_N:] for i in range(len(freqs))]

        magnitude_spectrum = np.abs(G_pos)
        max_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        dominant_freqs = [freq_pos[i][max_idx[i]] for i in range(len(freq_pos))]
        magnitude = magnitude_spectrum[max_idx] / (N ** dim)  # Normalize magnitude.

        dominant_freq = max(dominant_freqs)
        # print(f"Sample rate = {sample_rate} Hz, Dominant Frequency = {dominant_freq} Hz, Magnitude = {magnitude}")

        kappa_f = 2 * np.pi * dominant_freq if dominant_freq > 0 else 2 * np.pi * 0.01
        # print(f"New Kappa: {kappa_f}")
        return kappa_f, dominant_freq
    
    def sfftn(x_train, residue, sparsity_threshold=0.01, k=None):
        """
        Perform a Sparse Fast Fourier Transform (SFFT) to analyze the frequency domain of the residue,
        optimized for high-dimensional data. Assumes data is already on a complete grid.

        Args:
            x_train (tf.Tensor): Input training data with coordinates.
            residue (tf.Tensor): Residual errors between predictions and true values.
            sparsity_threshold (float): Threshold below which frequency components are set to zero (relative to max).
            max_frequencies (int, optional): Maximum number of frequency components to keep. If None, use threshold only.

        Returns:
            tuple: (kappa_f, dominant_freq, sparse_spectrum)
                - kappa_f: Adjusted scaling factor based on the dominant frequency
                - dominant_freq: The dominant frequency identified
                - sparse_spectrum: Sparse representation of the frequency spectrum
        """
        dim = x_train.shape[-1]
        N_train = int(round(x_train.shape[0] ** (1 / dim)))
        g = residue.numpy().flatten()
        
        # For very high dimensions, we may need to use a smaller grid
        if dim > 4:
            N_per_dim = min(N_train, max(8, int(32 / np.sqrt(dim))))
        else:
            N_per_dim = N_train
        
        GG = g.reshape([N_per_dim] * dim)
        G = fftn(GG)
        G_shifted = fftshift(G)
        
        N = N_per_dim
        total_time_range = 2  # Time range from -1 to 1
        sample_rate = N / total_time_range
        half_N = N // 2
        T = 1.0 / sample_rate
        
        idxs = tuple(slice(half_N, N, 1) for _ in range(dim))
        G_pos = G_shifted[idxs]
        
        freqs = [fftshift(fftfreq(N, d=T)) for _ in range(dim)]
        freq_pos = [freqs[i][half_N:] for i in range(dim)]

        magnitude_spectrum = np.abs(G_pos)
        max_magnitude = np.max(magnitude_spectrum)

        if sparsity_threshold > 0:
            mask = magnitude_spectrum > (sparsity_threshold * max_magnitude)
            if k is not None and np.sum(mask) > k:
                flat_idx = np.argsort(magnitude_spectrum.flatten())[::-1][:k]
                new_mask = np.zeros_like(magnitude_spectrum, dtype=bool).flatten()
                new_mask[flat_idx] = True
                mask = new_mask.reshape(magnitude_spectrum.shape)
        else:
            mask = np.ones_like(magnitude_spectrum, dtype=bool)
        
        sparse_magnitude = np.zeros_like(magnitude_spectrum)
        sparse_magnitude[mask] = magnitude_spectrum[mask]
        
        max_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        dominant_freqs = [freq_pos[i][max_idx[i]] for i in range(dim)]
        magnitude = magnitude_spectrum[max_idx] / (N ** dim)
        
        dominant_freq = max(dominant_freqs)
        kappa_f = 2 * np.pi * dominant_freq if dominant_freq > 0 else 2 * np.pi * 0.01
        
        if dim <= 2:
            sparse_spectrum = sparse_magnitude
        else:
            sparse_idx = np.where(mask)
            values = magnitude_spectrum[sparse_idx]
            sparse_spectrum = (sparse_idx, values, magnitude_spectrum.shape)
        
        return kappa_f, dominant_freq, sparse_spectrum

    def find_zeros(residue):
        sign_residue = np.sign(residue)
        num_zeros = 0
        for axis in range(residue.ndim):
            shifted_signs = np.roll(sign_residue, shift=-1, axis=axis)
            mask = (sign_residue[:-1] * shifted_signs[:-1]) < 0
            num_zeros += np.count_nonzero(mask)
        kappa = 3 * num_zeros
        return kappa

if __name__ == "__main__":

    dim = 10
    lo, hi = -1, 1
    points_per_dim = 5
    N = points_per_dim ** dim
    x_train = create_ds(dim, lo, hi, N)
    y_train = tf.reshape(poisson(x_train), [len(x_train), 1])
    print(f"Data shapes: {x_train.shape}, {y_train.shape}")
    
    # Compute FFT-based dominant frequency
    start_time = time.time()
    kappa_f1, dominant_freq1 = MultistageNeuralNetwork.fftn_(x_train, y_train)
    fft_time = time.time() - start_time

    start_time = time.time()
    kappa_f2, dominant_freq2, spectrum = MultistageNeuralNetwork.sfftn(x_train, y_train)
    sfft_time = time.time() - start_time

    start_time = time.time()
    kappa_f3 = MultistageNeuralNetwork.find_zeros(y_train)
    zeros_time = time.time() - start_time

    # Print results
    print(f"Full FFT:   Dominant Frequency = {dominant_freq1}, Kappa = {kappa_f1}, Time={fft_time}")
    print(f"Sparse FFT: Dominant Frequency = {dominant_freq2}, Kappa = {kappa_f2}, Time={sfft_time}")
    print(f"Zero Crossing: Kappa = {kappa_f3}, Time={zeros_time}")

    # Validate results
    assert np.isclose(dominant_freq1, dominant_freq2, atol=1e-3), "Dominant frequencies do not match closely!"

    print("✅ Test Passed: Sparse FFT produces similar results to Full FFT.")

    sys.exit(1)

    dim = args.dim
    L = args.L
    num_stages = args.num_stages
    num_hidden_layers = args.num_hidden_layers
    num_hidden_nodes = args.num_hidden_nodes
    precision = args.precision
    points_per_dim = 20

    N_train = points_per_dim ** dim
    x_train = create_ds(dim, -L/2, L/2, N_train)
    y_train = tf.reshape(poisson(x_train), [len(x_train), 1])
    training_iters = list([(3000, 6000)] + [(5000, 8000*i) for i in range(2, 15)])[:num_stages]

    MSNN = MultistageNeuralNetwork(x_train, num_hidden_layers, num_hidden_nodes)
    kappa = 1
    logging.info(f"TRAINING STAGE {1}: Data size: {x_train.shape}")
    MSNN.train(x_train, y_train, stage=0, kappa=1, iters=training_iters[0])
    curr_residue = y_train - tf.add_n([MSNN.stages[j].predict(x_train) for j in range(1)])
    mean_residue = tf.reduce_mean(tf.abs(curr_residue))
    logging.info(f"Completed training stage 1, loss={MSNN.stages[-1].loss[-1]}, residue={mean_residue}")

    i = 1
    while mean_residue > precision and i < num_stages:
        # kappa, f_d = MultistageNeuralNetwork.fftn_(x_train, curr_residue)
        # if f_d == 0: 
        #     logging.info("Oops! Your dominant frequency is 0...")
        #     break
        kappa = MultistageNeuralNetwork.find_zeros(curr_residue)
        
        # N_train = int(6 * np.pi * f_d)
        # x_train = create_ds(dim, -L/2, L/2, N_train, max_data_size)
        # y_train = tf.reshape(poisson(x_train), [len(x_train), 1])

        logging.info(f"TRAINING STAGE {i + 1}: Data size: {x_train.shape}")

        curr_residue = y_train - tf.add_n([MSNN.stages[j].predict(x_train) for j in range(i)])
        MSNN.train(x_train, curr_residue, stage=i, kappa=kappa, iters=training_iters[i])

        mean_residue = tf.reduce_mean(tf.abs(y_train - tf.add_n([MSNN.stages[j].predict(x_train) for j in range(i)])))
        logging.info(f"Completed training stage {i + 1}, loss={MSNN.stages[i].loss[-1]}, residue={mean_residue}")
        i += 1

    if i >= num_stages:
        logging.info(f"Reached maximum stages, loss={MSNN.stages[-1].loss[-1]}, residue={mean_residue}")
    elif mean_residue <= precision:
        logging.info(f"Reached threshold precision, loss={MSNN.stages[-1].loss[-1]}, residue={mean_residue}")
    else:
        logging.info(f"Exited training unexpectedly, previous loss={MSNN.stages[-2].loss[-1]}, previous residue={mean_residue}")

    if args.plot:
        loss = np.concatenate([stage.loss for stage in MSNN.stages])
        plt.figure()
        plt.plot(loss, 'b-')
        plt.yscale("log")
        plt.title(f"MSNN Dim {dim} Loss Plot")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(f"plots/MSNN_DIM_{dim}.png")
