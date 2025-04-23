import os
import sys
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.fft import fftfreq, fftshift, fftn
from utils_gpu import NeuralNet, create_ds, poisson

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
        
        grid = g.reshape([N_train] * dim)

        # For very high dimensions, we may need to use a smaller grid
        downsample_factor = 1
        if dim > 4:
            downsample_factor = max(1, N_train // 16)

        slices = tuple(slice(None, None, downsample_factor) for _ in range(dim))
        GG = grid[slices]
        G = fftn(GG)
        G_shifted = fftshift(G)
        
        N = GG.shape[0]
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

    def predict(self, x_test):
        """Make prediction using all stages combined."""
        return tf.add_n([self.stages[j].predict(x_test) for j in range(len(self.stages))])



def run_experiment(dim, output_file=None):
    print(f"\n===== Training MSNN for dimension {dim} =====")
    L = 2.04
    num_stages = 4
    num_hidden_layers = 3
    num_hidden_nodes = 20
    
    if dim == 1: points_per_dim = 100
    elif dim == 2: points_per_dim = 50
    elif dim <= 5: points_per_dim = 15
    elif dim <= 8: points_per_dim = 5
    else: points_per_dim = 3
    
    N_train = points_per_dim ** dim
    print(f"Using {points_per_dim} points per dimension (total points: {N_train})")
    
    test_points_per_dim = min(points_per_dim, 20) if dim > 5 else min(points_per_dim, 50)
    
    N_test = test_points_per_dim ** dim
    x_test = create_ds(dim, -L/2, L/2, N_test)
    y_test = tf.reshape(poisson(x_test), [len(x_test), 1])
    print(f"Test set: {test_points_per_dim} points per dimension (total: {N_test})")
    
    if dim == 1: batch_size = 1000
    elif dim == 2: batch_size = 800
    elif dim <= 5: batch_size = 500
    elif dim <= 8: batch_size = 200
    else: batch_size = 100

    max_batches = 10 if dim < 8 else 20
    training_iters = [(1000, 2000)] + [(2000, 4000)] * (num_stages-1)
    
    init_points_per_dim = min(5, points_per_dim)
    init_points = init_points_per_dim ** dim
    x_init = create_ds(dim, -L/2, L/2, init_points)
    msnn = MultistageNeuralNetwork(x_init, num_hidden_layers, num_hidden_nodes)
    
    num_batches = min(max_batches, (N_train + batch_size - 1) // batch_size)
    print(f"Using {num_batches} batches of size ~{batch_size} for training")
    
    total_time = 0
    stage_times = []
    mean_residues = []
    
    initial_residue = tf.reduce_mean(tf.abs(y_test)).numpy()
    print(f"Initial mean residue before training: {initial_residue:.6e}")
    
    for stage in range(num_stages):
        print(f"\nTRAINING STAGE {stage+1}")
        stage_start_time = time.time()
        kappa = 1 if stage == 0 else kappa_s
        
        for batch_idx in range(num_batches):
            tf.keras.backend.clear_session()

            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, N_train)
            actual_batch_size = batch_end - batch_start
            
            x_batch = create_ds(dim, -L/2, L/2, actual_batch_size)
            y_batch = tf.reshape(poisson(x_batch), [len(x_batch), 1])
            
            if stage > 0:
                pred_chunk_size = min(actual_batch_size, 500)
                y_pred = []
                
                for i in range(0, actual_batch_size, pred_chunk_size):
                    end_idx = min(i + pred_chunk_size, actual_batch_size)
                    x_pred_chunk = x_batch[i:end_idx]
                    pred_chunk = msnn.predict(x_pred_chunk)
                    y_pred.append(pred_chunk)
                
                y_pred = tf.concat(y_pred, axis=0)
                y_batch_residue = y_batch - y_pred
            else:
                y_batch_residue = y_batch
            
            print(f"  Batch {batch_idx+1}/{num_batches}: Training with {actual_batch_size} points")
            
            if stage == 0 and batch_idx == 0:
                msnn.train(x_batch, y_batch_residue, stage=stage, kappa=kappa, iters=training_iters[stage])
            else:
                adjusted_iters = (max(int(training_iters[stage][0] / (batch_idx + 1)), 200), 
                                 max(int(training_iters[stage][1] / (batch_idx + 1)), 500))
                msnn.train(x_batch, y_batch_residue, stage=stage, kappa=kappa, iters=adjusted_iters)
        
        stage_time = time.time() - stage_start_time
        total_time += stage_time
        stage_times.append(stage_time)
        print(f"Stage {stage+1} training completed in {stage_time:.1f}s")
        
        y_test_pred = msnn.predict(x_test)
        test_residue = y_test - y_test_pred
        curr_residue = tf.reduce_mean(tf.abs(test_residue)).numpy()
        mean_residues.append(curr_residue)
        print(f"Stage {stage+1} mean residue: {curr_residue:.6e}")
        
        if stage < num_stages - 1:
            y_test_pred = msnn.predict(x_test)
            test_residue = y_test - y_test_pred
            kappa_s, dominant_freq, _ = MultistageNeuralNetwork.sfftn(x_test, test_residue)
            print(f"Next stage kappa: {kappa_s:.4f}, Dominant frequency: {dominant_freq:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogy([0] + list(range(1, num_stages + 1)), [initial_residue] + mean_residues, 'o-', linewidth=2, markersize=10)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Stage', fontsize=14)
    plt.ylabel('Mean Absolute Residue', fontsize=14)
    plt.title(f'Residue Reduction over Stages for {dim}D Problem', fontsize=16)
    
    for i, residue in enumerate([initial_residue] + mean_residues):
        plt.annotate(f"{residue:.2e}", xy=(i, residue), xytext=(10, 0), textcoords="offset points", fontsize=11, ha='left')
    
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'plots/residue_reduction_dim{dim}.png', dpi=300)
    plt.close()
    
    # Prepare result
    result = {
        'dimension': dim,
        'final_residue': mean_residues[-1],
        'training_time': total_time,
        'stage_times': stage_times,
        'mean_residues': mean_residues
    }
    
    # For Poisson's equation with d dimensions, approximate L2 error is about residue / (2π²d)
    estimated_l2_error = mean_residues[-1] / (2 * np.pi**2 * dim)
    result['estimated_l2_error'] = estimated_l2_error
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Dimension: {dim}\n")
            f.write(f"Final Mean Residue: {mean_residues[-1]:.6e}\n")
            f.write(f"Estimated L2 Error: {estimated_l2_error:.6e}\n")
            f.write(f"Training Time: {total_time:.1f}s\n\n")
            
            f.write("Residue by Stage:\n")
            f.write(f"Initial: {initial_residue:.6e}\n")
            for i, residue in enumerate(mean_residues):
                f.write(f"Stage {i+1}: {residue:.6e}\n")
            
            f.write("\nTraining Time by Stage:\n")
            for i, t in enumerate(stage_times):
                f.write(f"Stage {i+1}: {t:.1f}s\n")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSNN Residue Analysis and Table Reproduction")
    parser.add_argument("--dim", type=int, required=True, help="Dimension to run")
    parser.add_argument("--output", type=str, default=None, help="Output file for detailed results")
    args = parser.parse_args()
    
    result = run_experiment(args.dim, args.output)
    print("\nTable 4.1 Entry:")
    print(f"{'Dimension d':<12} {'Mean Residue':<20} {'Est. L2 Error':<20} {'Training Time (s)':<15}")
    print("-" * 65)
    print(f"{result['dimension']:<12} {result['final_residue']:<.2e} {result['estimated_l2_error']:<.2e} {result['training_time']:<.1f}")