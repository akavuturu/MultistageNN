import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy.fft import fftfreq, fftshift, fftn
import pandas as pd
import pickle
import logging
from tqdm import tqdm
from itertools import product

# Ensure output directories exist
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/results_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)

# Import NeuralNet class and functions from utils_gpu
from utils_gpu import NeuralNet, create_ds, poisson

# Configure TensorFlow for GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use only the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU instead")

class MultistageNeuralNetwork:
    """
    Implementation of the Multistage Neural Network (MSNN) for solving PDEs.
    """
    def __init__(self, x_train, num_hidden_layers, num_hidden_nodes):
        """
        Initialize the MultistageNeuralNetwork instance.
        """
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
        self.dim = x_train.shape[-1]
        self.N = int(round(x_train.shape[0] ** (1 / self.dim)))
        self.stages = []
        self.layers = [self.dim] + ([num_hidden_nodes] * num_hidden_layers) + [1]
        self.lt = [tf.math.reduce_min(x_train[:, i]) for i in range(x_train.shape[-1])]
        self.ut = [tf.math.reduce_max(x_train[:, i]) for i in range(x_train.shape[-1])]

    def train(self, x_train, y_train, stage, kappa, iters):
        """
        Train a specific stage of the neural network.
        """
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float64)
        act = 0 if stage == 0 else 1  # Use different activation for first stage
        lt = [tf.cast(tf.math.reduce_min(x_train[:, i]).numpy(), dtype=tf.float64) for i in range(x_train.shape[-1])]
        ut = [tf.cast(tf.math.reduce_max(x_train[:, i]).numpy(), dtype=tf.float64) for i in range(x_train.shape[-1])]

        with tf.device('/GPU:0'):  # Explicitly use GPU if available
            self.stages.append(NeuralNet(x_train, y_train, self.layers, kappa=kappa, lt=lt, ut=ut, acts=act))
            self.stages[stage].train(iters[0], 1)  # Train using Adam optimizer
            self.stages[stage].train(iters[1], 2)  # Train using L-BFGS optimizer
    
    def predict(self, x_train):
        """
        Make predictions using all stages of the neural network.
        """
        pred = tf.add_n([self.stages[j].predict(x_train) for j in range(len(self.stages))])
        return pred

    @staticmethod
    def fftn_(x_train, residue):
        """
        Perform FFT on residue to determine dominant frequency for kappa scaling.
        """
        dim = x_train.shape[-1]
        N_train = int(round(x_train.shape[0] ** (1 / dim)))
        g = residue.numpy()

        GG = g.reshape([N_train] * dim)
        G = fftn(GG)
        G_shifted = fftshift(G)

        N = len(G)
        total_time_range = 2
        sample_rate = N / total_time_range

        half_N = N // 2
        T = 1.0 / sample_rate
        idxs = tuple(slice(half_N, N, 1) for _ in range(dim))
        G_pos = G_shifted[idxs]

        freqs = [fftshift(fftfreq(GG.shape[i], d=T)) for i in range(len(GG.shape))]
        freq_pos = [freqs[i][half_N:] for i in range(len(freqs))]

        magnitude_spectrum = np.abs(G_pos)
        max_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        dominant_freqs = [freq_pos[i][max_idx[i]] for i in range(len(freq_pos))]

        dominant_freq = max(dominant_freqs)
        kappa_f = 2 * np.pi * dominant_freq if dominant_freq > 0 else 2 * np.pi * 0.01
        
        return kappa_f, dominant_freq
    
    @staticmethod
    def sfftn(x_train, residue, sparsity_threshold=0.01, k=None):
        """
        Sparse FFT for high-dimensional data to determine kappa scaling.
        """
        dim = x_train.shape[-1]
        N_train = int(round(x_train.shape[0] ** (1 / dim)))
        g = residue.numpy().flatten()
        
        grid = g.reshape([N_train] * dim)

        # Downsample for higher dimensions
        downsample_factor = 1
        if dim > 4:
            downsample_factor = max(1, N_train // 16)

        slices = tuple(slice(None, None, downsample_factor) for _ in range(dim))
        GG = grid[slices]
        G = fftn(GG)
        G_shifted = fftshift(G)
        
        N = GG.shape[0]
        total_time_range = 2
        sample_rate = N / total_time_range
        half_N = N // 2
        T = 1.0 / sample_rate
        
        idxs = tuple(slice(half_N, N, 1) for _ in range(dim))
        G_pos = G_shifted[idxs]
        
        freqs = [fftshift(fftfreq(N, d=T)) for _ in range(dim)]
        freq_pos = [freqs[i][half_N:] for i in range(dim)]

        magnitude_spectrum = np.abs(G_pos)
        max_magnitude = np.max(magnitude_spectrum)

        # Apply sparsity threshold
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
        
        dominant_freq = max(dominant_freqs)
        kappa_f = 2 * np.pi * dominant_freq if dominant_freq > 0 else 2 * np.pi * 0.01
        
        if dim <= 2:
            sparse_spectrum = sparse_magnitude
        else:
            sparse_idx = np.where(mask)
            values = magnitude_spectrum[sparse_idx]
            sparse_spectrum = (sparse_idx, values, magnitude_spectrum.shape)
        
        return kappa_f, dominant_freq, sparse_spectrum

def analytical_solution(x):
    """
    Analytical solution to the d-dimensional Poisson equation:
    u(x) = prod(sin(pi*x_i))
    """
    return tf.reduce_prod(tf.math.sin(np.pi * x), axis=1)

def compute_error(model, x_test, exact_solution):
    """
    Compute relative L2 and L-infinity errors.
    """
    y_pred = model.predict(x_test)
    y_true = exact_solution(x_test)
    y_true = tf.reshape(y_true, y_pred.shape)
    
    abs_error = tf.abs(y_pred - y_true)
    l2_error = tf.sqrt(tf.reduce_mean(tf.square(abs_error))) / tf.sqrt(tf.reduce_mean(tf.square(y_true)))
    linf_error = tf.reduce_max(abs_error) / tf.reduce_max(tf.abs(y_true))
    
    return l2_error.numpy(), linf_error.numpy()

def save_model(model, file_path):
    """
    Save MSNN model to a pickle file.
    """
    model_data = {
        'dim': model.dim,
        'N': model.N,
        'layers': model.layers,
        'lt': [lt.numpy() for lt in model.lt],
        'ut': [ut.numpy() for ut in model.ut],
        'stages': []
    }

    for stage in model.stages:
        stage_data = {
            'weights': [w.numpy() for w in stage.weights],
            'biases': [b.numpy() for b in stage.biases],
            'lt': stage.lt,
            'ut': stage.ut,
            'kappa': stage.kappa,
            'acts': stage.actv,
            'loss': stage.loss
        }
        model_data['stages'].append(stage_data)

    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(file_path, x_train, y_train):
    """
    Load MSNN model from a pickle file.
    """
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = MultistageNeuralNetwork(x_train, len(model_data['layers']) - 2, model_data['layers'][1])
    model.dim = model_data['dim']
    model.N = model_data['N']
    model.layers = model_data['layers']
    model.lt = model_data['lt']
    model.ut = model_data['ut']

    for stage_data in model_data['stages']:
        nn = NeuralNet(x_train, y_train, layers=model.layers, kappa=stage_data['kappa'],
            lt=stage_data['lt'], ut=stage_data['ut'], acts=stage_data['acts']
        )
        nn.weights = [tf.Variable(w, dtype=tf.float64) for w in stage_data['weights']]
        nn.biases = [tf.Variable(b, dtype=tf.float64) for b in stage_data['biases']]
        nn.loss = stage_data['loss']
        model.stages.append(nn)

    return model

def train_or_load_model(dim, num_hidden_layers=3, num_hidden_nodes=20, points_per_dim=15, L=2.04, num_stages=5, pretrained_weights_path=None):
    """
    Train a new model or load a pre-trained one.
    """
    N_train = points_per_dim ** dim
    x_train = create_ds(dim, -L/2, L/2, N_train)
    y_train = tf.reshape(poisson(x_train), [len(x_train), 1])
    
    # Define training parameters
    training_iters = [(3000, 6000)] + [(5000, 8000*i) for i in range(2, num_stages+1)]
    
    # Try to load pre-trained model if a path is provided
    if pretrained_weights_path:
        try:
            logging.info(f"Loading pre-trained model for {dim}D from {pretrained_weights_path}")
            return load_model(pretrained_weights_path, x_train, y_train)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logging.warning(f"Could not load pre-trained model: {e}. Training a new one.")
    
    # Train a new model
    logging.info(f"Training new {dim}D model with {points_per_dim} points per dimension")
    MSNN = MultistageNeuralNetwork(x_train, num_hidden_layers, num_hidden_nodes)
    
    # Train stage 0
    start_time = time.time()
    MSNN.train(x_train, y_train, stage=0, kappa=1, iters=training_iters[0])
    curr_residue = y_train - MSNN.predict(x_train)
    mean_residue = tf.reduce_mean(tf.abs(curr_residue))
    
    # Train additional stages
    for i in range(1, num_stages):
        kappa_s, _, _ = MultistageNeuralNetwork.sfftn(x_train, curr_residue)
        logging.info(f"Training stage {i+1} with kappa={kappa_s:.4f}")
        
        MSNN.train(x_train, curr_residue, stage=i, kappa=kappa_s, iters=training_iters[i])
        curr_residue = y_train - MSNN.predict(x_train)
        mean_residue = tf.reduce_mean(tf.abs(curr_residue))
        logging.info(f"Stage {i+1} residue: {mean_residue:.6e}")
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f}s")
    
    return MSNN

def generate_1d_reproduction_data():
    """
    Reproduce the 1D results from Wang and Lai (2024).
    """
    logging.info("Generating 1D reproduction data")
    dim = 1
    points_per_dim = 50  # Higher resolution for 1D
    num_stages = 5
    
    # Use pre-trained model if available
    model_path = "./models/msnn_1d_pretrained.pkl"
    msnn_1d = train_or_load_model(dim, points_per_dim=points_per_dim, num_stages=num_stages, 
                                pretrained_weights_path=model_path)
    
    # Generate data for convergence plot (Figure 1)
    x_test = create_ds(dim, -1.0, 1.0, 200)
    stage_errors = []
    
    # Calculate error after each stage
    for s in range(1, len(msnn_1d.stages) + 1):
        partial_msnn = MultistageNeuralNetwork(x_test, 3, 20)
        partial_msnn.stages = msnn_1d.stages[:s]
        l2_error, linf_error = compute_error(partial_msnn, x_test, analytical_solution)
        stage_errors.append((s, l2_error, linf_error))
    
    # Create convergence plot
    plt.figure(figsize=(10, 6))
    stages, l2_errors, _ = zip(*stage_errors)
    plt.semilogy(stages, l2_errors, 'o-', linewidth=2, markersize=8)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Stage', fontsize=14)
    plt.ylabel('Relative $L^2$ Error', fontsize=14)
    plt.title('Convergence of MSNN for 1D Poisson Equation', fontsize=16)
    plt.savefig('plots/figure1_1d_convergence.png', dpi=300, bbox_inches='tight')
    
    # Save the data
    np.savetxt('data/1d_stage_errors.csv', np.array(stage_errors), delimiter=',', 
               header='stage,l2_error,linf_error')
    
    logging.info(f"1D reproduction complete: final L2 error = {l2_errors[-1]:.6e}")
    return stage_errors

def generate_high_dimensional_data():
    """
    Generate data for high-dimensional tests.
    """
    logging.info("Generating high-dimensional scaling data")
    dimensions = [1, 2, 5, 10, 25, 50, 100]
    results = []
    
    for dim in dimensions:
        logging.info(f"Processing dimension {dim}")
        # Adjust points per dimension for higher dimensions
        points_per_dim = 15
        if dim > 25:
            points_per_dim = 10
        if dim > 50:
            points_per_dim = 8
            
        # Use pre-trained models for specific dimensions if available
        model_path = None
        if dim in [1, 5, 8, 10]:
            model_path = f"./models/msnn_{dim}d_pretrained.pkl"
        
        # Train or load the model
        start_time = time.time()
        msnn = train_or_load_model(dim, points_per_dim=points_per_dim, 
                                  pretrained_weights_path=model_path)
        training_time = time.time() - start_time
        
        # Generate test points
        test_points_per_dim = max(5, min(10, points_per_dim))
        N_test = test_points_per_dim ** dim
        x_test = create_ds(dim, -1.0, 1.0, N_test)
        
        # Compute errors
        l2_error, linf_error = compute_error(msnn, x_test, analytical_solution)
        
        # Estimate epochs to convergence
        total_epochs = sum(len(stage.loss) for stage in msnn.stages)
        
        results.append({
            'dimension': dim,
            'l2_error': l2_error,
            'linf_error': linf_error,
            'training_time': training_time,
            'epochs_to_converge': total_epochs,
            'points_per_dim': points_per_dim
        })
        
        logging.info(f"Dimension {dim}: L2 error = {l2_error:.6e}, training time = {training_time:.2f}s")
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/high_dim_results.csv', index=False)
    
    # Generate Figure 2: Error vs Dimension
    plt.figure(figsize=(10, 6))
    plt.loglog(results_df['dimension'], results_df['l2_error'], 'o-', label='$L^2$ Error', linewidth=2)
    plt.loglog(results_df['dimension'], results_df['linf_error'], 's--', label='$L^\\infty$ Error', linewidth=2)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Dimension $d$', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('MSNN Error Scaling with Dimension', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('plots/figure2_error_vs_dimension.png', dpi=300, bbox_inches='tight')
    
    # Generate Figure 3: Training time vs Dimension
    plt.figure(figsize=(10, 6))
    plt.loglog(results_df['dimension'], results_df['training_time'], 'o-', linewidth=2, markersize=8)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Dimension $d$', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=14)
    plt.title('MSNN Training Time Scaling with Dimension', fontsize=16)
    plt.savefig('plots/figure3_training_time_vs_dimension.png', dpi=300, bbox_inches='tight')
    
    logging.info("High-dimensional scaling data generation complete")
    return results_df

def analyze_kappa_scaling_effect():
    """
    Analyze the effect of kappa-scaling on convergence.
    """
    logging.info("Analyzing kappa-scaling effect")
    dim = 10  # Use 10D for this analysis
    points_per_dim = 15
    num_stages = 5
    
    # Create dataset
    N_train = points_per_dim ** dim
    x_train = create_ds(dim, -1.0, 1.0, N_train)
    y_train = tf.reshape(poisson(x_train), [len(x_train), 1])
    
    # Training iterations
    training_iters = [(3000, 6000)] + [(5000, 8000*i) for i in range(2, num_stages+1)]
    
    # Try to load pre-trained models
    model_path_with_kappa = "./models/msnn_10d_with_kappa.pkl"
    model_path_without_kappa = "./models/msnn_10d_without_kappa.pkl"
    
    # Model with kappa-scaling
    try:
        msnn_with_kappa = load_model(model_path_with_kappa, x_train, y_train)
        logging.info("Loaded pre-trained model with kappa-scaling")
    except (FileNotFoundError, pickle.UnpicklingError):
        logging.info("Training new model with kappa-scaling")
        msnn_with_kappa = MultistageNeuralNetwork(x_train, 3, 20)
        
        # Train stage 0
        msnn_with_kappa.train(x_train, y_train, stage=0, kappa=1, iters=training_iters[0])
        
        # Train additional stages with kappa-scaling
        for i in range(1, num_stages):
            curr_residue = y_train - msnn_with_kappa.predict(x_train)
            kappa_s, _, _ = MultistageNeuralNetwork.sfftn(x_train, curr_residue)
            msnn_with_kappa.train(x_train, curr_residue, stage=i, kappa=kappa_s, iters=training_iters[i])
        
        save_model(msnn_with_kappa, model_path_with_kappa)
    
    # Model without kappa-scaling
    try:
        msnn_without_kappa = load_model(model_path_without_kappa, x_train, y_train)
        logging.info("Loaded pre-trained model without kappa-scaling")
    except (FileNotFoundError, pickle.UnpicklingError):
        logging.info("Training new model without kappa-scaling")
        msnn_without_kappa = MultistageNeuralNetwork(x_train, 3, 20)
        
        # Train stage 0
        msnn_without_kappa.train(x_train, y_train, stage=0, kappa=1, iters=training_iters[0])
        
        # Train additional stages without kappa-scaling (kappa=1)
        for i in range(1, num_stages):
            curr_residue = y_train - msnn_without_kappa.predict(x_train)
            msnn_without_kappa.train(x_train, curr_residue, stage=i, kappa=1.0, iters=training_iters[i])
        
        save_model(msnn_without_kappa, model_path_without_kappa)
    
    # Extract loss histories
    loss_with_kappa = np.concatenate([stage.loss for stage in msnn_with_kappa.stages])
    loss_without_kappa = np.concatenate([stage.loss for stage in msnn_without_kappa.stages])
    
    # Create Figure 4: Convergence with and without kappa-scaling
    plt.figure(figsize=(10, 6))
    plt.semilogy(np.arange(len(loss_with_kappa)), loss_with_kappa, 'b-', 
                 label='With $\\kappa$-scaling', linewidth=2)
    plt.semilogy(np.arange(len(loss_without_kappa)), loss_without_kappa, 'r--', 
                 label='Without $\\kappa$-scaling', linewidth=2)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Effect of $\\kappa$-scaling on Convergence (10D)', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('plots/figure4_kappa_scaling_effect.png', dpi=300, bbox_inches='tight')
    
    # Compare final errors
    x_test = create_ds(dim, -1.0, 1.0, points_per_dim**dim)
    l2_with_kappa, linf_with_kappa = compute_error(msnn_with_kappa, x_test, analytical_solution)
    l2_without_kappa, linf_without_kappa = compute_error(msnn_without_kappa, x_test, analytical_solution)
    
    kappa_comparison = {
        'method': ['With κ-scaling', 'Without κ-scaling'],
        'l2_error': [l2_with_kappa, l2_without_kappa],
        'linf_error': [linf_with_kappa, linf_without_kappa]
    }
    
    pd.DataFrame(kappa_comparison).to_csv('data/kappa_scaling_comparison.csv', index=False)
    
    logging.info(f"Kappa-scaling analysis complete: with kappa L2={l2_with_kappa:.6e}, "
                 f"without kappa L2={l2_without_kappa:.6e}")
    
    return kappa_comparison

def generate_summary_table():
    """
    Generate the summary table (Table 1) with performance metrics.
    """
    logging.info("Generating summary table")
    
    # Load results if available
    try:
        results_df = pd.read_csv('data/high_dim_results.csv')
        logging.info("Loaded existing high-dimensional results")
    except FileNotFoundError:
        results_df = generate_high_dimensional_data()
    
    # Create LaTeX table
    with open('data/summary_table.tex', 'w') as f:
        f.write('\\begin{table}[h]\n')
        f.write('\\centering\n')
        f.write('\\caption{MSNN performance metrics across different dimensions.}\n')
        f.write('\\label{tab:summary_results}\n')
        f.write('\\begin{tabular}{c|c|c|c|c}\n')
        f.write('Dimension $d$ & Rel. $L^2$ Error & Rel. $L^\\infty$ Error & Training Time (s) & Epochs to Converge \\\\\n')
        f.write('\\hline\n')
        
        for _, row in results_df.iterrows():
            f.write(f"{int(row['dimension'])} & "
                   f"{row['l2_error']:.1e} & "
                   f"{row['linf_error']:.1e} & "
                   f"{int(row['training_time'])} & "
                   f"{int(row['epochs_to_converge'])} \\\\\n")
        
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    
    # Also create a CSV version for easier reading
    results_df.to_csv('data/summary_table.csv', index=False)
    
    logging.info("Summary table generation complete")
    return results_df

def main():
    """
    Main function to generate all results for the paper.
    """
    logging.info("Starting results generation")
    
    # Figure 1: 1D Reproduction Results
    stage_errors = generate_1d_reproduction_data()
    
    # Figure 2 & 3: High-dimensional Scaling
    high_dim_results = generate_high_dimensional_data()
    
    # Figure 4: Effect of kappa-scaling
    kappa_comparison = analyze_kappa_scaling_effect()
    
    # Table 1: Summary of Performance Trends
    summary_results = generate_summary_table()
    
    logging.info("All results generation complete")
    
    # Print summary of key results
    print("\nKey Results Summary:")
    print("-" * 50)
    print(f"1D Reproduction: Final L2 error = {stage_errors[-1][1]:.6e}")
    print("\nScaling to Higher Dimensions:")
    for dim in [1, 10, 50, 100]:
        row = high_dim_results[high_dim_results['dimension'] == dim].iloc[0]
        print(f"  {dim}D: L2 error = {row['l2_error']:.6e}, Training time = {row['training_time']:.1f}s")
    
    print("\nEffect of κ-scaling (10D):")
    print(f"  With κ-scaling: L2 error = {kappa_comparison['l2_error'][0]:.6e}")
    print(f"  Without κ-scaling: L2 error = {kappa_comparison['l2_error'][1]:.6e}")
    print(f"  Improvement factor: {kappa_comparison['l2_error'][1]/kappa_comparison['l2_error'][0]:.2f}x")
    
    print("\nAll plots saved to plots/ directory")
    print("All data tables saved to data/ directory")
    print("-" * 50)

if __name__ == "__main__":
    main()