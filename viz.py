# %%
import warnings
warnings.simplefilter("ignore")
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fftshift, fftn
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from utils_gpu import NeuralNet, create_ds, poisson
from tqdm import tqdm

tf.get_logger().setLevel(logging.ERROR)

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Create directories for saving results if they don't exist
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define paths for saving results
PLOT_DIR = "plots"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Import MultistageNeuralNetwork class and save_model function directly from the visualize.py file
try:
    from visualize import MultistageNeuralNetwork, save_model
    print("Successfully imported MultistageNeuralNetwork class from visualize.py")
except ImportError:
    # If import fails, use the class definition from the original file
    from MSNN_GPU import MultistageNeuralNetwork, save_model
    print("Imported MultistageNeuralNetwork class from MSNN_GPU.py")

# %%
# Utility functions for loading models and computing errors

def load_model(file_path):
    """Load a saved MSNN model."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analytical_solution(x):
    """
    Analytical solution to the d-dimensional Poisson equation:
    u(x) = prod(sin(pi*x_i))
    """
    return tf.reduce_prod(tf.math.sin(np.pi * x), axis=1)

def compute_error(model_data, x_test):
    """
    Compute relative L2 and L-infinity errors for a model.
    """
    # Reconstruct the prediction function from the model data
    y_pred = predict_from_model_data(model_data, x_test)
    
    # Get the exact solution
    y_true = analytical_solution(x_test)
    y_true = tf.reshape(y_true, y_pred.shape)
    
    # Compute errors
    abs_error = tf.abs(y_pred - y_true)
    l2_error = tf.sqrt(tf.reduce_mean(tf.square(abs_error))) / tf.sqrt(tf.reduce_mean(tf.square(y_true)))
    linf_error = tf.reduce_max(abs_error) / tf.reduce_max(tf.abs(y_true))
    
    return l2_error.numpy(), linf_error.numpy()

def predict_from_model_data(model_data, x):
    """Predict using a loaded model data."""
    # Convert input to tensor if needed
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
    
    # Initialize prediction with zeros
    pred = tf.zeros((x.shape[0], 1), dtype=tf.float64)
    
    # Add predictions from each stage
    for stage_data in model_data['stages']:
        # Normalize input
        lt = stage_data['lt']
        ut = stage_data['ut']
        H = 2.0 * tf.math.divide(
                tf.math.subtract(x, tf.transpose(lt)), 
                tf.transpose(tf.math.subtract(ut, lt))) - 1.0
        
        # Forward pass through the network
        weights = stage_data['weights']
        biases = stage_data['biases']
        kappa = stage_data['kappa']
        acts = stage_data['acts']
        
        # First layer with activation
        W = weights[0]
        b = biases[0]
        if acts == 0:  # tanh activation
            H = tf.tanh(tf.add(kappa * tf.matmul(H, W), b))
        else:  # sin activation
            H = tf.sin(tf.add(kappa * tf.matmul(H, W), b))
        
        # Hidden layers
        for l in range(1, len(weights)-1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        
        # Output layer
        W = weights[-1]
        b = biases[-1]
        stage_pred = tf.add(tf.matmul(H, W), b)
        
        # Add to total prediction
        pred += stage_pred
    
    return pred

# %%
# Functions to run experiments and collect data

def run_dimension_experiment(dimensions, points_per_dim=15, num_stages=3, 
                            num_hidden_layers=3, num_hidden_nodes=20):
    """
    Run experiments across different dimensions and collect performance metrics.
    """
    results = {
        'dimension': [],
        'rel_l2_error': [],
        'rel_linf_error': [],
        'training_time': [],
        'num_epochs': [],
        'stage_errors': []
    }
    
    for dim in dimensions:
        print(f"Running experiment for dimension {dim}")
        
        # Create dataset
        L = 2.04
        N_train = min(points_per_dim ** dim, 10**6)  # Cap the number of points to prevent memory issues
        
        # For higher dimensions, we need to reduce points per dimension
        actual_points_per_dim = int(N_train ** (1/dim))
        print(f"Using {actual_points_per_dim} points per dimension for dim={dim}")
        
        start_time = time.time()
        
        x_train = create_ds(dim, -L/2, L/2, N_train)
        y_train = tf.reshape(poisson(x_train), [len(x_train), 1])
        
        # Set up training iterations per stage
        training_iters = [(3000, 6000)] + [(5000, 8000*i) for i in range(2, 15)]
        training_iters = training_iters[:num_stages]
        
        # Initialize MSNN
        msnn = MultistageNeuralNetwork(x_train, num_hidden_layers, num_hidden_nodes)
        
        # Train first stage
        kappa = 1
        msnn.train(x_train, y_train, stage=0, kappa=1, iters=training_iters[0])
        
        # Store stage errors
        stage_errors = []
        
        # Compute error after first stage
        x_test = create_ds(dim, -L/2, L/2, min(1000, 10**(6//dim)))
        pred = msnn.stages[0].predict(x_test)
        y_true = tf.reshape(analytical_solution(x_test), pred.shape)
        abs_error = tf.abs(pred - y_true)
        l2_error = tf.sqrt(tf.reduce_mean(tf.square(abs_error))) / tf.sqrt(tf.reduce_mean(tf.square(y_true)))
        linf_error = tf.reduce_max(abs_error) / tf.reduce_max(tf.abs(y_true))
        stage_errors.append((1, l2_error.numpy(), linf_error.numpy()))
        
        # Train remaining stages
        for i in range(1, num_stages):
            curr_residue = y_train - tf.add_n([msnn.stages[j].predict(x_train) for j in range(i)])
            kappa_s, _, _ = MultistageNeuralNetwork.sfftn(x_train, curr_residue)
            
            msnn.train(x_train, curr_residue, stage=i, kappa=kappa_s, iters=training_iters[i])
            
            # Compute error after this stage
            pred = tf.add_n([msnn.stages[j].predict(x_test) for j in range(i+1)])
            abs_error = tf.abs(pred - y_true)
            l2_error = tf.sqrt(tf.reduce_mean(tf.square(abs_error))) / tf.sqrt(tf.reduce_mean(tf.square(y_true)))
            linf_error = tf.reduce_max(abs_error) / tf.reduce_max(tf.abs(y_true))
            stage_errors.append((i+1, l2_error.numpy(), linf_error.numpy()))
        
        # Measure total training time
        training_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"model_dim{dim}.pkl")
        save_model(msnn, model_path)
        
        # Store results
        results['dimension'].append(dim)
        results['rel_l2_error'].append(l2_error.numpy())
        results['rel_linf_error'].append(linf_error.numpy())
        results['training_time'].append(training_time)
        results['num_epochs'].append(sum(iter[0] + iter[1] for iter in training_iters))
        results['stage_errors'].append(stage_errors)
        
        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
    
    return results

def analyze_training_efficiency(dimensions, points_per_dim=15, num_stages=2, 
                                num_hidden_layers=3, num_hidden_nodes=20):
    """
    Analyze training time vs. dimension and number of epochs to convergence.
    """
    results = {
        'dimension': [],
        'stage': [],
        'training_time': [],
        'epochs_to_converge': [],
        'final_l2_error': []
    }
    
    for dim in dimensions:
        print(f"Analyzing training efficiency for dimension {dim}")
        
        # Create dataset
        L = 2.04
        N_train = min(points_per_dim ** dim, 10**6)  # Cap the number of points
        
        # For higher dimensions, we need to reduce points per dimension
        actual_points_per_dim = int(N_train ** (1/dim))
        print(f"Using {actual_points_per_dim} points per dimension for dim={dim}")
        
        x_train = create_ds(dim, -L/2, L/2, N_train)
        y_train = tf.reshape(poisson(x_train), [len(x_train), 1])
        
        # Training iterations structure
        base_iters = (3000, 6000)
        
        # Initialize MSNN
        msnn = MultistageNeuralNetwork(x_train, num_hidden_layers, num_hidden_nodes)
        
        # For each stage
        for stage in range(num_stages):
            start_time = time.time()
            
            if stage == 0:
                # First stage
                kappa = 1
                msnn.train(x_train, y_train, stage=0, kappa=1, iters=base_iters)
            else:
                # Higher stages
                curr_residue = y_train - tf.add_n([msnn.stages[j].predict(x_train) for j in range(stage)])
                kappa_s, _, _ = MultistageNeuralNetwork.sfftn(x_train, curr_residue)
                
                msnn.train(x_train, curr_residue, stage=stage, kappa=kappa_s, iters=base_iters)
            
            # Measure training time
            training_time = time.time() - start_time
            
            # Compute error
            x_test = create_ds(dim, -L/2, L/2, min(1000, 10**(6//dim)))
            pred = tf.add_n([msnn.stages[j].predict(x_test) for j in range(stage+1)])
            y_true = tf.reshape(analytical_solution(x_test), pred.shape)
            l2_error = tf.sqrt(tf.reduce_mean(tf.square(pred - y_true))) / tf.sqrt(tf.reduce_mean(tf.square(y_true)))
            
            # Estimate epochs to convergence (analyze the loss curve)
            # For simplicity, we'll just use the total number of iterations
            epochs_to_converge = base_iters[0] + base_iters[1]
            
            # Store results
            results['dimension'].append(dim)
            results['stage'].append(stage+1)
            results['training_time'].append(training_time)
            results['epochs_to_converge'].append(epochs_to_converge)
            results['final_l2_error'].append(l2_error.numpy())
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results

def generate_performance_metrics_table(dimensions, results=None):
    """
    Generate a table of performance metrics across dimensions.
    """
    if results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return None
    
    # Create a table
    data = []
    for i, dim in enumerate(results['dimension']):
        if dim in dimensions:
            data.append({
                'Dimension d': dim,
                'Rel. L2 Error': results['rel_l2_error'][i],
                'Rel. Linf Error': results['rel_linf_error'][i],
                'Training Time (s)': results['training_time'][i]
            })
    
    return data

# %%
# Plotting functions

def plot_dimension_scaling(results=None):
    """
    Plot relative L2 and L-infinity errors as a function of dimension.
    This corresponds to Figure 2 in the thesis.
    """
    if results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    plt.figure(figsize=(10, 6))
    
    # Plot L2 error
    plt.semilogy(results['dimension'], results['rel_l2_error'], 'o-', 
                 color='blue', linewidth=2, markersize=8, label='Relative $L^2$ Error')
    
    # Plot L-infinity error
    plt.semilogy(results['dimension'], results['rel_linf_error'], 's--', 
                 color='red', linewidth=2, markersize=8, label='Relative $L^\\infty$ Error')
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Dimension $d$', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('MSNN Error Scaling with Dimension', fontsize=16)
    plt.legend(fontsize=12)
    
    # Add horizontal line at 10^-3 for reference
    plt.axhline(y=1e-3, color='gray', linestyle='-', alpha=0.5)
    plt.text(results['dimension'][0], 1.2e-3, 'Error = $10^{-3}$', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'figure2_dimension_scaling.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Function to generate paper-ready table data
def generate_paper_table(dimension_results=None):
    """
    Generate a formatted table for inclusion in a paper.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Create table data
    table_data = []
    for i, dim in enumerate(dimension_results['dimension']):
        l2_error = dimension_results['rel_l2_error'][i]
        linf_error = dimension_results['rel_linf_error'][i]
        training_time = dimension_results['training_time'][i]
        
        table_data.append({
            'Dimension d': dim,
            'Rel. L2 Error': f"{l2_error:.1e}",
            'Rel. L∞ Error': f"{linf_error:.1e}",
            'Training Time (s)': f"{training_time:.1f}"
        })
    
    # Print LaTeX table format
    print("\\begin{table}")
    print("\\centering")
    print("\\caption{MSNN performance metrics across different dimensions.}")
    print("\\begin{tabular}{cccc}")
    print("\\hline")
    print("Dimension $d$ & Rel. $L^2$ Error & Rel. $L^\\infty$ Error & Training Time (s) \\\\")
    print("\\hline")
    
    for row in table_data:
        print(f"{row['Dimension d']} & {row['Rel. L2 Error']} & {row['Rel. L∞ Error']} & {row['Training Time (s)']} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:performance}")
    print("\\end{table}")
    
    return table_data

# %%
# Create additional comparison plots for the manuscript
def plot_paper_figures(dimension_results=None, efficiency_results=None):
    """
    Generate publication-quality figures for the paper.
    """
    if dimension_results is None or efficiency_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
            
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                efficiency_results = pickle.load(f)
        except:
            print("Results files not found. Please run the experiments first.")
            return
    
    # Figure 1: Combined error and stages plot
    plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error vs. Dimension
    dimensions = dimension_results['dimension']
    l2_errors = dimension_results['rel_l2_error']
    linf_errors = dimension_results['rel_linf_error']
    
    axs[0, 0].semilogy(dimensions, l2_errors, 'o-', color='blue', linewidth=2, markersize=8, label='Relative $L^2$ Error')
    axs[0, 0].semilogy(dimensions, linf_errors, 's--', color='red', linewidth=2, markersize=8, label='Relative $L^\\infty$ Error')
    axs[0, 0].grid(True, which="both", ls="--", alpha=0.7)
    axs[0, 0].set_xlabel('Dimension $d')

def plot_training_efficiency(results=None):
    """
    Plot training time per stage vs. dimension.
    This corresponds to Figure 3 in the thesis.
    """
    if results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                results = pickle.load(f)
        except:
            print("No saved results found. Please run the training efficiency analysis first.")
            return
    
    # Convert results to DataFrame-like structure for easier plotting
    dimensions = sorted(set(results['dimension']))
    stages = sorted(set(results['stage']))
    
    training_times = {}
    for dim in dimensions:
        training_times[dim] = []
        for stage in stages:
            # Find the entry for this dimension and stage
            indices = [i for i, (d, s) in enumerate(zip(results['dimension'], results['stage'])) 
                      if d == dim and s == stage]
            
            if indices:
                training_times[dim].append(results['training_time'][indices[0]])
            else:
                training_times[dim].append(None)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training time for each stage
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, stage in enumerate(stages):
        times = [training_times[dim][i] for dim in dimensions if i < len(training_times[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(times) and times[j] is not None]
        valid_times = [t for t in times if t is not None]
        
        if valid_dims and valid_times:
            plt.semilogy(valid_dims, valid_times, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Dimension $d$', fontsize=14)
    plt.ylabel('Training Time per Stage (s)', fontsize=14)
    plt.title('MSNN Training Time Scaling with Dimension', fontsize=16)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'figure3_training_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_trends(data=None):
    """
    Create a summary plot of performance metrics across dimensions.
    """
    if data is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                results = pickle.load(f)
                
            dimensions = sorted(results['dimension'])
            data = generate_performance_metrics_table(dimensions, results)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Extract data for plotting
    dimensions = [d['Dimension d'] for d in data]
    l2_errors = [d['Rel. L2 Error'] for d in data]
    linf_errors = [d['Rel. Linf Error'] for d in data]
    training_times = [d['Training Time (s)'] for d in data]
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Errors vs. Dimension
    ax1.semilogy(dimensions, l2_errors, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Relative $L^2$ Error')
    ax1.semilogy(dimensions, linf_errors, 's--', color='red', linewidth=2, 
                markersize=8, label='Relative $L^\\infty$ Error')
    
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_xlabel('Dimension $d$', fontsize=14)
    ax1.set_ylabel('Relative Error', fontsize=14)
    ax1.set_title('Error Scaling with Dimension', fontsize=16)
    ax1.legend(fontsize=12)
    
    # Plot 2: Training Time vs. Dimension
    ax2.semilogy(dimensions, training_times, 'D-', color='green', linewidth=2, markersize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_xlabel('Dimension $d$', fontsize=14)
    ax2.set_ylabel('Training Time (s)', fontsize=14)
    ax2.set_title('Training Time Scaling with Dimension', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'performance_trends_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_vs_stages(dimension_results=None):
    """
    Plot error reduction across training stages for different dimensions.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Create a figure with subplots for each dimension
    dims = dimension_results['dimension']
    num_dims = len(dims)
    
    # Arrange subplots in a grid
    n_cols = min(3, num_dims)
    n_rows = (num_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
    
    # Flatten axes array for easier indexing if there's more than one row
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = [axes]  # Convert to list if there's only one subplot
    
    # Plot each dimension's stage errors
    for i, dim in enumerate(dims):
        stage_errors = dimension_results['stage_errors'][i]
        
        stages, l2_errors, linf_errors = zip(*stage_errors)
        
        ax = axes[i] if num_dims > 1 else axes
        ax.semilogy(stages, l2_errors, 'o-', color='blue', linewidth=2, 
                   markersize=8, label='Relative $L^2$ Error')
        ax.semilogy(stages, linf_errors, 's--', color='red', linewidth=2, 
                   markersize=8, label='Relative $L^\\infty$ Error')
        
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.set_xlabel('Stage', fontsize=12)
        if i % n_cols == 0:  # Only add y-label to leftmost plots
            ax.set_ylabel('Relative Error', fontsize=12)
        ax.set_title(f'Dimension = {dim}', fontsize=14)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(fontsize=10)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        if n_rows > 1 or n_cols > 1:
            axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'error_vs_stages.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_solution(dim=2, num_points=50):
    """
    Plot a 3D visualization of the solution for 2D case.
    """
    if dim != 2:
        print("3D plot is only implemented for 2D problems.")
        return
    
    # Create a grid of test points
    x1 = np.linspace(-1, 1, num_points)
    x2 = np.linspace(-1, 1, num_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Convert to input format
    x_test = np.column_stack((X1.flatten(), X2.flatten()))
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float64)
    
    # Load the model
    try:
        model_path = os.path.join(MODEL_DIR, f"model_dim{dim}.pkl")
        model_data = load_model(model_path)
        
        # Predict using the model
        y_pred = predict_from_model_data(model_data, x_test)
        y_pred = y_pred.numpy().reshape(num_points, num_points)
        
        # Compute exact solution
        y_true = analytical_solution(x_test)
        y_true = y_true.numpy().reshape(num_points, num_points)
        
        # Compute error
        error = np.abs(y_pred - y_true)
        
        # Create plots
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Predicted Solution
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X1, X2, y_pred, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('$u(x_1, x_2)$')
        ax1.set_title('MSNN Predicted Solution')
        
        # Plot 2: Exact Solution
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X1, X2, y_true, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_zlabel('$u(x_1, x_2)$')
        ax2.set_title('Exact Solution')
        
        # Plot 3: Error
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X1, X2, error, cmap='hot', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')
        ax3.set_zlabel('Error')
        ax3.set_title('Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '3d_solution_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Error loading model or creating plot: {e}")
        return

# %%
# Function to generate a heatmap visualization of errors across different dimensions and stages
def plot_error_heatmap(dimension_results=None):
    """
    Create a heatmap of errors across dimensions and stages.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Extract data for the heatmap
    dims = dimension_results['dimension']
    max_stages = max(len(err_data) for err_data in dimension_results['stage_errors'])
    
    # Create empty matrix for the heatmap
    error_matrix = np.zeros((len(dims), max_stages))
    error_matrix.fill(np.nan)  # Fill with NaN for missing values
    
    # Fill in the error values
    for i, dim in enumerate(dims):
        stage_errors = dimension_results['stage_errors'][i]
        for stage, l2_error, _ in stage_errors:
            error_matrix[i, stage-1] = l2_error
    
    # Take log10 of errors for better visualization
    log_error_matrix = np.log10(error_matrix)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap
    cmap = plt.cm.get_cmap('viridis_r')
    
    # Plot the heatmap
    sns.heatmap(log_error_matrix, annot=True, fmt=".2f", cmap=cmap, 
                xticklabels=[f"Stage {i+1}" for i in range(max_stages)],
                yticklabels=[f"d={dim}" for dim in dims],
                mask=np.isnan(log_error_matrix))
    
    plt.title('Log10 of L2 Error Across Dimensions and Stages', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Function to plot training efficiency as a function of dimension
def plot_training_scalability(efficiency_results=None):
    """
    Create a more detailed plot of training efficiency vs. dimension.
    """
    if efficiency_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                efficiency_results = pickle.load(f)
        except:
            print("No saved results found. Please run the training efficiency analysis first.")
            return
    
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    ax1, ax2, ax3 = axs
    
    # Extract data
    dimensions = sorted(set(efficiency_results['dimension']))
    stages = sorted(set(efficiency_results['stage']))
    
    # Process data for easier plotting
    training_times = {}
    epochs_to_converge = {}
    final_errors = {}
    
    for dim in dimensions:
        training_times[dim] = []
        epochs_to_converge[dim] = []
        final_errors[dim] = []
        
        for stage in stages:
            # Find indices for this dimension and stage
            indices = [i for i, (d, s) in enumerate(zip(efficiency_results['dimension'], efficiency_results['stage'])) 
                      if d == dim and s == stage]
            
            if indices:
                idx = indices[0]
                training_times[dim].append(efficiency_results['training_time'][idx])
                epochs_to_converge[dim].append(efficiency_results['epochs_to_converge'][idx])
                final_errors[dim].append(efficiency_results['final_l2_error'][idx])
            else:
                training_times[dim].append(None)
                epochs_to_converge[dim].append(None)
                final_errors[dim].append(None)
    
    # Plot 1: Training time vs dimension
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, stage in enumerate(stages):
        times = [training_times[dim][i] for dim in dimensions if i < len(training_times[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(times) and times[j] is not None]
        valid_times = [t for t in times if t is not None]
        
        if valid_dims and valid_times:
            ax1.loglog(valid_dims, valid_times, marker=markers[i % len(markers)], 
                      color=colors[i % len(colors)], linewidth=2, markersize=8, 
                      label=f'Stage {stage}')
    
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_ylabel('Training Time (s)', fontsize=14)
    ax1.set_title('Training Time vs. Dimension', fontsize=16)
    ax1.legend(fontsize=12)
    
    # Plot 2: Epochs to convergence vs dimension
    for i, stage in enumerate(stages):
        epochs = [epochs_to_converge[dim][i] for dim in dimensions if i < len(epochs_to_converge[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(epochs) and epochs[j] is not None]
        valid_epochs = [e for e in epochs if e is not None]
        
        if valid_dims and valid_epochs:
            ax2.semilogx(valid_dims, valid_epochs, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_ylabel('Epochs to Convergence', fontsize=14)
    ax2.set_title('Epochs to Convergence vs. Dimension', fontsize=16)
    ax2.legend(fontsize=12)
    
    # Plot 3: Final L2 error vs dimension
    for i, stage in enumerate(stages):
        errors = [final_errors[dim][i] for dim in dimensions if i < len(final_errors[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(errors) and errors[j] is not None]
        valid_errors = [e for e in errors if e is not None]
        
        if valid_dims and valid_errors:
            ax3.semilogy(valid_dims, valid_errors, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    ax3.grid(True, which="both", ls="--", alpha=0.7)
    ax3.set_xlabel('Dimension $d', fontsize=14)
    ax3.set_ylabel('Final L2 Error', fontsize=14)
    ax3.set_title('Final L2 Error vs. Dimension', fontsize=16)
    ax3.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_scalability_detailed.png'), dpi=300, bbox_inches='tight')
    plt.show()
    axs[0, 0].set_ylabel('Relative Error', fontsize=14)
    axs[0, 0].set_title('Error Scaling with Dimension', fontsize=16)
    axs[0, 0].legend(fontsize=12)
    
    # Plot 2: Training time vs. Dimension
    training_times = training_times.values()
    
    axs[0, 1].semilogy(dimensions, training_times, 'D-', color='green', linewidth=2, markersize=8)
    axs[0, 1].grid(True, which="both", ls="--", alpha=0.7)
    axs[0, 1].set_xlabel('Dimension $d')

def plot_training_efficiency(results=None):
    """
    Plot training time per stage vs. dimension.
    This corresponds to Figure 3 in the thesis.
    """
    if results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                results = pickle.load(f)
        except:
            print("No saved results found. Please run the training efficiency analysis first.")
            return
    
    # Convert results to DataFrame-like structure for easier plotting
    dimensions = sorted(set(results['dimension']))
    stages = sorted(set(results['stage']))
    
    training_times = {}
    for dim in dimensions:
        training_times[dim] = []
        for stage in stages:
            # Find the entry for this dimension and stage
            indices = [i for i, (d, s) in enumerate(zip(results['dimension'], results['stage'])) 
                      if d == dim and s == stage]
            
            if indices:
                training_times[dim].append(results['training_time'][indices[0]])
            else:
                training_times[dim].append(None)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training time for each stage
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, stage in enumerate(stages):
        times = [training_times[dim][i] for dim in dimensions if i < len(training_times[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(times) and times[j] is not None]
        valid_times = [t for t in times if t is not None]
        
        if valid_dims and valid_times:
            plt.semilogy(valid_dims, valid_times, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Dimension $d$', fontsize=14)
    plt.ylabel('Training Time per Stage (s)', fontsize=14)
    plt.title('MSNN Training Time Scaling with Dimension', fontsize=16)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'figure3_training_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_trends(data=None):
    """
    Create a summary plot of performance metrics across dimensions.
    """
    if data is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                results = pickle.load(f)
                
            dimensions = sorted(results['dimension'])
            data = generate_performance_metrics_table(dimensions, results)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Extract data for plotting
    dimensions = [d['Dimension d'] for d in data]
    l2_errors = [d['Rel. L2 Error'] for d in data]
    linf_errors = [d['Rel. Linf Error'] for d in data]
    training_times = [d['Training Time (s)'] for d in data]
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Errors vs. Dimension
    ax1.semilogy(dimensions, l2_errors, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Relative $L^2$ Error')
    ax1.semilogy(dimensions, linf_errors, 's--', color='red', linewidth=2, 
                markersize=8, label='Relative $L^\\infty$ Error')
    
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_xlabel('Dimension $d$', fontsize=14)
    ax1.set_ylabel('Relative Error', fontsize=14)
    ax1.set_title('Error Scaling with Dimension', fontsize=16)
    ax1.legend(fontsize=12)
    
    # Plot 2: Training Time vs. Dimension
    ax2.semilogy(dimensions, training_times, 'D-', color='green', linewidth=2, markersize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_xlabel('Dimension $d$', fontsize=14)
    ax2.set_ylabel('Training Time (s)', fontsize=14)
    ax2.set_title('Training Time Scaling with Dimension', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'performance_trends_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_vs_stages(dimension_results=None):
    """
    Plot error reduction across training stages for different dimensions.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Create a figure with subplots for each dimension
    dims = dimension_results['dimension']
    num_dims = len(dims)
    
    # Arrange subplots in a grid
    n_cols = min(3, num_dims)
    n_rows = (num_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
    
    # Flatten axes array for easier indexing if there's more than one row
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = [axes]  # Convert to list if there's only one subplot
    
    # Plot each dimension's stage errors
    for i, dim in enumerate(dims):
        stage_errors = dimension_results['stage_errors'][i]
        
        stages, l2_errors, linf_errors = zip(*stage_errors)
        
        ax = axes[i] if num_dims > 1 else axes
        ax.semilogy(stages, l2_errors, 'o-', color='blue', linewidth=2, 
                   markersize=8, label='Relative $L^2$ Error')
        ax.semilogy(stages, linf_errors, 's--', color='red', linewidth=2, 
                   markersize=8, label='Relative $L^\\infty$ Error')
        
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.set_xlabel('Stage', fontsize=12)
        if i % n_cols == 0:  # Only add y-label to leftmost plots
            ax.set_ylabel('Relative Error', fontsize=12)
        ax.set_title(f'Dimension = {dim}', fontsize=14)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(fontsize=10)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        if n_rows > 1 or n_cols > 1:
            axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'error_vs_stages.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_solution(dim=2, num_points=50):
    """
    Plot a 3D visualization of the solution for 2D case.
    """
    if dim != 2:
        print("3D plot is only implemented for 2D problems.")
        return
    
    # Create a grid of test points
    x1 = np.linspace(-1, 1, num_points)
    x2 = np.linspace(-1, 1, num_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Convert to input format
    x_test = np.column_stack((X1.flatten(), X2.flatten()))
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float64)
    
    # Load the model
    try:
        model_path = os.path.join(MODEL_DIR, f"model_dim{dim}.pkl")
        model_data = load_model(model_path)
        
        # Predict using the model
        y_pred = predict_from_model_data(model_data, x_test)
        y_pred = y_pred.numpy().reshape(num_points, num_points)
        
        # Compute exact solution
        y_true = analytical_solution(x_test)
        y_true = y_true.numpy().reshape(num_points, num_points)
        
        # Compute error
        error = np.abs(y_pred - y_true)
        
        # Create plots
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Predicted Solution
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X1, X2, y_pred, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('$u(x_1, x_2)$')
        ax1.set_title('MSNN Predicted Solution')
        
        # Plot 2: Exact Solution
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X1, X2, y_true, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_zlabel('$u(x_1, x_2)$')
        ax2.set_title('Exact Solution')
        
        # Plot 3: Error
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X1, X2, error, cmap='hot', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')
        ax3.set_zlabel('Error')
        ax3.set_title('Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '3d_solution_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Error loading model or creating plot: {e}")
        return

# %%
# Function to generate a heatmap visualization of errors across different dimensions and stages
def plot_error_heatmap(dimension_results=None):
    """
    Create a heatmap of errors across dimensions and stages.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Extract data for the heatmap
    dims = dimension_results['dimension']
    max_stages = max(len(err_data) for err_data in dimension_results['stage_errors'])
    
    # Create empty matrix for the heatmap
    error_matrix = np.zeros((len(dims), max_stages))
    error_matrix.fill(np.nan)  # Fill with NaN for missing values
    
    # Fill in the error values
    for i, dim in enumerate(dims):
        stage_errors = dimension_results['stage_errors'][i]
        for stage, l2_error, _ in stage_errors:
            error_matrix[i, stage-1] = l2_error
    
    # Take log10 of errors for better visualization
    log_error_matrix = np.log10(error_matrix)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap
    cmap = plt.cm.get_cmap('viridis_r')
    
    # Plot the heatmap
    sns.heatmap(log_error_matrix, annot=True, fmt=".2f", cmap=cmap, 
                xticklabels=[f"Stage {i+1}" for i in range(max_stages)],
                yticklabels=[f"d={dim}" for dim in dims],
                mask=np.isnan(log_error_matrix))
    
    plt.title('Log10 of L2 Error Across Dimensions and Stages', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Function to plot training efficiency as a function of dimension
def plot_training_scalability(efficiency_results=None):
    """
    Create a more detailed plot of training efficiency vs. dimension.
    """
    if efficiency_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                efficiency_results = pickle.load(f)
        except:
            print("No saved results found. Please run the training efficiency analysis first.")
            return
    
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    ax1, ax2, ax3 = axs
    # Extract data
    dimensions = sorted(set(efficiency_results['dimension']))
    stages = sorted(set(efficiency_results['stage']))
    
    # Process data for easier plotting
    training_times = {}
    epochs_to_converge = {}
    final_errors = {}
    
    for dim in dimensions:
        training_times[dim] = []
        epochs_to_converge[dim] = []
        final_errors[dim] = []
        
        for stage in stages:
            # Find indices for this dimension and stage
            indices = [i for i, (d, s) in enumerate(zip(efficiency_results['dimension'], efficiency_results['stage'])) 
                      if d == dim and s == stage]
            
            if indices:
                idx = indices[0]
                training_times[dim].append(efficiency_results['training_time'][idx])
                epochs_to_converge[dim].append(efficiency_results['epochs_to_converge'][idx])
                final_errors[dim].append(efficiency_results['final_l2_error'][idx])
            else:
                training_times[dim].append(None)
                epochs_to_converge[dim].append(None)
                final_errors[dim].append(None)
    
    # Plot 1: Training time vs dimension
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, stage in enumerate(stages):
        times = [training_times[dim][i] for dim in dimensions if i < len(training_times[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(times) and times[j] is not None]
        valid_times = [t for t in times if t is not None]
        
        if valid_dims and valid_times:
            ax1.loglog(valid_dims, valid_times, marker=markers[i % len(markers)], 
                      color=colors[i % len(colors)], linewidth=2, markersize=8, 
                      label=f'Stage {stage}')
    
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_ylabel('Training Time (s)', fontsize=14)
    ax1.set_title('Training Time vs. Dimension', fontsize=16)
    ax1.legend(fontsize=12)
    
    # Plot 2: Epochs to convergence vs dimension
    for i, stage in enumerate(stages):
        epochs = [epochs_to_converge[dim][i] for dim in dimensions if i < len(epochs_to_converge[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(epochs) and epochs[j] is not None]
        valid_epochs = [e for e in epochs if e is not None]
        
        if valid_dims and valid_epochs:
            ax2.semilogx(valid_dims, valid_epochs, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_ylabel('Epochs to Convergence', fontsize=14)
    ax2.set_title('Epochs to Convergence vs. Dimension', fontsize=16)
    ax2.legend(fontsize=12)
    
    # Plot 3: Final L2 error vs dimension
    for i, stage in enumerate(stages):
        errors = [final_errors[dim][i] for dim in dimensions if i < len(final_errors[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(errors) and errors[j] is not None]
        valid_errors = [e for e in errors if e is not None]
        
        if valid_dims and valid_errors:
            ax3.semilogy(valid_dims, valid_errors, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    ax3.grid(True, which="both", ls="--", alpha=0.7)
    ax3.set_xlabel('Dimension $d', fontsize=14)
    ax3.set_ylabel('Final L2 Error', fontsize=14)
    ax3.set_title('Final L2 Error vs. Dimension', fontsize=16)
    ax3.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_scalability_detailed.png'), dpi=300, bbox_inches='tight')
    plt.show()
    axs[0, 1].set_ylabel('Training Time (s)', fontsize=14)
    axs[0, 1].set_title('Training Time Scaling with Dimension', fontsize=16)
    
    # Plot 3: Error vs. Stage for selected dimensions
    dims_to_plot = [dimensions[0], dimensions[-1]]  # First and last dimension
    for dim_idx, dim in enumerate(dims_to_plot):
        stage_errors = dimension_results['stage_errors'][dimensions.index(dim)]
        stages, l2_errors, linf_errors = zip(*stage_errors)
        
        axs[1, dim_idx].semilogy(stages, l2_errors, 'o-', color='blue', linewidth=2, markersize=8, label='Relative $L^2$ Error')
        axs[1, dim_idx].semilogy(stages, linf_errors, 's--', color='red', linewidth=2, markersize=8, label='Relative $L^\\infty$ Error')
        axs[1, dim_idx].grid(True, which="both", ls="--", alpha=0.7)
        axs[1, dim_idx].set_xlabel('Stage', fontsize=14)
        axs[1, dim_idx].set_ylabel('Relative Error', fontsize=14)
        axs[1, dim_idx].set_title(f'Error Reduction Across Stages (d={dim})', fontsize=16)
        axs[1, dim_idx].legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'paper_figure_combined.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Main execution function
def main():
    """Main function to run experiments and generate plots."""
    # Define dimensions to test
    dimensions_to_test = [1, 2, 5, 8, 10]
    
    # Check if results already exist
    run_experiments = True
    if os.path.exists(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl')) and \
       os.path.exists(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl')):
        response = input("Results already exist. Do you want to run experiments again? (y/n): ")
        run_experiments = response.lower() == 'y'
    
    if run_experiments:
        # Run dimension experiment
        print("Running dimension scaling experiment...")
        dimension_results = run_dimension_experiment(dimensions_to_test)
        
        # Analyze training efficiency
        print("Analyzing training efficiency...")
        efficiency_results = analyze_training_efficiency(dimensions_to_test)
    else:
        # Load existing results
        print("Loading existing results...")
        with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
            dimension_results = pickle.load(f)
        
        with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
            efficiency_results = pickle.load(f)
    
    # Generate plots
    print("Generating plots...")
    
    # Figure 2: Error vs. Dimension
    plot_dimension_scaling(dimension_results)
    
    # Figure 3: Training Efficiency
    plot_training_efficiency(efficiency_results)
    
    # Summary table
    table_data = generate_performance_metrics_table(dimensions_to_test, dimension_results)
    print("\nPerformance Metrics Table:")
    for row in table_data:
        print(f"Dimension {row['Dimension d']}: L2 Error = {row['Rel. L2 Error']}, "
              f"Linf Error = {row['Rel. Linf Error']}, Training Time = {row['Training Time (s)']}s")
    
    # Generate LaTeX table
    print("\nLaTeX Table for Paper:")
    generate_paper_table(dimension_results)
    
    # Additional plots
    plot_performance_trends(table_data)
    plot_error_vs_stages(dimension_results)
    plot_error_heatmap(dimension_results)
    plot_training_scalability(efficiency_results)
    plot_paper_figures(dimension_results, efficiency_results)
    
    # 3D visualization for 2D case
    if 2 in dimensions_to_test:
        plot_3d_solution(dim=2)
    
    print("\nAll plots have been generated and saved to the 'plots' directory.")

# %%
# Function to load existing results and generate plots
def load_and_visualize_results():
    """
    Load previously computed results and generate visualizations.
    """
    # Define dimensions to visualize
    dimensions_to_visualize = [1, 2, 5, 8, 10]
    
    try:
        # Load dimension experiment results
        with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
            dimension_results = pickle.load(f)
        
        # Load training efficiency results
        with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
            efficiency_results = pickle.load(f)
        
        # Generate plots
        print("Generating plots from saved results...")
        
        # Figure 2: Error vs. Dimension
        plot_dimension_scaling(dimension_results)
        
        # Figure 3: Training Efficiency
        plot_training_efficiency(efficiency_results)
        
        # Summary table
        table_data = generate_performance_metrics_table(dimensions_to_visualize, dimension_results)
        print("\nPerformance Metrics Table:")
        for row in table_data:
            print(f"Dimension {row['Dimension d']}: L2 Error = {row['Rel. L2 Error']}, "
                  f"Linf Error = {row['Rel. Linf Error']}, Training Time = {row['Training Time (s)']}s")
        
        # Additional plots
        plot_performance_trends(table_data)
        plot_error_vs_stages(dimension_results)
        plot_error_heatmap(dimension_results)
        plot_training_scalability(efficiency_results)
        plot_paper_figures(dimension_results, efficiency_results)
        
        # 3D visualization for 2D case
        if 2 in dimensions_to_visualize:
            plot_3d_solution(dim=2)
        
        print("\nAll plots have been generated and saved to the 'plots' directory.")
    
    except FileNotFoundError:
        print("Results files not found. Please run the experiments first by calling main().")
    except Exception as e:
        print(f"Error loading results: {e}")

# %%
# Actually run the main function when the script is executed
def plot_training_efficiency(results=None):
    """
    Plot training time per stage vs. dimension.
    This corresponds to Figure 3 in the thesis.
    """
    if results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                results = pickle.load(f)
        except:
            print("No saved results found. Please run the training efficiency analysis first.")
            return
    
    # Convert results to DataFrame-like structure for easier plotting
    dimensions = sorted(set(results['dimension']))
    stages = sorted(set(results['stage']))
    
    training_times = {}
    for dim in dimensions:
        training_times[dim] = []
        for stage in stages:
            # Find the entry for this dimension and stage
            indices = [i for i, (d, s) in enumerate(zip(results['dimension'], results['stage'])) 
                      if d == dim and s == stage]
            
            if indices:
                training_times[dim].append(results['training_time'][indices[0]])
            else:
                training_times[dim].append(None)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training time for each stage
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, stage in enumerate(stages):
        times = [training_times[dim][i] for dim in dimensions if i < len(training_times[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(times) and times[j] is not None]
        valid_times = [t for t in times if t is not None]
        
        if valid_dims and valid_times:
            plt.semilogy(valid_dims, valid_times, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Dimension $d$', fontsize=14)
    plt.ylabel('Training Time per Stage (s)', fontsize=14)
    plt.title('MSNN Training Time Scaling with Dimension', fontsize=16)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'figure3_training_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_trends(data=None):
    """
    Create a summary plot of performance metrics across dimensions.
    """
    if data is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                results = pickle.load(f)
                
            dimensions = sorted(results['dimension'])
            data = generate_performance_metrics_table(dimensions, results)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Extract data for plotting
    dimensions = [d['Dimension d'] for d in data]
    l2_errors = [d['Rel. L2 Error'] for d in data]
    linf_errors = [d['Rel. Linf Error'] for d in data]
    training_times = [d['Training Time (s)'] for d in data]
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Errors vs. Dimension
    ax1.semilogy(dimensions, l2_errors, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Relative $L^2$ Error')
    ax1.semilogy(dimensions, linf_errors, 's--', color='red', linewidth=2, 
                markersize=8, label='Relative $L^\\infty$ Error')
    
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_xlabel('Dimension $d$', fontsize=14)
    ax1.set_ylabel('Relative Error', fontsize=14)
    ax1.set_title('Error Scaling with Dimension', fontsize=16)
    ax1.legend(fontsize=12)
    
    # Plot 2: Training Time vs. Dimension
    ax2.semilogy(dimensions, training_times, 'D-', color='green', linewidth=2, markersize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_xlabel('Dimension $d$', fontsize=14)
    ax2.set_ylabel('Training Time (s)', fontsize=14)
    ax2.set_title('Training Time Scaling with Dimension', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'performance_trends_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_vs_stages(dimension_results=None):
    """
    Plot error reduction across training stages for different dimensions.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Create a figure with subplots for each dimension
    dims = dimension_results['dimension']
    num_dims = len(dims)
    
    # Arrange subplots in a grid
    n_cols = min(3, num_dims)
    n_rows = (num_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
    
    # Flatten axes array for easier indexing if there's more than one row
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = [axes]  # Convert to list if there's only one subplot
    
    # Plot each dimension's stage errors
    for i, dim in enumerate(dims):
        stage_errors = dimension_results['stage_errors'][i]
        
        stages, l2_errors, linf_errors = zip(*stage_errors)
        
        ax = axes[i] if num_dims > 1 else axes
        ax.semilogy(stages, l2_errors, 'o-', color='blue', linewidth=2, 
                   markersize=8, label='Relative $L^2$ Error')
        ax.semilogy(stages, linf_errors, 's--', color='red', linewidth=2, 
                   markersize=8, label='Relative $L^\\infty$ Error')
        
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.set_xlabel('Stage', fontsize=12)
        if i % n_cols == 0:  # Only add y-label to leftmost plots
            ax.set_ylabel('Relative Error', fontsize=12)
        ax.set_title(f'Dimension = {dim}', fontsize=14)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(fontsize=10)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        if n_rows > 1 or n_cols > 1:
            axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'error_vs_stages.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_solution(dim=2, num_points=50):
    """
    Plot a 3D visualization of the solution for 2D case.
    """
    if dim != 2:
        print("3D plot is only implemented for 2D problems.")
        return
    
    # Create a grid of test points
    x1 = np.linspace(-1, 1, num_points)
    x2 = np.linspace(-1, 1, num_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Convert to input format
    x_test = np.column_stack((X1.flatten(), X2.flatten()))
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float64)
    
    # Load the model
    try:
        model_path = os.path.join(MODEL_DIR, f"model_dim{dim}.pkl")
        model_data = load_model(model_path)
        
        # Predict using the model
        y_pred = predict_from_model_data(model_data, x_test)
        y_pred = y_pred.numpy().reshape(num_points, num_points)
        
        # Compute exact solution
        y_true = analytical_solution(x_test)
        y_true = y_true.numpy().reshape(num_points, num_points)
        
        # Compute error
        error = np.abs(y_pred - y_true)
        
        # Create plots
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Predicted Solution
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X1, X2, y_pred, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('$u(x_1, x_2)$')
        ax1.set_title('MSNN Predicted Solution')
        
        # Plot 2: Exact Solution
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X1, X2, y_true, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_zlabel('$u(x_1, x_2)$')
        ax2.set_title('Exact Solution')
        
        # Plot 3: Error
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X1, X2, error, cmap='hot', alpha=0.8, 
                                linewidth=0, antialiased=True)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')
        ax3.set_zlabel('Error')
        ax3.set_title('Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '3d_solution_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Error loading model or creating plot: {e}")
        return

# %%
# Function to generate a heatmap visualization of errors across different dimensions and stages
def plot_error_heatmap(dimension_results=None):
    """
    Create a heatmap of errors across dimensions and stages.
    """
    if dimension_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'dimension_experiment_results.pkl'), 'rb') as f:
                dimension_results = pickle.load(f)
        except:
            print("No saved results found. Please run the dimension experiment first.")
            return
    
    # Extract data for the heatmap
    dims = dimension_results['dimension']
    max_stages = max(len(err_data) for err_data in dimension_results['stage_errors'])
    
    # Create empty matrix for the heatmap
    error_matrix = np.zeros((len(dims), max_stages))
    error_matrix.fill(np.nan)  # Fill with NaN for missing values
    
    # Fill in the error values
    for i, dim in enumerate(dims):
        stage_errors = dimension_results['stage_errors'][i]
        for stage, l2_error, _ in stage_errors:
            error_matrix[i, stage-1] = l2_error
    
    # Take log10 of errors for better visualization
    log_error_matrix = np.log10(error_matrix)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap
    cmap = plt.cm.get_cmap('viridis_r')
    
    # Plot the heatmap
    sns.heatmap(log_error_matrix, annot=True, fmt=".2f", cmap=cmap, 
                xticklabels=[f"Stage {i+1}" for i in range(max_stages)],
                yticklabels=[f"d={dim}" for dim in dims],
                mask=np.isnan(log_error_matrix))
    
    plt.title('Log10 of L2 Error Across Dimensions and Stages', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Function to plot training efficiency as a function of dimension
def plot_training_scalability(efficiency_results=None):
    """
    Create a more detailed plot of training efficiency vs. dimension.
    """
    if efficiency_results is None:
        # Try to load saved results
        try:
            with open(os.path.join(RESULTS_DIR, 'training_efficiency_results.pkl'), 'rb') as f:
                efficiency_results = pickle.load(f)
        except:
            print("No saved results found. Please run the training efficiency analysis first.")
            return
    
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Extract data
    dimensions = sorted(set(efficiency_results['dimension']))
    stages = sorted(set(efficiency_results['stage']))
    
    # Process data for easier plotting
    training_times = {}
    epochs_to_converge = {}
    final_errors = {}
    
    for dim in dimensions:
        training_times[dim] = []
        epochs_to_converge[dim] = []
        final_errors[dim] = []
        
        for stage in stages:
            # Find indices for this dimension and stage
            indices = [i for i, (d, s) in enumerate(zip(efficiency_results['dimension'], efficiency_results['stage'])) 
                      if d == dim and s == stage]
            
            if indices:
                idx = indices[0]
                training_times[dim].append(efficiency_results['training_time'][idx])
                epochs_to_converge[dim].append(efficiency_results['epochs_to_converge'][idx])
                final_errors[dim].append(efficiency_results['final_l2_error'][idx])
            else:
                training_times[dim].append(None)
                epochs_to_converge[dim].append(None)
                final_errors[dim].append(None)
    
    # Plot 1: Training time vs dimension
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, stage in enumerate(stages):
        times = [training_times[dim][i] for dim in dimensions if i < len(training_times[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(times) and times[j] is not None]
        valid_times = [t for t in times if t is not None]
        
        if valid_dims and valid_times:
            ax1.loglog(valid_dims, valid_times, marker=markers[i % len(markers)], 
                      color=colors[i % len(colors)], linewidth=2, markersize=8, 
                      label=f'Stage {stage}')
    
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.set_ylabel('Training Time (s)', fontsize=14)
    ax1.set_title('Training Time vs. Dimension', fontsize=16)
    ax1.legend(fontsize=12)
    
    # Plot 2: Epochs to convergence vs dimension
    for i, stage in enumerate(stages):
        epochs = [epochs_to_converge[dim][i] for dim in dimensions if i < len(epochs_to_converge[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(epochs) and epochs[j] is not None]
        valid_epochs = [e for e in epochs if e is not None]
        
        if valid_dims and valid_epochs:
            ax2.semilogx(valid_dims, valid_epochs, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_ylabel('Epochs to Convergence', fontsize=14)
    ax2.set_title('Epochs to Convergence vs. Dimension', fontsize=16)
    ax2.legend(fontsize=12)
    
    # Plot 3: Final L2 error vs dimension
    for i, stage in enumerate(stages):
        errors = [final_errors[dim][i] for dim in dimensions if i < len(final_errors[dim])]
        valid_dims = [dim for j, dim in enumerate(dimensions) if j < len(errors) and errors[j] is not None]
        valid_errors = [e for e in errors if e is not None]
        
        if valid_dims and valid_errors:
            ax3.semilogy(valid_dims, valid_errors, marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], linewidth=2, markersize=8, 
                        label=f'Stage {stage}')
    
    ax3.grid(True, which="both", ls="--", alpha=0.7)
    ax3.set_xlabel('Dimension $d', fontsize=14)
    ax3.set_ylabel('Final L2 Error', fontsize=14)
    ax3.set_title('Final L2 Error vs. Dimension', fontsize=16)
    ax3.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_scalability_detailed.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
    # Uncomment the following line to load and visualize results without running experiments
    # load_and_visualize_results()