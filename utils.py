import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from itertools import product

class NeuralNet:
    # Initialize the class
    def __init__(self, t_u, x_u, layers, kappa, lt, ut, acts=0):

        self.scale = tf.reduce_max(tf.abs(x_u)) / 2
        x_u2 = x_u / self.scale
        actv = [tf.tanh, tf.sin]

        self.t_u = t_u
        self.x_u = x_u2
        self.datatype = t_u.dtype

        self.lt = lt
        self.ut = ut

        self.layers = layers
        self.kappa = kappa

        # determine the activation function to use
        self.actv = actv[acts]

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # Create a list including all training variables
        self.train_variables = self.weights + self.biases
        # Key point: anything updates in train_variables will be
        #            automatically updated in the original tf.Variable

        # define the loss function
        self.loss0 = self.scale ** 2
        self.loss = []
        self.loss_0 = self.loss_NN()

        self.optimizer_Adam = tf.optimizers.Adam()

    '''
    Functions used to establish the initial neural network
    ===============================================================
    '''

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)

        for l in range(0, num_layers - 1):
            W = self.MPL_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.datatype))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def MPL_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.datatype))

    def get_params(self):
        return (self.weights, self.biases)

    '''
    Functions used to building the physics-informed contrainst and loss
    ===============================================================
    '''

    def neural_net(self, X):
        weights = self.weights
        biases = self.biases

        num_layers = len(weights) + 1

        H = 2.0 * tf.math.divide(
                    tf.math.subtract(X, tf.transpose(self.lt)), 
                    tf.transpose(tf.math.subtract(self.ut, self.lt))) \
            - 1.0

        W = weights[0]
        b = biases[0]
        H = self.actv(tf.add(self.kappa * tf.matmul(H, W), b))

        for l in range(1, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    @tf.function(reduce_retracing=True)
    # calculate the physics-informed loss function
    def loss_NN(self):
        self.x_pred = self.neural_net(self.t_u)
        loss = tf.reduce_mean(tf.square(self.x_u - self.x_pred))
        return loss

    '''
    Functions used to define ADAM optimizers
    ===============================================================
    '''

    # define the function to apply the ADAM optimizer
    def adam_function(self):
        @tf.function(reduce_retracing=True)
        def f1():
            # calculate the loss
            loss_norm = self.loss_NN()
            loss_value = loss_norm * self.loss0
            # store loss value so we can retrieve later
            tf.py_function(f1.loss.append, inp=[loss_value], Tout=[])

            # print out iteration & loss
            f1.iter.assign_add(1)

            str_iter = tf.strings.as_string([f1.iter])
            str_loss = tf.strings.as_string([loss_value], precision=4, scientific=True)

            str_print = tf.strings.join(["Mode: Adam", "Iter: ", str_iter[0],
                                        ", loss: ", str_loss[0]])
            # tf.cond(
            #     f1.iter % 10 == 0,
            #     lambda: tf.print(str_print),
            #     lambda: tf.constant(True)  # return arbitrary for non-printing case
            # )
            return loss_norm

        f1.iter = tf.Variable(0)
        f1.term = []
        f1.loss = []
        return f1

    def Adam_optimizer(self, nIter):
        varlist = self.train_variables
        func_adam = self.adam_function()
        for it in tqdm(range(nIter), desc="Adam  Optimization"):
            tf.keras.optimizers.Adam(func_adam, varlist)
            #self.optimizer_Adam.minimize(func_adam, varlist)
        return func_adam

    '''
    Functions used to define L-BFGS optimizers
    ===============================================================
    '''

    # A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    def Lbfgs_function(self, varlist):
        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(varlist)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.prod(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        def assign_new_model_parameters(params_1d):
            # A function updating the model's parameters with a 1D tf.Tensor.
            # Sub-function under function of class not need to input self

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                varlist[i].assign(tf.reshape(param, shape))

        @tf.function(reduce_retracing=True)
        def f2(params_1d):
            # A function that can be used by tfp.optimizer.lbfgs_minimize.
            # This function is created by function_factory.
            # Sub-function under function of class not need to input self

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                # this step is critical for self-defined function for L-BFGS
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_norm = self.loss_NN()
                loss_value = loss_norm * self.loss0

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_norm, varlist)
            grads = tf.dynamic_stitch(idx, grads)

            # store loss value so we can retrieve later
            tf.py_function(f2.loss.append, inp=[loss_value], Tout=[])

            # print out iteration & loss
            f2.iter.assign_add(1)

            str_iter = tf.strings.as_string([f2.iter])
            str_loss = tf.strings.as_string([loss_value], precision=4, scientific=True)

            str_print = tf.strings.join(["\nMode: LBFGS", "Iter: ", str_iter[0],
                                        ", loss: ", str_loss[0]])
            tf.cond(
                f2.iter % 3000 == 0,
                lambda: tf.print(str_print),
                lambda: tf.constant(True)  # return arbitrary for non-printing case
            )

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f2.iter = tf.Variable(0)
        f2.idx = idx
        f2.part = part
        f2.shapes = shapes
        f2.assign_new_model_parameters = assign_new_model_parameters
        f2.loss = []

        return f2

    # define the function to apply the L-BFGS optimizer
    def Lbfgs_optimizer(self, nIter, varlist):

        func = self.Lbfgs_function(varlist)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, varlist)

        max_nIter = tf.cast(nIter / 3, dtype=tf.int32)

        # Initialize the tqdm progress bar
        with tqdm(total=nIter, desc="LBFGS Optimization", unit="iter") as pbar:

            # Create a custom function for value and gradients that we can manually control
            def value_and_grads_fn(params_1d):
                # This function computes both the loss and gradients while updating the progress bar.
                f2 = func(params_1d)
                pbar.update(1)
                return f2

            # Run the L-BFGS optimizer with a custom value_and_grads_fn
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=value_and_grads_fn,
                initial_position=init_params,
                tolerance=1e-11,
                max_iterations=max_nIter
            )

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)

        return func

    '''
    Function used for training the model
    ===============================================================
    '''

    def train(self, nIter, idxOpt):
        if idxOpt == 1:
            # mode 1: running the Adam optimization
            func_adam = self.Adam_optimizer(nIter)
            self.loss += func_adam.loss
        elif idxOpt == 2:
            # mode 2: running the Lbfgs optimization
            func_bfgs = self.Lbfgs_optimizer(nIter, self.train_variables)
            self.loss += func_bfgs.loss

    # @tf.function
    def predict(self, t):
        x_p = self.neural_net(t) * self.scale
        return x_p

def create_ds(dim, lo, hi, N, max_data_size):
    # while (N ** dim) > max_data_size:
    #     N -= 1
    points_per_dim = int(N ** (1/dim))
    grids = [np.linspace(lo, hi, points_per_dim) for _ in range(dim)]
    mesh_points = np.array(list(product(*grids)))  # Cartesian product
    mesh_tf = tf.cast(mesh_points, dtype=tf.float64)
    return mesh_tf

def example_1d_fun(x_train):
    y_train = x_train ** 3 / (0.01 + x_train ** 4)
    return y_train

def example_2d_fun(x_train1, x_train2):
    y_train = (np.sin(x_train1 + 1) - 0.5 * np.sin(x_train1))*(1 - x_train2 ** 2)
    return y_train

def poisson(x_train):
    N_f = x_train.shape[0]
    dim = x_train.shape[-1]
    coeffs = 1 # used to be random from 1 to dim - 1
    const_2 = 1
    x_radius = 1

    xf = x_train.numpy() # used to be random points of size (N_f, args.dim) normalized by sqrt(sum(squares))
    x = xf

    u1 = x_radius**2 - np.sum(x**2, 1, keepdims=True)
    du1_dx = -2 * x
    d2u1_dx2 = -2

    if dim == 1: x1, x2 = x, x
    else: x1, x2 = x[:, :-1], x[:, 1:]
    u2 = coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1))
    u2 = np.sum(u2, 1, keepdims=True)
    du2_dx_part1 = coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * \
            (1 - x2 * np.sin(x1))
    du2_dx_part2 = coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * \
            (-const_2 * np.sin(x2) + np.cos(x1))
    du2_dx = np.zeros((N_f, dim))
    du2_dx[:, :-1] += du2_dx_part1
    du2_dx[:, 1:] += du2_dx_part2
    d2u2_dx2_part1 = -coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (1 - x2 * np.sin(x1))**2 + \
            coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (- x2 * np.cos(x1))
    d2u2_dx2_part2 = -coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (-const_2 * np.sin(x2) + np.cos(x1))**2 + \
            coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (-const_2 * np.cos(x2))
    d2u2_dx2 = np.zeros((N_f, dim))
    d2u2_dx2[:, :-1] += d2u2_dx2_part1
    d2u2_dx2[:, 1:] += d2u2_dx2_part2
    ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
    ff = np.sum(ff, 1)
    u = (u1 * u2).reshape(-1)
    ff = ff + np.sin(u)

    ff = tf.convert_to_tensor(ff, dtype=tf.float64)  
    return ff

def calculate_N(f_max, L):
    dx = 1 / (2 * f_max) 
    N = int(np.ceil(L / dx))
    N = int(2 ** np.ceil(np.log2(N)))
    return N

def normalize(x):
    mean, var = tf.nn.moments(x, axes=0)
    normalized_x = (x - mean) / (tf.sqrt(var))
    return normalized_x