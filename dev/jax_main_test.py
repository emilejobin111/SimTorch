


import matplotlib
matplotlib.use('TkAgg')
import jax_src.core.nn as nn
import jax_src.core.optim as optim
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


layer_list = [
    nn.Linear.new(input_count=2, output_count=256),
    nn.Tanh(), # as Tanh is fluid, it will predict well our function
    nn.Linear.new(input_count=256, output_count=256),
    nn.Tanh(),
    nn.Linear.new(input_count=256, output_count=256),
    nn.Tanh(),
    nn.Linear.new(input_count=256, output_count=1)
]


neural_network = nn.Sequential(*layer_list)


def f(x:jnp.ndarray,y:jnp.ndarray)->jnp.ndarray:
    sqrt = jnp.sqrt(x**2+y**2)
    sqrt = jnp.where(sqrt == 0, 1e-9, sqrt) # to prevent the /0
    z = jnp.sin(2*sqrt)/sqrt
    return z

# 1. Define the axes (between -5 and 5 with 128 points per axis)
x = jnp.linspace(-5, 5, 256,dtype=jnp.float32)
y = jnp.linspace(-5, 5, 256,dtype=jnp.float32)

# 2. Create the 2D grid (X and Y will both have a shape of (128, 128))
X, Y = jnp.meshgrid(x, y)

# 3. Flatten and stack into (x, y) pairs -> Final shape: (16384, 2)
X_train = jnp.column_stack((X.ravel(), Y.ravel()))

# 4. get the z axis
Z = f(Y,X)

Y_train = Z.ravel().reshape(-1, 1)
print(Y_train.size)



def plot_3d_surface(X_data, Y_data, title="3D Surface Visualization", cmap="coolwarm"):
    """
    Plots a 3D surface from flattened data arrays.
    
    Parameters:
    - X_data: numpy array of shape (N, 2) containing the [x, y] pairs
    - Y_data: numpy array of shape (N, 1) or (N,) containing the z values
    - title: string, title of the plot
    - cmap: string, colormap name for the surface rendering
    """
    # 1. Deduce the grid resolution assuming a square grid (NxN)
    num_samples = X_data.shape[0]
    grid_res = int(jnp.sqrt(num_samples))
    
    if grid_res * grid_res != num_samples:
        raise ValueError("Data size must be a perfect square to reconstruct a uniform grid.")
    
    # 2. Reshape the flattened arrays back into 2D grids (grid_res, grid_res)
    X_grid = X_data[:, 0].reshape(grid_res, grid_res)
    Y_grid = X_data[:, 1].reshape(grid_res, grid_res)
    Z_grid = Y_data.reshape(grid_res, grid_res)
    
    # 3. Initialize the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 4. Generate the surface plot
    surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap, edgecolor='none', alpha=0.9)
    
    # 5. Formatting and annotations
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Add colorbar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Z Value')
    
    # Set optimal initial viewing angle
    ax.view_init(elev=35, azim=45)
    plt.tight_layout()
    plt.show()

# plot_3d_surface(X_train,Y_train)



Y_predicted = neural_network.forward(X_train) # forward(data) is asking the nn it's prediction




optimiser = optim.AdamW.new(net=neural_network,maximise=False,alpha=3e-4)

@jax.jit
def epoch_step(net:nn.JaxModule, optimiser:optim.AdamW, X, Y):
    loss, grad_net = nn.mse(net,X,Y)
    new_net, new_optim = optimiser.step(net,grad_net)
    return new_net, new_optim, loss
neural_network, optimiser, loss = epoch_step(neural_network, optimiser, X_train, Y_train)

import time

t = time.perf_counter()
for i in range(1000):
    neural_network, optimiser, loss = epoch_step(neural_network, optimiser, X_train, Y_train)
    if i % 100 == 0:
        print(f"epoch {i} | loss : {loss}")
print(time.perf_counter()-t)

predicted = neural_network.forward(X_train)
plot_3d_surface(X_train, predicted)

