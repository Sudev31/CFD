import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return np.sin(2.0 * np.pi * x)  # Initial condition

# Parameters
X = 1
T = 5  # Final t and final x, assuming min = 0 for both.
del_x = 0.001
del_t = 0.5 * del_x  # CFL condition satisfied
l = del_t / del_x
size_x = int((X / del_x) + 1)
size_t = int((T / del_t) + 1)
u = np.zeros((size_t, size_x))

# Initial condition
for i in range(size_x):
    u[0, i] = g(i * del_x)

# Matrix assembly for BTBS
matrix = np.zeros((size_x, size_x))
for i in range(size_x):
    matrix[i][i] = 1 + l
    matrix[i][i - 1] = -l  # Handles the backward difference in space

# For periodic boundary conditions
matrix[0, -1] = -l  # Link the last element to the first one

# Invert the matrix (since this is an implicit method)
Inv = np.linalg.inv(matrix)

# Time-stepping loop
for i in range(size_t - 1):
    u[i + 1, :] = np.dot(Inv, u[i, :])

# Set up domains for plotting
x_domain = np.arange(0, 1 + del_x, del_x)
t_domain = np.arange(0, 5 + del_t, del_t)
X, T = np.meshgrid(x_domain, t_domain)

# Exact solution
u_exact = np.sin(2 * np.pi * (X - T))

# Plotting at specific time steps
u_t_4 = u[int(4 / del_t), :]
exact_solution_t_4 = np.sin(2 * np.pi * (x_domain - 4))
u_t_45 = u[int(4.5 / del_t), :]
exact_solution_t_45 = np.sin(2 * np.pi * (x_domain - 4.5))

# Plot the solutions
fig = plt.figure(figsize=(14, 12))
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, T, u, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u(x, t) (approx.)')
ax1.set_title('Approximate Solution (BTBS)')

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(X, T, u_exact, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x, t)')
ax2.set_title('Exact Solution (sin(2π(x - t)))')

ax3 = fig.add_subplot(223)
ax3.plot(x_domain, u_t_4, label='Numerical Solution at t = 4', linestyle='--')
ax3.plot(x_domain, exact_solution_t_4, label='Exact Solution at t = 4', linestyle='-')
ax3.set_xlabel('x')
ax3.set_ylabel('u(x, t)')
ax3.set_title('Comparison at t = 4')
ax3.legend()

ax4 = fig.add_subplot(224)
ax4.plot(x_domain, u_t_45, label='Numerical Solution at t = 4.5', linestyle='--')
ax4.plot(x_domain, exact_solution_t_45, label='Exact Solution at t = 4.5', linestyle='-')
ax4.set_xlabel('x')
ax4.set_ylabel('u(x, t)')
ax4.set_title('Comparison at t = 4.5')
ax4.legend()

plt.show()
