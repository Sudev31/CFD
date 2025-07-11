import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
del_x = 0.1
del_t = del_x / 2
x_domain = np.arange(0, 1 + del_x, del_x)
t_domain = np.arange(0, 5 + del_t, del_t)
u_approx = np.zeros((len(t_domain), len(x_domain)))
u_approx[0, :] = np.sin(2 * np.pi * x_domain)

# Numerical solution
for n in range(0, len(t_domain)-1):
    for i in range(1, len(x_domain)):
        u_approx[n + 1, i] = u_approx[n, i] + (del_t / del_x) * (u_approx[n, i-1] - u_approx[n, i])
    u_approx[n + 1, 0] = u_approx[n + 1, -1]  # Boundary condition

# Create meshgrid
X, T = np.meshgrid(x_domain, t_domain)

# Exact solution at specific times
exact_solution_t_4 = np.sin(2 * np.pi * (x_domain - 4))
exact_solution_t_45 = np.sin(2 * np.pi * (x_domain - 4.5))

# Ensure indices are within bounds
index_4 = int(4 / del_t)
index_45 = int(4.5 / del_t)

# Handle potential index out of bounds
if index_4 < len(u_approx):
    u_t_4 = u_approx[index_4, :]
else:
    u_t_4 = np.zeros_like(x_domain)

if index_45 < len(u_approx):
    u_t_45 = u_approx[index_45, :]
else:
    u_t_45 = np.zeros_like(x_domain)

# Plotting
fig = plt.figure(figsize=(14, 12))

# Approximate Solution
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, T, u_approx, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u(x, t) (approx.)')
ax1.set_title('Approximate Solution')

# Exact Solution
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(X, T, np.sin(2 * np.pi * (X - T)), cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x, t)')
ax2.set_title('Exact Solution (sin(2π(x - t)))')

# Comparison at t = 4
ax3 = fig.add_subplot(223)
ax3.plot(x_domain, u_t_4, label='Numerical Solution at t = 4', linestyle='--')
ax3.plot(x_domain, exact_solution_t_4, label='Exact Solution at t = 4', linestyle='-')
ax3.set_xlabel('x')
ax3.set_ylabel('u(x, t)')
ax3.set_title('Comparison of Numerical and Exact Solutions at t = 4')
ax3.legend()

# Comparison at t = 4.5
ax4 = fig.add_subplot(224)
ax4.plot(x_domain, u_t_45, label='Numerical Solution at t = 4.5', linestyle='--')
ax4.plot(x_domain, exact_solution_t_45, label='Exact Solution at t = 4.5', linestyle='-')
ax4.set_xlabel('x')
ax4.set_ylabel('u(x, t)')
ax4.set_title('Comparison of Numerical and Exact Solutions at t = 4.5')
ax4.legend()

plt.tight_layout()  # Adjust layout
plt.show()
