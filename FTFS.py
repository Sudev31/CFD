#u_t + u_x = 0 using FTBS method, my attempt!:)
import numpy as np
import matplotlib.pyplot as plt
def g(x):
    return np.sin(2.0 * np.pi * x) #inital condition

X = 1
T = 5 #final t and final x, assuming min = 0 for both.
del_x = 0.01
del_t = 1/2*del_x
size_x = int((X/del_x)+1)
size_t = int((T/del_t)+1)
u = np.zeros((size_t, size_x))

for i in range(size_x):
    u[0,i] = g(i*del_x)


for i in range(size_t-1):
    for j in range(size_x-1):
        
        u[i+1,j] = u[i,j] - (del_t/del_x)*(u[i,j+1]-u[i,j])
    
    u[i+1,size_x-1] = u[i+1,0] 
print(u[1,:])
x_domain = np.arange(0, 1 + del_x, del_x)
t_domain = np.arange(0, 5 + del_t, del_t)
X, T = np.meshgrid(x_domain, t_domain)
u_exact = np.sin(2 * np.pi * (X-T))
u_t_4 = u[int((4/del_t)+1), :]
exact_solution_t_4 = np.sin(2 * np.pi * (x_domain - 4))
u_t_45 = u[int((4.5/del_t)+1), :]
exact_solution_t_45 = np.sin(2 * np.pi * (x_domain - 4.5))

fig = plt.figure(figsize=(14, 12))
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, T, u, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u(x, t) (approx.)')
ax1.set_title('Approximate Solution')

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
ax3.set_title('Comparison of Numerical and Exact Solutions at t = 4')
ax3.legend()

ax4 = fig.add_subplot(224)
ax4.plot(x_domain, u_t_45, label='Numerical Solution at t = 4.5', linestyle='--')
ax4.plot(x_domain, exact_solution_t_45, label='Exact Solution at t = 4.5', linestyle='-')
ax4.set_xlabel('x')
ax4.set_ylabel('u(x, t)')
ax4.set_title('Comparison of Numerical and Exact Solutions at t = 4.5')
ax4.legend()

plt.show()





