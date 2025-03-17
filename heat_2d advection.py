import os, glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys
# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-pde', choices=('linear', 'varadv', 'burger', 'bucklev'),
                    help='PDE', default='linear')
parser.add_argument('-scheme', choices=('uw','rk2' ), help='uw',
                    default='uw')
parser.add_argument('-ncellx', type=int, help='Number of x cells', default=50)
parser.add_argument('-ncelly', type=int, help='Number of y cells', default=50)
parser.add_argument('-cfl', type=float, help='CFL number', default=1.0)
parser.add_argument('-Tf', type=float, help='Final time', default=1.0)
parser.add_argument('-plot_freq', type=int, help='Frequency to plot solution',
                    default=1)
parser.add_argument('-ic', choices=('sin2pi', 'expo','hat', 'solid'),
                    help='Initial condition', default='sin2pi')
parser.add_argument('-save_freq', type=int, help='Frequency to save solution',
                    default=0)
args = parser.parse_args()

# Select PDE

# Select initial condition
if args.ic == 'sin2pi':
    from sin2pi import *
else:
    print('Unknown initial condition')
    exit()

# Select cfl
cfl = args.cfl
nx = args.ncellx       # number of cells in the x-direction
ny = args.ncelly       # number of cells in the y-direction
global fileid
fileid = 0
dx = (xmax - xmin)/nx
dy = (ymax - ymin)/ny
# Allocate solution variables
v = np.zeros((nx+5, ny+5))  # 2 ghost points each side
# Set initial condition by interpolation
for i in range(nx+5):
    for j in range(ny+5):
        x = xmin + (i-2) * dx     
        y = ymin + (j-2) * dy
        val = initial_condition(x, y)
        v[i, j] = val
# copy the initial condition
v0 = v[2:nx+3, 2:ny+3].copy()
# it stores the coordinates of real cell vertices 
xgrid1 = np.linspace(xmin, xmax, nx + 1)
ygrid1 = np.linspace(ymin, ymax, ny +1 )
ygrid, xgrid = np.meshgrid(ygrid1, xgrid1)
#------------To save solution--------------------------------------------
def getfilename(file, fileid):
    if fileid <10:
        file = file +"00"+str(fileid)+".plt"
    elif fileid <99:
        file = file +"0"+str(fileid)+".plt"
    else:
        file =file+str(fileid)+".plt"
    return file
# save solution to a file
def savesol(t, var_u):
    global fileid
    if not os.path.isdir("sol"): # creat a dir if not
       os.makedirs("sol")
       print('Directory "sol" is created')
    if fileid == 0: # remove the content of the folder
        print('The directory "sol" is going to be formated!')
        if input('Do You Want To Continue? [y/n] ') != 'y':
            sys.exit('Execution is treminated')
        fs = glob.glob('./sol/*')
        for f in fs:
           os.remove(f)
    filename = "sol"
    filename = getfilename(filename, fileid)
    file = open("./sol/"+filename,"a")
    
    file.write('TITLE = "Solution" \n')
    file.write('VARIABLES = "x", "y", "sol" \n')
    file.write("ZONE STRANDID=1, SOLUTIONTIME= "+ str(t)+ ", I= "+str(nx+1)+", J ="+str(ny+1)+", DATAPACKING=POINT \n")
    for j in range(0, ny+1):
        for i in range(0, nx+1):
            file.write( str(xmin + i * dx) + "," + str(ymin + j * dy) +"," + str(var_u[i+2, j+2])+"\n")
        file.write("\n")
    file.close()
    fileid = fileid + 1
    
# Initialize plot
# Initialize plot
def init_plot(ax2, u0):
    # Initialize heatmap plot
    heatmap = ax2.imshow(u0, cmap='hot', interpolation='nearest', origin='lower',
                         extent=[xmin, xmax, ymin, ymax])
    ax2.set_title('Initial condition (Heatmap)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(heatmap, ax=ax2)

    plt.draw()
    plt.savefig("Q5_heat_inital_problemsheet6.pdf", format="pdf", bbox_inches="tight")
    plt.pause(0.1)  # Pause to update the plot
    plt.clf()  # Clear the figure

# Update plot
# Update plot for heatmap
def update_plot(t, u1):
    ax2 = fig.add_subplot(111)
    heatmap = ax2.imshow(u1, cmap='hot', interpolation='nearest', extent=[xmin, xmax, ymin, ymax])
    ax2.set_title(str(nx)+'X'+str(ny)+' cells, CFL = '+str(round(cfl, 3)) +
              ', t = '+str(round(t, 3)))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.colorbar(heatmap)
    
    plt.draw()
    plt.savefig("Q5_heat_final_problemsheet6.pdf", format="pdf", bbox_inches="tight")

    plt.pause(0.1)
    plt.clf()  # Clear the figure after drawing to prepare for the next update



# Fill ghost cells using periodicity
def update_ghost(v1):
    # left ghost cell
    v1[0,:] = v1[nx,:]
    v1[1,:] = v1[nx+1,:]
    # right ghost cell
    v1[nx+3,:] = v1[3,:]
    v1[nx+4,:] = v1[4,:]
    # bottom ghost cell
    v1[:,0]= v1[:,ny]
    v1[:,1]= v1[:,ny+1]
    # top ghost cell
    v1[:,ny+4] = v1[:,4]
    v1[:,ny+3] = v1[:,3]

if args.plot_freq > 0:
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    
    init_plot(ax2, v0)
    wait = input("Press enter to continue ")

# Set time step from CFL condition
dt = cfl/(1.0/dx + 1.0/dy + 1.0e-14)

iter, t = 0, 0.0
Tf = args.Tf   
#save initial data
savesol(t, v)
while t < Tf:
    if t+dt > Tf:
        dt = Tf - t
    lamx, lamy = dt/dx,  dt/dy
    # Loop over real cells (no ghost cell) and compute cell integral
    update_ghost(v)
    v_old = v.copy()
    for i in range(2, nx+3):
        for j in range(2, ny+3):
            v[i,j] = v_old[i,j] - lamx *(v_old[i,j] - v_old[i-1,j]) - lamy *(v_old[i,j] - v_old[i,j-1])
    t, iter = t+dt, iter+1
    if args.save_freq > 0:
        if iter % args.save_freq == 0:
            savesol(t, v)
    if args.plot_freq > 0:
        print('iter,t,min,max =', iter, t, v[2:nx+3,2:ny+3].min(), v[2:nx+3,2:ny+3].max())
        if iter% args.plot_freq == 0:
            update_plot(t, v[2:nx+3,2:ny+3])

# print final data
savesol(t,v)
print('iter,t,min,max =', iter, t, v[2:nx+3,2:ny+3].min(), v[2:nx+3,2:ny+3].max())
print('solution saved to .plt files')

if args.plot_freq > 0: 
    plt.show()
