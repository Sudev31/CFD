"""
number of cells = nx x ny
Solve scalar conservation law with periodic bc
To get help, type
    python lwfr.py -h
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def extract_and_plot_vtk(vtk_data, nx, ny):
    # Get the points from the VTK data
    points_vtk = vtk_data.GetPoints().GetData()
    
    # Convert VTK points to NumPy array
    points = vtk_to_numpy(points_vtk)
    
    # Reshape the points array into (nx+1, ny+1, 3) where 3 corresponds to (x, y, z)
    if points.shape[0] != (nx + 1) * (ny + 1):
        raise ValueError(f"Mismatch in points data, expected {(nx + 1) * (ny + 1)}, but got {points.shape[0]}")
    
    points = points.reshape((nx + 1, ny + 1, 3))

    # Extract x, y coordinates from points
    x = points[:, :, 0]
    y = points[:, :, 1]
    
    # Extract z (scalar values) from VTK data
    z = vtk_to_numpy(vtk_data.GetPointData().GetScalars())
    z = z.reshape((nx+1,ny+1))
    
    # Debugging: Print shapes of x, y, and z
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(f"z.shape before reshape: {z.shape}")
    
    # Ensure z has the same shape as x and y
    
    # Debugging: Check after reshape
    # Now plot using contour
    fig2, ax2 = plt.subplots()  # Create a new figure for contour plot
    cp = ax2.contour(x, y, z, levels=16)  # Use x, y, z instead of undefined xgrid, ygrid
    plt.colorbar(cp)  # Add colorbar for better visualization
    plt.title("Contour Plot at the given time step")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # Now plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    plt.draw()
    plt.show()

# Function to load VTK file
def load_vtk_file(file_path):
    # Use vtkStructuredGridReader for structured grid data
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(file_path)
    
    # Update the reader to read the data
    reader.Update()
    
    # Get the output data from the reader
    vtk_data = reader.GetOutput()
    
    return vtk_data  # Return the VTK data object

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
                    default=1)
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
v = np.zeros((nx + 5, ny + 5))  # 2 ghost points each side
# Set initial condition by interpolation
for i in range(nx + 5):
    for j in range(ny + 5):
        x = xmin + (i - 2) * dx     
        y = ymin + (j - 2) * dy
        val = initial_condition(x, y)
        v[i, j] = val
# copy the initial condition
v0 = v[2:nx + 3, 2:ny + 3].copy()
# it stores the coordinates of real cell vertices 
xgrid1 = np.linspace(xmin, xmax, nx + 1)
ygrid1 = np.linspace(ymin, ymax, ny + 1)
ygrid, xgrid = np.meshgrid(ygrid1, xgrid1)

#------------To save solution--------------------------------------------

def getfilename(file, fileid):
    if fileid < 10:
        file = file + "00" + str(fileid) + ".vtk"
    elif fileid < 99:
        file = file + "0" + str(fileid) + ".vtk"
    else:
        file = file + str(fileid) + ".vtk"
    return file

# save solution to a file
def savesol(t, var_u):
    global fileid
    if not os.path.isdir("sol"):  # create a dir if not
        os.makedirs("sol")
        print('Directory "sol" is created')
    if fileid == 0:  # remove the content of the folder
        print('The directory "sol" is going to be formatted!')
        if input('Do You Want To Continue? [y/n] ') != 'y':
            sys.exit('Execution is terminated')
        fs = glob.glob('./sol/*')
        for f in fs:
            os.remove(f)
    filename = "sol"
    filename = getfilename(filename, fileid)
    file = open("./sol/" + filename, "a")

    file.write("# vtk DataFile Version 3.0\n")
    file.write("Solution data\n")
    file.write("ASCII\n")
    file.write("DATASET STRUCTURED_GRID\n")
    file.write(f"DIMENSIONS {nx + 1} {ny + 1} 1\n")
        
    # Write grid points
    file.write(f"POINTS {(nx + 1) * (ny + 1)} float\n")
    for j in range(ny + 1):
        y = ymin + j * dy
        for i in range(nx + 1):
            x = xmin + i * dx
            file.write(f"{x} {y} 0.0\n")
        
    # Write solution data at grid points
    file.write(f"POINT_DATA {(nx + 1) * (ny + 1)}\n")
    file.write("SCALARS solution float 1\n")
    file.write("LOOKUP_TABLE default\n")
    for j in range(ny + 1):
        for i in range(nx + 1):
            file.write(f"{var_u[i + 2, j + 2]}\n")

    fileid = fileid + 1

# Initialize plot
def init_plot(ax1, ax2, u0):
    # Use the correct slicing to match the size of u0 (51x51)
    u0_real = u0[2:nx+3, 2:ny+3]  # Now matching the shape of u1 as 51x51

    # Debugging info
    print(f"xgrid shape: {xgrid.shape}")
    print(f"ygrid shape: {ygrid.shape}")
    print(f"u0_real shape: {u0_real.shape}")  # Should be (51, 51)

    cp = ax2.contour(xgrid[:nx+1], ygrid[:ny+1], u0, levels=16)  # Match grid shape
    ax2.set_title('Initial condition')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(cp)

    plt.draw()
    plt.pause(0.1)
    plt.clf()

# Update plot
# Update plot
def update_plot(t, u1):
    # Use the correct slicing to match the size of u1 (51x51)
    u1_real = u1[2:nx+3, 2:ny+3]  # Exclude ghost cells, resulting in 51x51 grid

    # Debugging info
    print(f"xgrid shape: {xgrid.shape}")
    print(f"ygrid shape: {ygrid.shape}")
    print(f"u1_real shape: {u1_real.shape}")  # Should be (51, 51)
    
    ax2 = fig.add_subplot(111)
    cp = ax2.contour(xgrid[:nx+1], ygrid[:ny+1], u1, levels=16)  # Match grid shape
    ax2.set_title(str(nx) + 'X' + str(ny) + ' cells, CFL = ' + str(round(cfl, 3)) +
              ', t = ' + str(round(t, 3)))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(cp)

    plt.draw()
    plt.savefig("Q4_final_problemsheet6.pdf", format="pdf", bbox_inches="tight")

    plt.pause(0.1)
    plt.clf()

# Fill ghost cells using periodicity
def update_ghost(v1):
    # left ghost cell
    v1[0, :] = v1[nx, :]
    v1[1, :] = v1[nx + 1, :]
    # right ghost cell
    v1[nx + 3, :] = v1[3, :]
    v1[nx + 4, :] = v1[4, :]
    # bottom ghost cell
    v1[:, 0] = v1[:, ny]
    v1[:, 1] = v1[:, ny + 1]
    # top ghost cell
    v1[:, ny + 4] = v1[:, 4]
    v1[:, ny + 3] = v1[:, 3]

if args.plot_freq > 0:
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    init_plot(ax2, ax2, v0)
    wait = input("Press enter to continue ")

# Set time step from CFL condition
dt = cfl / (1.0 / dx + 1.0 / dy + 1.0e-14)
iter, t = 0, 0.0
Tf = args.Tf   
# save initial data
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

# Load and plot VTK file
i = input('time at which you want the graph')
i = float(i)
if i<0 or i>Tf:
    print('choose within the limits')
else:
    i = int(i/dt) + 1
    if (i>9):
        vtk_file = f"sol\\sol0{i}.vtk"  # Load the last saved VTK file
    else:
        vtk_file = f"sol\\sol00{i}.vtk"  # Load the last saved VTK file
    print(i)
    vtk_data = load_vtk_file(vtk_file)
    extract_and_plot_vtk(vtk_data, nx, ny)


