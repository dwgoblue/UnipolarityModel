import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import MoveMorphoNumerical
from MatplotProp import CanvasStyle, PltProps

# Plot style.
PltProps()

# Inputs for parameters kd, ks, mss, uss.
para = [10**(-3.78), 10**(-6), 10**3, 10**2]

# Run simulation.
res = MoveMorphoNumerical.main(para)

# Settings for plotting.
max_t = int(3600000) # The last time point of simulation.
samples = 1000 # Resolution for showing trajectories of the changes of distribution.
interval = max_t/samples

# Create an empty figure.
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.grid(False)

# Draw results.
tot_sig = res['m'][:, max_t]+res['u'][:, max_t]
# Line plot for the total signals.
ax.plot(tot_sig, label='Total', color='tomato', linewidth=4)
# line plot for the membrane-bound signals.
ax.plot(res['m'][:, max_t], label='m', color='navy', linewidth=4)
# line plot for the diffuse signals.
ax.plot(res['u'][:, max_t], label='u', color='grey', linewidth=4)
# axes style.
y_min, _ = ax.get_ylim()
x_min, _ = ax.get_xlim()
ax.set_xlabel('Length ($\mu{m}$)')
ax.set_ylabel('Concentration (nM)')
ax.set_xticklabels(np.linspace(0, 2, 5))
ax.set_xticks(np.linspace(0, 9, 5))
ax = CanvasStyle(ax, x_min=x_min, y_min=y_min)
ax.legend(loc=(1.05, 0.), fontsize='24')
plt.show()

# Print out informations.
print('Ratios at the beginning of the simulation.')
print('Ratio of total signals between the first and the last grids:',
      (res['m'][0, 0]+res['u'][0, 0])/(res['m'][-1, 0]+res['u'][-1, 0]))
print('Ratio of diffuse signals between the first and the last grids:',
      (res['u'][0, 0])/(res['u'][-1, 0]))
print('Ratio of membrane-bound signals between the first and the last grids:',
      (res['m'][0, 0])/(res['m'][-1, 0]))

print('Ratios at the end of the simulation.')
print('Ratio of total signals between the first and the last grids:',
      (res['m'][0, max_t]+res['u'][0, max_t])/(res['m'][-1, max_t]+res['u'][-1, max_t]))
print('Ratio of diffuse signals between the first and the last grids:',
      (res['u'][0, max_t])/(res['u'][-1, max_t]))
print('Ratio of membrane-bound signals between the first and the last grids:',
      (res['m'][0, max_t])/(res['m'][-1, max_t]))


# Save results to a .pickle file.
import _pickle as cPickle
obj = {'u':res['u'][:, max_t], 'm':res['m'][:, max_t]}
with open('Distribution.pickle', 'wb') as picklefile:
        cPickle.dump(obj, picklefile, True)


# Trajectories for u (diffuse signals).
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111)
ax.grid(False)
diffuse_sig = res['u'] / res['u'].max(axis=0)
pic = ax.imshow(diffuse_sig[:, 0:max_t:int(interval)])
ax.set_xlabel('Time steps')
ax.set_ylabel('Length')
ax.set_title('u')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="20%", pad=1)
plt.colorbar(pic, cax=cax, orientation="horizontal")
plt.show()


# Trajectories for m (membrane-bound signals).
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111)
ax.grid(False)
mem_sig = res['m'] / res['m'].max(axis=0)
pic = ax.imshow(mem_sig[:, 0:max_t:int(interval)])
ax.set_xlabel('Time steps')
ax.set_ylabel('Length')
ax.set_title('m')
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="20%", pad=1)
plt.colorbar(pic, cax=cax, orientation="horizontal")
plt.show()