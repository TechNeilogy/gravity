import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as py
from matplotlib import animation
import math

# Represents the smallest considered distance value.
# Prevents division by zero when calculating attractions
# for close together stars.
distance_epsilon = 0.000001

# A dimensionless quantity that scales the attraction
# for display size and animation speed.
# Empirically determined by watching the animation.
attraction_scale = 0.000000005

# A dimensionless quantity that imparts a "drag"
# to space itself.
ambient_friction = 0.9999

# Number of stars.
star_count = 2000

# Display width.
width = 1.0

# Display center.
width_2 = width / 2

# Setup the figure.
fig = py.figure(1, figsize=(5, 5))
fig.patch.set_facecolor((0.1, 0.1, 0.1, 1))

# Set up the axis.
ax = py.axes(xlim=(0, width), ylim=(0, width))
ax.set_facecolor((0.0, 0.0, 0.0, 1))

# Make some star X, Y positions.
x = np.zeros(star_count)
y = np.zeros(star_count)

# Add the stars to the axis.
scat = ax.scatter(x, y, s=1, c=(0.2, 0.2, 0.5, 0.3))

# Assign a mass to each star.
# With the number of stars being so low,
# things tend to work best if mass
# does not vary too widely.
mass = np.random.uniform(0.8, 1.0, star_count)
#mass = np.random.normal(0.7, 0.3, star_count)

# Cache some things related to mass.
mass_max = mass.max()
mass_mean = mass.mean()
mass_fraction = mass / mass.sum()
massSums = np.add.outer(mass, mass)

# "net" hold the net motion of each star,
# all things considered.
# At the start, there is no motion.
x_net = np.zeros((star_count))
y_net = np.zeros((star_count))

# Because some initial distributions may
# be off-center, or because of later
# rounding errors, etc.,
# the stars as a whole will sometimes drift.
# This will re-center the cluster to
# the center of mass.
def center_mass():

    global  x, y

    x_mass_center = np.sum(x * mass_fraction)
    y_mass_center = np.sum(y * mass_fraction)

    x = (x - x_mass_center) + width_2
    y = (y - y_mass_center) + width_2

# Creates an initial distribution of stars
# having random clusters.
def hot_spots(spot_count):

    h_x = np.zeros(spot_count)
    h_y = np.zeros(spot_count)

    h_polar_angle_xy = np.random.rand(spot_count) * np.pi * 2

    mn = 0.0
    mx = 0.75
    md = mx - mn

    h_polar_distance = (np.random.rand(spot_count) * md + mn) * width_2

    for p in range(0, spot_count):
        h_x[p] = math.sin(h_polar_angle_xy[p]) * h_polar_distance[p] + width_2
        h_y[p] = math.cos(h_polar_angle_xy[p]) * h_polar_distance[p] + width_2

    global x, y

    polar_angle_xy = np.random.rand(star_count) * np.pi * 2

    mn = 0.0
    mx = np.random.rand(star_count) / 5
    md = mx - mn

    polar_distance = (np.random.rand(star_count) * md + mn)

    for p in range(0, star_count):
        spot = np.random.randint(0, spot_count)
        x[p] = math.sin(polar_angle_xy[p]) * polar_distance[p] + h_x[spot]
        y[p] = math.cos(polar_angle_xy[p]) * polar_distance[p] + h_y[spot]

# Creates an initial distribution of stars
# in a uniform rectangle.
def rectangle(
        x_size,
        y_size
):
    global x, y

    x_size = x_size / 2
    y_size = y_size / 2

    x = np.random.uniform(0.5 - x_size, 0.5 + x_size, star_count)
    y = np.random.uniform(0.5 - y_size, 0.5 + y_size, star_count)

# Create the scatter graph.
graph = ax.scatter(
    x,
    y,
    s=1,
    c=(1, 1, 1, 0.3)
)

# Initialize the stars.
# rectangle(0.5, 0.5)
hot_spots(5)
center_mass()

# Animation callback.
def animate(_frame):

    global graph
    global x, y
    global x_net, y_net
    global massSums

    # Generate x, y, distance matrices from the vectors.
    # Note that this is a little wasteful,
    # since we just need the triangular differences.

    xd = np.subtract.outer(x, x)
    yd = np.subtract.outer(y, y)

    # Compute the squared distances.
    # Add a fudge factor to prevent division by zero.

    distances = xd * xd + yd * yd + distance_epsilon

    # Divide each pair of mass sums, by the distance.
    # Multiply by an empirically-determined scaling factor.

    attraction = massSums / distances * attraction_scale

    # Compute the sums of the distances and the attractions.
    # Note that everything is added twice,
    # but the scaling above can be used to take care of that.

    x_delta = np.sum(xd * attraction, 0)
    y_delta = np.sum(yd * attraction, 0)

    # Compute the change in net motion.

    x_net = x_delta + x_net * ambient_friction
    y_net = y_delta + y_net * ambient_friction

    # Update the positions.

    x += x_net
    y += y_net

    # If the cluster starts to drift,
    # turn this on.
    # center_mass()

    # Reshape and set the animation data.

    data = np.hstack((
        x[:star_count, np.newaxis],
        y[:star_count, np.newaxis]
    ))

    graph.set_offsets(data)

    return graph,

# Start the animation loop.
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(x),
    interval=5,
    blit=True,
    repeat=True
)

# Show the plot.
plt.show()
