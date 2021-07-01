import math
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("TkAgg")

# First, I'll set some constants
# at the top of the file, for convenience.

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

# Setup the figure in pyplot.
fig = plt.figure(1, figsize=(5, 5))
fig.patch.set_facecolor((0.1, 0.1, 0.1, 1))

# Set up the axes.
ax = plt.axes(xlim=(0, width), ylim=(0, width))
ax.set_facecolor((0.0, 0.0, 0.0, 1))

# Make some star X, Y positions.
#
# Note that x and y are vectors of length star_count.
# As a general rule, things associated with each star
# will be kept in vectors, with the vector index
# being the star ID.
#
# These will be initialized later,
# but creating them here lets us
# go ahead and add them to the plot.
x = np.zeros(star_count)
y = np.zeros(star_count)

# Add the stars to the axis.
# The colors and alpha can be adjusted
# depending on your preferences.
scat = ax.scatter(x, y, s=1, c=(0.2, 0.2, 0.5, 0.3))

# Assign a mass to each star.
# With the number of stars being so low,
# and mass being handled by a fudge factor,
# things tend to work best if mass
# does not vary too widely.
mass = np.random.uniform(0.8, 1.0, star_count)

# Cache some things related to mass.
mass_max = mass.max(initial=0)
mass_mean = mass.mean()
# Fraction of the total mass in each star.
mass_fraction = mass / mass.sum()
# Matrix of product of all possible mass pairs.
massProds = np.outer(mass, mass)

# Expanded, the above might look like this:
#
#     for si0 in range(0, len(mass))
#         for si1 in range(0, len(mass))
#             massProds[si0][si1] = mass[si0] * mass[si1]
#
# This is very slow in Python.
# NumPy computes the above in C, which is much faster.

# "net" holds the net motion of each star,
# all things considered.
# At the start, there is no motion.
x_net = np.zeros(star_count)
y_net = np.zeros(star_count)


# Because some initial distributions may
# be off-center, or because of later
# rounding errors, etc.,
# the stars as a whole will sometimes drift.
# This will re-center the cluster to
# the center of mass.
def center_mass():

    global x, y

    x_mass_center = np.sum(x * mass_fraction)
    y_mass_center = np.sum(y * mass_fraction)

    x = (x - x_mass_center) + width_2
    y = (y - y_mass_center) + width_2


# These next few functions can be used to initialize
# the star XY positions to various configurations.

# Creates an initial distribution of stars having random clusters.
def hot_spots(
        spot_count
):
    # Make spot_count "super clusters."

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

    # Make smaller clusters around each "super cluster."
    
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


# Creates an initial distribution of stars in a uniform rectangle.
def rectangle(
        x_size,
        y_size
):
    global x, y

    x_size = x_size / 2
    y_size = y_size / 2

    x = np.random.uniform(0.5 - x_size, 0.5 + x_size, star_count)
    y = np.random.uniform(0.5 - y_size, 0.5 + y_size, star_count)


# Creates an initial distribution of stars in a circle.
#
# If normal is False:
#     Create a ring of stars where the radius is uniformly distributed
#     between a and b.
#
# If normal is True:
#     Create a ring of stars where the mean radius of the ring is (a + b) / 2,
#     and the standard deviation is (b - a).
#
def circle(
        a,
        b,
        normal,
        star_min,
        star_max
):
    global x, y

    if normal:
        func = np.random.normal
        a0 = (a + b) / 2
        b = (b - a)
        a = a0
    else:
        func = np.random.uniform

    for p in range(star_min, star_max):
        h_polar_angle_xy = np.random.uniform(0, np.pi * 2)
        h_polar_distance = func(a, b)
        x[p] = math.sin(h_polar_angle_xy) * h_polar_distance
        y[p] = math.cos(h_polar_angle_xy) * h_polar_distance


# Create the scatter graph.
# The colors and alpha can be adjusted
# depending on your preferences.
graph = ax.scatter(
    x,
    y,
    s=1,
    c=(1, 1, 1, 0.3)
)

# Initialize the stars.
hot_spots(5)
# Various interesting intializations:
# rectangle(0.5, 0.5)
# rectangle(0.01, 0.6)
# circle(0.1, 0.105, True, 0, star_count)
# three = star_count // 3
# circle(0.35, 0.351, True, 0, three)
# circle(0.20, 0.201, True, three, three * 2)
# circle(0.37, 0.371, True, three * 2, star_count)

# Center the mass in the view.
center_mass()

# Animation callback.
def animate(_frame):

    global graph
    global x, y
    global x_net, y_net
    global massProds

    # Generate x, y, distance matrices from the vectors.
    # Note that this is a little wasteful,
    # since we just need the triangular differences.

    xd = np.subtract.outer(x, x)
    yd = np.subtract.outer(y, y)

    # Compute the squared distances.
    # Add a fudge factor to prevent division by zero.

    distances = xd * xd + yd * yd + distance_epsilon

    # Experiment with weird things,
    # like repulsion at close distances!
    # rev = -0.00000001
    # distances = xd * xd + yd * yd + rev
    # mask = distances == 0
    # distances[mask] = rev

    # Divide each pair of mass sums by the distance.
    # Multiply by an empirically-determined scaling factor.

    attraction = massProds / distances * attraction_scale

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
