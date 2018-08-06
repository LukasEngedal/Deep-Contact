import os
import time
import numpy as np

from .util import load_xml_return_grid


# Grid parameters
# Grid lower left point
p_ll = (0, 0)
# Grid upper right point
p_ur = (15, 15)
# Grid x-resolution
xRes = 0.25
# Grid y-resolution
yRes = 0.25
# Support radius
h = 0.5

# The available body attributes are:
# mass, inertia, vx, vy, omega, theta
# The body attributes that we will use are
body_channels = ["mass", "vx", "vy", "omega"]

# The available contact attributes are:
# nx, ny, ni, ti
# The contact attributes that we will use are
contact_channels = ["nx", "ny", "ni", "ti"]

# The body and contact attributes to use as input
feature_channels = ["mass", "vx", "vy", "omega", "nx", "ny"]
# The body and contact attributes to use as label
label_channels = ["ni", "ti"]

# Data information
xml_path  = "../gen_data/data/xml1515100/"
grid_path = "../gen_data/data/grid1515100_025_05/"
numbers = range(110)
steps = 250


# ----- Misc -----
# If path is not absolute, or folder does not exist, fix it
if not os.path.isabs(grid_path):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    grid_path = os.path.join(file_dir, grid_path)
if not os.path.exists(grid_path):
    os.makedirs(grid_path)


# ----- Data generation -----
for i in range(len(numbers)):
    print("Processing dataset %d of %d" % (i+1, len(numbers)))
    start = time.time()
    n = numbers[i]

    # Load the xml file and convert to grids
    features, labels = load_xml_return_grid(
        xml_path, n, steps,
        body_channels, contact_channels,
        feature_channels, label_channels,
        p_ll, p_ur, xRes, yRes, h
    )

    # Save the grids using numpy
    file_name = grid_path + str(n)
    np.savez(file_name, features=features, labels=labels)
    print("Processing took %d s" % (time.time() - start))
