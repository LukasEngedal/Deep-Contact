import numpy as np

from scipy import spatial, interpolate

from Box2D import b2World, b2_dynamicBody
from .kernels import W_poly6_2D


# Manages grid information like sizes and resolution
class GridManager:
    def __init__(self, p_ll, p_ur, xRes, yRes, h):
        xlo, ylo = p_ll
        xhi, yhi = p_ur

        self.h = h
        self.x = np.arange(xlo, xhi+xRes, xRes)
        self.y = np.arange(ylo, yhi+xRes, yRes)
        self.X, self.Y = np.mgrid[xlo:(xhi+xRes):xRes, ylo:(yhi+yRes):yRes]
        self.N_x = round((p_ur[0] - p_ll[0] + xRes) / xRes)
        self.N_y = round((p_ur[1] - p_ll[1] + yRes) / yRes)


# Creates a set of grids of values using SPH - returns (name, grid) pairs
def particles_to_grids(G, df, channels, f_krn=W_poly6_2D):
    '''
    splatters the points and their values onto grids, one grid per value, using SPH
    :param G:        Grid manager object
    :param df:       Pandas dataframe containing point data
    :param channels: List of channels to create grids for
    :param f_krn:    The SPH kernel to use
    :return:         List of channel, grid pairs
    '''
    n = len(channels)
    if n == 0:
        return []

    # We unpack the needed dataframe information
    Px, Py = df.px, df.py
    values = df[channels].values

    # We create the grids
    Gx_sz, Gy_sz = G.X.shape
    h = G.h
    grids = [(c, np.zeros((Gx_sz, Gy_sz), dtype=np.float32)) for c in channels] # TODO: change to sparse

    # Create array where each row is a x- and y-coordinate of a node in the grid
    P_grid   = np.c_[G.X.ravel(), G.Y.ravel()]
    P_points = np.c_[Px, Py]

    # For each point we determine all grid nodes within radius h
    KDTree = spatial.cKDTree(P_grid)
    NNs = KDTree.query_ball_point(P_points, h)

    # For each point
    for i in range(NNs.shape[0]):
        # We determine distances between point and neighbouring grid nodes
        neighbours = NNs[i]
        rs = P_points[i] - P_grid[neighbours]

        # We determine weights for each neighbouring grid node
        Ws = f_krn(rs.T, h)

        # For all neighbors, multiply weight with all point values and store in grid
        for j in range(len(neighbours)):
            gxi, gyi = np.unravel_index(neighbours[j], (Gx_sz, Gy_sz))
            for k in range(n):
                grids[k][1][gxi, gyi] += Ws[j] * values[i, k]

    return grids


# Takes a list of particles and a list of grids as input, and returns a list
# of lists of values, one list for each particle with one value for each grid
def grids_to_particles(G, grids, points):
    '''
    gathers particles values from grid nodes using SPH
    :param G:      Grid manager object
    :param grids:  List of grids of values
    :param points: Positions of particles
    :return:       List of values for the particles
    '''
    # We need arrays
    grids = np.array(grids)
    points = np.array(points)

    grid_nodes = np.c_[G.X.ravel(), G.Y.ravel()]
    N = grid_nodes.shape[0]

    # We create the tree
    tree = spatial.cKDTree(points)

    # For each grid point, determine neighbouring particles
    NNs = tree.query_ball_point(grid_nodes, G.h)

    # An array for storing a value for each grid for each particle
    values = np.zeros([points.shape[0], grids.shape[0]], dtype=np.float64)

    # For each grid node
    for i in range(N):
        # We determine distances between grid node and neighbouring particles
        neighbours = NNs[i]
        if len(neighbours) > 0:
            rs = grid_nodes[i] - points[neighbours]

            # We determine weights for each neighbouring particle
            Ws = W_poly6_2D(rs.T, G.h)

            # iy and ix are the index into the unflattened array
            iy, ix = np.unravel_index(i, G.X.shape)

            # We multiply weights with values to get the particle values
            for j in range(len(neighbours)):
                n = neighbours[j]
                w = Ws[j]
                for k in range(grids.shape[0]):
                    values[n, k] += w * grids[k, iy, ix]

    return values


# A class for creating a managing interpolations functions given grids
class InterpolationManager:
    def __init__(self):
        self.f_interps = {}

    # The user specifies a grid to calculate interpolation for, and a name
    def create_interp(self, G, grid, name):
        self.f_interps[name] = interpolate.RectBivariateSpline(G.x, G.y, grid)

    # Only intended to be used to query for a single point at a time
    def query_interp(self, name, Px, Py):
        return self.f_interps[name](Px,Py)[0][0]


#
def grids_from_dataframes(
        G,
        df_b,
        df_c,
        body_channels,
        contact_channels,
        feature_channels,
        label_channels,
):
    b_grids = particles_to_grids(G, df_b, body_channels)
    c_grids = particles_to_grids(G, df_c, contact_channels)
    grids = b_grids + c_grids


    # We separate the feature and the label grids
    labels = []
    features = []
    staticMass = 100
    for c, grid in grids:
        if c == "mass":
            for i in range(G.N_x):
                grid[i, 0] = staticMass
            for i in range(G.N_y):
                grid[0, i] = staticMass
                grid[G.N_x-1, i] = staticMass

        if c in feature_channels:
            features.append(grid)
        if c in label_channels:
            labels.append(grid)

    # We reshape our data
    features = np.array(features, dtype=np.float32)
    features = np.rollaxis(features, 0, 3)

    labels = np.array(labels, dtype=np.float32)
    labels = np.ndarray.flatten(labels)

    return features, labels
