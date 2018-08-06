import numpy as np

from itertools import product
from Box2D import (b2World, b2Vec2, b2_dynamicBody)

from ..gen_world import new_confined_clustered_circles_world

from .grids import GridManager, particles_to_grids, grids_to_particles
from .dataframes import dataframes_from_b2World


# Number of worlds to generate
nWorlds = 25
# Number of bodies in world
nBodies = 100
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
p_ll = (0, 0)
p_ur = (15, 15)
# Radius of bodies
r = (0.5, 0.5)

# Create worlds
worlds = [b2World() for _ in range(nWorlds)]
# Fill worlds with static box and circles
for i in range(nWorlds):
    new_confined_clustered_circles_world(worlds[i], nBodies, b2Vec2(p_ll), b2Vec2(p_ur), r, sigma_coef, i)
# Simulate 200 steps for worlds
for w in worlds:
    for j in range(200):
        w.Step(0.01, 10, 10)

# Determine original totals - change attribute here if wanted
originals = []
for w in worlds:
    original = 0
    for c in w.contacts:
        if c.touching:
            original += c.manifold.points[0].normalImpulse
    originals.append(original)
# Attribute to create grids for
attribute = "ni"


# Grid parameters to try
# Grid resolution
res = [0.125, 0.25, 0.5, 0.75, 1]
# Support radius
h = [0.125, 0.25, 0.5, 0.75, 1]

# Create all combinations of grid parameters
parameters = list(product(*[res, h]))
nParameters = len(parameters)

# Create grid managers, tell them to create grids, and query for all bodies in world
worldTotalDifferenceAverages = [.0]*nParameters
contactDifferenceAverageAverages = [.0]*nParameters
for i in range(nParameters):
    p = parameters[i]

    print("Trying parameter set %2d of %2d" % (i+1, nParameters))

    # Create gridmanager
    G = GridManager(p_ll, p_ur, p[0], p[0], p[1])

    worldTotalDifferences = [.0]*nWorlds
    bodyDifferenceAverages = [.0]*nWorlds
    for j in range(nWorlds):
        world = worlds[j]

        # Create data frames
        _, df_c = dataframes_from_b2World(world)

        # Tell GM to create grid
        _, grid = particles_to_grids(G, df_c, [attribute])[0]

        # Extract contacts and their positions
        contacts = [c for c in world.contacts if c.touching]
        nContacts = len(contacts)
        points = np.array([[c.worldManifold.points[0][0], c.worldManifold.points[0][1]] for c in contacts])

        # Tell GM to transfer from grids to bodies
        values = grids_to_particles(G, np.array([grid]), points)

        # Query for all contacts and sum
        worldTotal = 0
        contactDifferences = [.0]*nContacts
        k = 0
        for k in range(nContacts):
            c = contacts[k]
            value = values[k, 0]
            worldTotal += value
            contactDifferences[k] = abs(c.manifold.points[0].normalImpulse - value)

        # Store results
        worldTotalDifferences[j] = abs(originals[j] - worldTotal)
        bodyDifferenceAverages[j] = np.mean(contactDifferences)

    # Averate results and store
    worldTotalDifferenceAverages[i] = np.mean(worldTotalDifferences)
    contactDifferenceAverageAverages[i] = np.mean(bodyDifferenceAverages)

# Combine averages with parameters
pairs = list(zip(parameters, worldTotalDifferenceAverages, contactDifferenceAverageAverages))

# Sort pairs
sortedPairs = sorted(pairs, key=lambda p: p[2])

# Print results
print("Original total: {0:3.2f}".format(originals[0]))
print("({0:4s}, {1:4s})\t{2:14s}\t{3:13s}".format("res", "h", "Avg total diff", "Avg contact diff"))
for p, t, b in sortedPairs:
    print("({0:1.2f}, {1:1.2f})\t{2:9.4f}\t{3:8.4f}".format(p[0], p[1], t, b))
