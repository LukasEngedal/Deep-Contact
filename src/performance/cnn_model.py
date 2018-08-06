import numpy as np
import pandas as pd

from Box2D import b2ContactListener

from ..sph.dataframes import dataframes_from_b2World
from ..sph.grids import GridManager, grids_from_dataframes, grids_to_particles
from .util import CustomContactListener, copyWorld, getPredictions
from .models import Model

# Disables stupid tensorflow warnings about cpu instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Our CNN model
class CNNModel(Model):
    def __init__(self, cnn):
        super(CNNModel, self).__init__()

        self.cnn = cnn
        self.ContactListener = CustomContactListener()

        # We create the grid manager
        params = self.cnn.params
        self.params = params
        self.G = GridManager(
            params["p_ll"], params["p_ur"],
            params["xRes"], params["yRes"],
            params["h"]
        )


    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(CNNModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

        # Reset
        self.ContactListener.reset()

        # We create the body dataframes for the current world
        df_b, _ = dataframes_from_b2World(world)

        # We create a copy of our world, and ask it to take a step
        # This is done in order to obtain all contact information, including
        # contacts that might only appear during the step
        copy = copyWorld(world)
        copy.contactListener = self.ContactListener

        copy.Step(timeStep, velocityIterations, positionIterations)

        df_c = self.ContactListener.df_c
        N = df_c.shape[0]
        if N == 0:
            return

        # We create the grids
        params = self.params
        features, _ = grids_from_dataframes(
            self.G, df_b, df_c,
            params["body_channels"],
            params["contact_channels"],
            params["feature_channels"],
            []
        )

        # We feed the input to the cnn
        ni_grid, ti_grid = self.cnn.predict([features])
        ni_grid = np.reshape(ni_grid, (self.G.N_x, self.G.N_y))
        ti_grid = np.reshape(ti_grid, (self.G.N_x, self.G.N_y))

        # We extract contact positions
        points = []
        for i in range(N):
            row = df_c.loc[i]
            px = row[2]
            py = row[3]

            points.append([px, py])

        # Transfer from grids to contacts
        grids = np.array([ni_grid, ti_grid])
        points = np.array(points)
        values = grids_to_particles(self.G, grids, points)

        # We store the results for easy acces
        for i in range(N):
            row = df_c.loc[i]
            master = int(row[0])
            slave = int(row[1])
            key = (master, slave)

            px = row[2]
            py = row[3]

            prediction = [px, py] + list(values[i])

            if key in self.predictionDict:
                self.predictionDict[key].append(prediction)
            else:
                self.predictionDict[key] = [prediction]


    # Set impulses based on stored predictions
    def PreSolve(self, contact, old):
        super(CNNModel, self).PreSolve(contact, old)


    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(CNNModel, self).PostSolve(contact, impulse)
