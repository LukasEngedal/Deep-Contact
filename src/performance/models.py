import random
import pandas as pd

from Box2D import b2ContactListener

from .util import CustomContactListener, copyWorld, storePredictions, getPredictions
from ..sph.grids import GridManager, particles_to_grids, grids_to_particles
from..sph.dataframes import dataframes_from_b2World

# A model has three functions - __init__, Step and PreSolve.
# __init__ takes different input for different models, depending on what they need
# Step and PreSolve takes the same input for all models, irregardless of whether a
# specific model needs the input, in order to make switching models easy
# NOTE: This is not actually a model intended to be used, simply an example and a
# way to unify some behaviour across other models
class Model(b2ContactListener):
    # Initializes the Model
    def __init__(self):
        super(Model, self).__init__()

    # Tells the model to take a step, and typically create and store predictions
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        self.predictionDict = {}
        self.normalPairs = []
        self.tangentPairs = []

    # Takes a contact and sets the contact manifold points' normalImpulse and
    # tangentImpulse values, typically using the set predictions
    # Third argument is unused
    def PreSolve(self, contact, _):
        predictions = getPredictions(self.predictionDict, contact)

        for i in range(contact.manifold.pointCount):
            point = contact.manifold.points[i]
            point.normalImpulse = predictions[i][0]
            point.tangentImpulse = predictions[i][1]

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        predictions = getPredictions(self.predictionDict, contact)

        for i in range(contact.manifold.pointCount):
            pred_ni = predictions[i][0]
            pred_ti = predictions[i][1]

            res_ni = impulse.normalImpulses[i]
            res_ti = impulse.tangentImpulses[i]

            self.normalPairs.append((pred_ni, res_ni))
            self.tangentPairs.append((pred_ti, res_ti))


# A model which effectively disables warm-starting, by using 0's as starting iterates
class NoWarmStartModel(Model):
    def __init__(self):
        super(NoWarmStartModel, self).__init__()

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(NoWarmStartModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    # Predicts 0's
    def PreSolve(self, contact, _):
        for i in range(contact.manifold.pointCount):
            point = contact.manifold.points[i]
            point.normalImpulse = 0
            point.tangentImpulse = 0

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        for i in range(contact.manifold.pointCount):
            ni = impulse.normalImpulses[i]
            ti = impulse.tangentImpulses[i]

            self.normalPairs.append((0, ni))
            self.tangentPairs.append((0, ti))



# A model which does nothing, resulting in the simulator using the built-in warm-starting
class BuiltinWarmStartModel(Model):
    def __init__(self):
        super(BuiltinWarmStartModel, self).__init__()

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(BuiltinWarmStartModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    # Uses last steps results as predictions
    def PreSolve(self, contact, _):
        predictions = []
        for p in contact.manifold.points:
            predictions.append([p.normalImpulse, p.tangentImpulse])

        storePredictions(self.predictionDict, contact, predictions)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(BuiltinWarmStartModel, self).PostSolve(contact, impulse)



# Provides a very bad prediction, irregardless of input
class BadModel(Model):
    def __init__(self):
        super(BadModel, self).__init__()
        self.p = 50

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(BadModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    def PreSolve(self, contact, _):
        for point in contact.manifold.points:
            point.normalImpulse = self.p
            point.tangentImpulse = self.p

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        for i in range(contact.manifold.pointCount):
            ni = impulse.normalImpulses[i]
            ti = impulse.tangentImpulses[i]

            self.normalPairs.append((self.p, ni))
            self.tangentPairs.append((self.p, ti))



# A model which predicts a random, some-what 'reasonable' set of impulses
class RandomModel(Model):
    # We manually choose a seed to ensure the same 'random' numbers each time
    def __init__(self, seed):
        super(RandomModel, self).__init__()

        random.seed(seed)

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(RandomModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    def PreSolve(self, contact, _):
        predictions = []
        for point in contact.manifold.points:
            # Normal impulses seems to be in the range 0 to 5
            point.normalImpulse = random.uniform(0, 5)

            # Tangential impulses seems to be in the range -2 to 2
            point.tangentImpulse = random.uniform(-2, 2)

            predictions.append([point.normalImpulse, point.tangentImpulse])

        storePredictions(self.predictionDict, contact, predictions)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(RandomModel, self).PostSolve(contact, impulse)



# A model which creates a copy of the current world, asks that copy to take a step,
# and reports the results back to the original world rounded to set accuracy
class CopyWorldModel(Model):
    # 'accuracy' is the argument passed to round, i.e. the number of decimals to round up to
    # if not set, no rounding is done
    def __init__(self, accuracy=None):
        super(CopyWorldModel, self).__init__()

        self.accuracy = accuracy

    # Creates a copy of the world, tells it to take a step
    # and prepares a dictionary with impulse-results for use by Predict
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(CopyWorldModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

        copy = copyWorld(world)
        copy.Step(timeStep, velocityIterations, positionIterations)

        for contact in copy.contacts:
            res = []
            for p in contact.manifold.points:
                if self.accuracy != None:
                    normal = round(p.normalImpulse, self.accuracy)
                    tangent = round(p.tangentImpulse, self.accuracy)
                else:
                    normal = p.normalImpulse
                    tangent = p.tangentImpulse

                res.append([normal, tangent])

            storePredictions(self.predictionDict, contact, res)

    # Predict a result by looking up in dictionary
    def PreSolve(self, contact, old):
        super(CopyWorldModel, self).PreSolve(contact, old)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(CopyWorldModel, self).PostSolve(contact, impulse)



# A model which takes the impulses from last step, similar to how the build-in
# warm start works, but then transfers them onto a grid, from the grid back
# to the particles, and then uses the results as predictions
class IdentityGridModel(Model):
    def __init__(self, p_ll, p_ur, xRes, yRes, h):
        super(IdentityGridModel, self).__init__()

        # Initialize the grid
        self.G = GridManager(p_ll, p_ur, xRes, yRes, h)

        # Create custom contact listener
        self.ContactListener = CustomContactListener()

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(IdentityGridModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

        # Reset
        self.ContactListener.reset()

        # Create a copy of world and step
        copy = copyWorld(world)
        copy.contactListener = self.ContactListener

        copy.Step(timeStep, velocityIterations, positionIterations)

        df_c = self.ContactListener.df_c
        N = df_c.shape[0]
        if N == 0:
            return

        # Transfer from particles to grids
        grids = particles_to_grids(self.G, df_c, ["ni", "ti"])
        grids = [p[1] for p in grids]

        # Extract contact positions
        points = [list(df_c.loc[i][2:4]) for i in range(N)]

        # Transfer from grids to particles
        values = grids_to_particles(self.G, grids, points)

        # Store predictions
        for i in range(N):
            row = df_c.loc[i]
            master = int(row[0])
            slave = int(row[1])
            key = (master, slave)

            prediction = points[i] + list(values[i])

            if key in self.predictionDict:
                self.predictionDict[key].append(prediction)
            else:
                self.predictionDict[key] = [prediction]

    # Set impulses based on stored predictions
    def PreSolve(self, contact, old):
        super(IdentityGridModel, self).PreSolve(contact, old)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(IdentityGridModel, self).PostSolve(contact, impulse)
