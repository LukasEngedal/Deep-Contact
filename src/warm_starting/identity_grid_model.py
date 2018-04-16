from .model import Model
from ..sph.gridsplat import SPHGridManager

class IdentityGridModel (Model):
    def __init__(self, world, p_ll, p_ur, xRes, yRes, h):
        # Initialize the grid
        self.grid = SPHGridManager(world, p_ll, p_ur, xRes, yRes, h)


    def Step(self, world, timeStep, velocityIterations, positionIterations):
        # Tell the grid to update
        self.grid.Step(["normal_impulse", "tangent_impulse"])


    def Predict(self, contact):
        predictions = []

        for i in range(contact.manifold.pointCount):
            px = contact.worldManifold.points[i][0]
            py = contact.worldManifold.points[i][1]

            id = contact.manifold.points[i].id
            normalImpulse = self.grid.query(px, py, "normal_impulse")
            tangentImpulse = self.grid.query(px, py, "tangent_impulse")

            predictions.append((id, normalImpulse, tangentImpulse))
        return predictions
