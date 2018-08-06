import numpy as np
import pandas as pd

from Box2D import b2World, b2FixtureDef, b2Vec2, b2ContactListener


# Utility class used for gathering contact values
class CustomContactListener(b2ContactListener):
    def __init__(self):
        super(CustomContactListener, self).__init__()

        self.reset()

    def reset(self):
        self.df_c = pd.DataFrame(columns = ["master", "slave", "px", "py", "nx", "ny", "ni", "ti"])
        self.n_c = 0

    def PreSolve(self, contact, _):
        contact.userData = self.n_c
        for i in range(contact.manifold.pointCount):
            worldPoint = contact.worldManifold.points[i]
            px = worldPoint[0]
            py = worldPoint[1]

            normal = contact.worldManifold.normal
            nx = normal[0]
            ny = normal[1]

            master = contact.fixtureA.body.userData.id
            slave = contact.fixtureB.body.userData.id

            self.df_c.loc[self.n_c] = [master, slave, px, py, nx, ny, 0, 0]
            self.n_c += 1

    def PostSolve(self, contact, impulse):
        n_c = contact.userData
        for i in range(contact.manifold.pointCount):
            normal = impulse.normalImpulses[i]
            tangent = impulse.tangentImpulses[i]

            self.df_c.loc[n_c+i].ni = normal
            self.df_c.loc[n_c+i].ti = tangent


# Creates a copy of the given world by creating copies of all bodies
def copyWorld(world):
    copy = b2World(gravity=world.gravity, doSleep=world.allowSleeping)

    copy.continuousPhysics  = world.continuousPhysics

    copy.velocityThreshold = world.velocityThreshold
    copy.positionThreshold = world.positionThreshold

    for body in world.bodies:
        fixtures = []
        for fixture in body.fixtures:
            fixtures.append(b2FixtureDef(
                shape=fixture.shape,
                density=fixture.density,
                restitution=fixture.restitution,
                friction=fixture.friction
            ))

        copy.CreateBody(
            type=body.type,
            fixtures=fixtures,
            userData=body.userData,
            position=b2Vec2(body.position.x, body.position.y),
            angle=body.angle,
            linearVelocity=b2Vec2(body.linearVelocity.x, body.linearVelocity.y),
            angularVelocity=body.angularVelocity
        )

    for body in copy.bodies:
        body.sleepingAllowed = False


    return copy


# Stores normal and tangential impulse predictions for a contact in a dictionary
# The fact that there might be more than one contact between bodies makes this
# a bit more complicated, it is solved by also using coordinates
def storePredictions(predictionDict, contact, predictions):
    idA = contact.fixtureA.body.userData.id
    idB = contact.fixtureB.body.userData.id
    key = (idA, idB)

    if key in predictionDict:
        storedPredictions = predictionDict[key]
    else:
        storedPredictions = []
        predictionDict[key] = storedPredictions

    for i in range(len(predictions)):
        worldPoint = contact.worldManifold.points[i]
        px = worldPoint[0]
        py = worldPoint[1]

        ni = predictions[i][0]
        ti = predictions[i][1]

        storedPredictions.append([px, py, ni, ti])


# Stores the normal and tangential impulse prediction for a contact from a dictionary
# The fact that there might be more than one contact between bodies makes this
# a bit more complicated, it is solved by also using coordinates
def getPredictions(predictionDict, contact):
    idA = contact.fixtureA.body.userData.id
    idB = contact.fixtureB.body.userData.id
    key = (idA, idB)

    if key in predictionDict:
        storedPredictions = predictionDict[(idA, idB)]
    else:
        print("Contact not found in prediction dict")
        return [[0,0], [0,0]]

    predictions = []
    for i in range(contact.manifold.pointCount):
        worldPoint = contact.worldManifold.points[i]
        px = worldPoint[0]
        py = worldPoint[1]

        # We choose the predictions whose position is closest to the position of the contact point
        # Might be bad in case of extreme behaviour like very high velocities
        pred = min(storedPredictions, key=lambda p: (px-p[0])**2 + (py-p[1])**2)

        predictions.append(pred[2:4])

    return predictions


# Function for smoothing data using a moving average as far as I recall
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
