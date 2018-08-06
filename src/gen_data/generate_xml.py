import cv2
import os

from Box2D import b2World, b2Vec2

from ..gen_world import new_confined_clustered_circles_world
from ..sim_types import SimData
from ..xml_writing.b2d_2_xml import XMLExporter
from ..opencv_draw import OpencvDrawFuncs
from .util import ContactListener


# ----- Parameters -----
# Number of bodies in worlds
nBodies = 100
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
xlow, xhi = 0, 15
ylow, yhi = 0, 15
# body radius min and max
radius = (0.5, 0.5)
# Seeds to use for body generator
seeds = range(110)

# Timestep
timeStep = 1.0 / 100
# Iteration limits
velocityIterations = 1000
positionIterations = 250
# Iteration thresholds
velocityThreshold = 1*10**-4
positionThreshold = 1*10**-4
# Number of steps
steps = 250

# Path to store data
path = "../gen_data/data/xml1515100/"

# Decides whether to store configurations without any contacts
skipContactless = True
# Print various iteration numbers as simulation is running
quiet = True
# Show visualization of world as simulation is running
# note: significantly slower
visualize = False


# ----- Misc -----
# If path is not absolute, or folder does not exist, fix it
if not os.path.isabs(path):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(file_dir, path)
if not os.path.exists(path):
    os.makedirs(path)


# ----- Data generation -----
if __name__ == '__main__':
    for i in range(len(seeds)):
        # image generation
        drawer = OpencvDrawFuncs(w=300, h=300, ppm=10)
        drawer.install()

        seed = seeds[i]
        print("Running world %d of %d" % (i+1, len(seeds)))

        # Create world
        world = b2World()
        world.userData = SimData(
            name=str(seed),
            d_t=timeStep,
            vel_iter=velocityIterations,
            pos_iter=positionIterations,
            vel_thres=velocityThreshold,
            pos_thres=positionThreshold
        )

        # Fill world with static box and circles
        new_confined_clustered_circles_world(
            world,
            nBodies,
            b2Vec2(xlow, ylow),
            b2Vec2(xhi, yhi),
            radius,
            sigma_coef,
            seed
        )

        # Set iteration thresholds
        world.velocityThreshold = velocityThreshold
        world.positionThreshold = positionThreshold

        # Initialize XML exporter
        xml_exp = XMLExporter(world, path)

        # Initialize contact listener
        listener = ContactListener(xml_exp)
        world.contactListener = listener

        # Run simulation
        for step in range(steps):
            if not quiet:
                print("step", step)

            # Reset the contact listener
            listener.reset()

            # Reset xml exporter and take snapshot of bodies
            xml_exp.reset()
            xml_exp.snapshot_bodies()

            # Tell the world to take a step
            world.Step(timeStep, velocityIterations, positionIterations)
            world.userData.tick()
            world.ClearForces()

            # Draw the world
            if visualize:
                drawer.clear_screen()
                drawer.draw_world(world)

                cv2.imshow('World', drawer.screen)
                cv2.waitKey(25)

            # Save the snapshot if wanted
            if not skipContactless or world.GetProfile().contactsSolved > 0:
                xml_exp.save_snapshot()
