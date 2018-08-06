import time
import numpy as np
import cv2

from Box2D import b2ContactListener

from ..opencv_draw import OpencvDrawFuncs


# Run a specific simulation, using the various specified options and values
def run_world(
        world,
        timeStep,
        steps,
        velocityIterations,
        positionIterations,
        velocityThreshold = 0,
        positionThreshold = 1000,
        model = None,
        iterations = False,
        convergenceRates = False,
        lambdaErrors = False,
        quiet = True,
        visualize = False,
        export_path = None,
):

    # ----- Setup -----
    # Enable/disable convergence rates
    world.convergenceRates = convergenceRates

    # Set iteration thresholds
    world.velocityThreshold = velocityThreshold
    world.positionThreshold = positionThreshold

    # Attach model as listener if given a model
    if model:
        world.contactListener = model
    else:
        world.warmStarting = False

    # Define a drawer if set
    drawer = OpencvDrawFuncs(w=500, h=500, ppm=10)
    drawer.install()

    # We store the performance data in a dictionary
    result = {}

    # ----- Run World -----
    totalStepTimes          = []
    contactsSolved          = []
    totalVelocityIterations = []
    totalPositionIterations = []
    velocityLambdaTwoNorms  = []
    velocityLambdaInfNorms  = []
    positionLambdas         = []
    normalPairs             = []
    tangentPairs            = []
    for i in range(steps):
        if not quiet:
            print("step", i)

        # Start step timer
        step = time.time()

        # Tell the model to take a step
        if model:
            model.Step(world, timeStep, velocityIterations, positionIterations)

        # Tell the world to take a step
        world.Step(timeStep, velocityIterations, positionIterations)
        world.ClearForces()

        # Determine total step time
        step = time.time() - step
        totalStepTimes.append(step)

        # Draw the world
        drawer.clear_screen()
        drawer.draw_world(world)

        if visualize:
            cv2.imshow('World', drawer.screen)
            cv2.waitKey(25)

        if export_path:
            if not path.exists(export_path):
                mkdir(export_path)
            cv2.imwrite(
                path.join(export_path, '{}.png'.format(i)),
                drawer.screen)

        # Extract and store profiling data
        profile = world.GetProfile()

        contactsSolved.append(profile.contactsSolved)

        if iterations:
            totalVelocityIterations.append(profile.maxIslandVelocityIterations)
            totalPositionIterations.append(profile.maxIslandPositionIterations)

        if convergenceRates:
            velocityLambdaTwoNorms.append(profile.velocityLambdaTwoNorms)
            velocityLambdaInfNorms.append(profile.velocityLambdaInfNorms)
            positionLambdas.append(profile.positionLambdas)

        if lambdaErrors:
            normalPairs.append(model.normalPairs)
            tangentPairs.append(model.tangentPairs)

        if not quiet:
            print("Contacts: %d, vel_iter: %d, pos_iter: %d" %
                  (profile.contactsSolved, profile.velocityIterations, profile.positionIterations))


    # Print results
    if not quiet:
        if iterations:
            print("\nVelocity:")
            print("Total   = %d"   % np.sum(totalVelocityIterations))
            print("Average = %.2f" % np.mean(totalVelocityIterations))
            print("Median  = %d"   % np.median(totalVelocityIterations))
            print("Std     = %.2f" % np.std(totalVelocityIterations))

            print("\nPosition:")
            print("Total   = %d"   % np.sum(totalPositionIterations))
            print("Average = %.2f" % np.mean(totalPositionIterations))
            print("Median  = %d"   % np.median(totalPositionIterations))
            print("Std     = %.2f" % np.std(totalPositionIterations))


    # Store results
    result["totalStepTimes"] = totalStepTimes
    result["contactsSolved"] = contactsSolved

    if iterations:
        result["totalVelocityIterations"] = totalVelocityIterations
        result["totalPositionIterations"] = totalPositionIterations

    if convergenceRates:
        result["velocityLambdaTwoNorms"] = velocityLambdaTwoNorms
        result["velocityLambdaInfNorms"] = velocityLambdaInfNorms
        result["positionLambdas"] = positionLambdas

    if lambdaErrors:
        result["normalPairs"] = normalPairs
        result["tangentPairs"] = tangentPairs

    # Return results
    return result
