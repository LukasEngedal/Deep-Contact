import numpy as np

# Takes as input a list of result dicts from simulating one or more worlds,
# and processes them into a dict of lists ready for plotting
def process_results(
        results,
        maxVelocityIterations,
        maxPositionIterations,
        iterationCounters = False,
        velocityConvergenceRates = False,
        positionConvergenceRates = False,
        velocityNorm = "velocityLambdaInfNorms",   # either velocityLambdaTwoNorms or velocityLambdaInfNorms
        lambdaErrors = False,
        lambdaErrorMethod = "MAE",   # Either MAE or
):
    output = {}

    if iterationCounters or lambdaErrors:
        output["contactsSolved"] = np.mean([res["contactsSolved"] for res in results],
                                           axis=0, dtype=np.float64)

        # ----- Iteration Counters -----
    if iterationCounters:
        output["stepTimes"] = np.mean([res["totalStepTimes"] for res in results],
                                      axis=0, dtype=np.float64)

        output["totalVelocityIterations"] = np.mean([res["totalVelocityIterations"] for res in results],
                                                    axis=0, dtype=np.float64)

        output["totalPositionIterations"] = np.mean([res["totalPositionIterations"] for res in results],
                                                    axis=0, dtype=np.float64)


    # ----- Velocity Convergence Rates -----
    if velocityConvergenceRates:
        if velocityNorm == "inf":
            norms = "velocityLambdaInfNorms"
        elif velocityNorm == "two":
            norms = "velocityLambdaTwoNorms"
        else:
            raise "Unknown norm, choose 'inf' or 'two'"

        # We filter out the convergence rates for steps with no contacts
        velocityLambdaLists = [list(filter(lambda l: l[-1] != 0, res[norms])) for res in results]

        # Pad convergence rates to be same length, which is the max, by repeating the last element
        paddedVelocityLambdaLists = [[l + [l[-1]]*(maxVelocityIterations-len(l)) for l in lambdas]
                                         for lambdas in velocityLambdaLists]

        # We transform the data into an array
        velocityLambdaArray = np.concatenate([np.array(l) for l in paddedVelocityLambdaLists])
        output["velocityLambdas"] = velocityLambdaArray


        # We determine the number of steps still iterating for each iteration
        velocityIteratorCountLists = [[np.sum([len(l) >= i for l in lambdas])
                                       for i in range(maxVelocityIterations)]
                                      for lambdas in velocityLambdaLists]

        # Pad iterator counts to be same length, by adding zeros
        paddedVelocityIteratorCountLists = [l + [0]*(maxVelocityIterations-len(l))
                                            for l in velocityIteratorCountLists]

        # We take the mean of the iterator count lists
        output["velocityIteratorCounts"] = np.mean(paddedVelocityIteratorCountLists, axis=0, dtype=np.float64)


    # ----- Position Convergence Rates -----
    if positionConvergenceRates:
        # We filter out the convergence rates for steps with no contacts
        positionLambdaLists = [list(filter(lambda l: l[-1] != 0, res["positionLambdas"])) for res in results]

        # Pad convergence rates to be same length, which is the max, by repeating the last element
        paddedPositionLambdaLists = [[l + [l[-1]]*(maxPositionIterations-len(l)) for l in lambdas]
                                     for lambdas in positionLambdaLists]

        # We transform the data into an array
        positionLambdaArray = np.concatenate([np.array(l) for l in paddedPositionLambdaLists])
        output["positionLambdas"] = positionLambdaArray


        # We determine the number of steps still iterating for each iteration
        positionIteratorCountLists = [[np.sum([len(l) >= i for l in lambdas])
                                       for i in range(maxPositionIterations)]
                                      for lambdas in positionLambdaLists]

        # Pad iterator counts to be same length, by adding zeros
        paddedPositionIteratorCountLists = [l + [0]*(maxPositionIterations-len(l))
                                            for l in positionIteratorCountLists]

        # We take the mean of the iterator count lists
        output["positionIteratorCounts"] = np.mean(paddedPositionIteratorCountLists, axis = 0, dtype=np.float64)


    # ----- Prediction Errors -----
    if lambdaErrors:
        normalPairsListLists = [res["normalPairs"] for res in results]
        tangentPairsListLists = [res["tangentPairs"] for res in results]

        if lambdaErrorMethod == "MSE":
            normalMSELists = [[np.mean([(pair[0] - pair[1])**2 for pair in step]) for step in world]
                              for world in normalPairsListLists]
            tangentMSELists = [[np.mean([(pair[0] - pair[1])**2 for pair in step]) for step in world]
                               for world in tangentPairsListLists]

            normalErrors = np.nanmean(normalMSELists, axis=0)
            tangentErrors = np.nanmean(tangentMSELists, axis=0)
        else:
            raise "Unknown error method, choose MSE or "

        output["normalErrors"] = normalErrors
        output["tangentErrors"] = tangentErrors


    return output
