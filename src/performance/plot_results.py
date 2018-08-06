import os
import numpy as np
import matplotlib.pyplot as plt

from .util import smooth
from .process_results import process_results


# ----- Parameters -----
# List of result file paths, and a name and a color for each
models = [
#    ("./results/none.npz", "None", "blue"),
#    ("./results/builtin.npz", "Builtin", "orange"),
#    ("./results/bad.npz", "Bad", "black"),
#    ("./results/random.npz", "Random", "yellow"),
#    ("./results/copy.npz", "Copy", "green"),
#    ("./results/copy0.npz", "Copy0", "orange"),
#    ("./results/copy2.npz", "Copy2", "red"),
#    ("./results/grid.npz", "Grid", "brown"),
    ("./results/peak.npz", "Peak", "red"),
#    ("./results/pressure.npz", "Pressure", "red"),
]

# Iteration limits used when generating data
velocityIterations = 1000
positionIterations = 250
# Iteration thresholds used when generating data
velocityThreshold = 1*10**-4
positionThreshold = 1*10**-4
# Number of steps used when generating data
steps = 250

# Iteration counter plots
plotIterationCounters = True
# Iteration plots Smoothing factor - set to 1 to turn off smoothing
smoothingFactor = 1


# Velocity convergence rate plots
plotVelocityConvergenceRates = True
# Chooses which norm to use, currently two options, "two" or "inf"
velocityNorm = "inf"
# Cutoff for convergence rate plot
velocityCutoff = 500
# Chooses whether to combine convergence rates using the percentile or the mean function
velocityCombineMethod = "percentile"
#velocityCombineMethod = "mean"
# Percentile to plot if set
# 0 for min, 25 for 1st quantile, 50 for median, 75 for 3rd quantile, 100 for max
velocityPercentile = 50
# Add 1st and 3rd quantiles to plot as dotted lines, if percentile method is used
velocityAddQuantiles = False


# Position convergence rate plots
plotPositionConvergenceRates = False
# Cutoff for convergence rate plot
positionCutoff = 100
# Chooses whether to plot using the percentile or the mean function
positionCombineMethod = "percentile"
#positionCombineMethod = "mean"
# Percentile to plot if set
positionPercentile = 50
# Add 1st and 3rd quantiles to plot as dotted lines, if percentile method is used
# Might get visually messy if there are a lot of models
positionAddQuantiles = False


# Plots errors in lambda predictions
plotLambdaErrors = True
# Method used to determine errors - only MSE for now
lambdaErrorMethod = "MSE"
# Error plots smoothing factor - set to 1 to turn off smoothing
errorSmoothingFactor = 1



# ----- Load Results -----
print("Loading results")

nModels = len(models)
results = []
for i in range(nModels):
    file_path = models[i][0]

    # If path is not absolute, fix it
    if not os.path.isabs(file_path):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(file_dir, file_path)

    npzfile = np.load(file_path)

    results.append(npzfile["results"])


# ----- Process Results -----
print("Processing results")

processed_results = []
for i in range(nModels):
    processed_res = process_results(
        results[i],
        velocityIterations,
        positionIterations,
        iterationCounters = plotIterationCounters,
        velocityConvergenceRates = plotVelocityConvergenceRates,
        positionConvergenceRates = plotPositionConvergenceRates,
        velocityNorm = velocityNorm,
        lambdaErrors = plotLambdaErrors,
        lambdaErrorMethod = lambdaErrorMethod,
    )
    processed_results.append(processed_res)


if plotVelocityConvergenceRates:
    # Lambdas
    if velocityCombineMethod == "percentile":
        velocityConvData = [np.percentile(res["velocityLambdas"], velocityPercentile, axis=0)
                            for res in processed_results]
        if velocityAddQuantiles:
            velocityQ1 = [np.percentile(res["velocityLambdas"], 25, axis=0)
                          for res in processed_results]
            velocityQ3 = [np.percentile(res["velocityLambdas"], 75, axis=0)
                          for res in processed_results]

    elif velocityCombineMethod == "mean":
        velocityConvData = [np.mean(res["velocityLambdas"], axis=0, dtype=np.float64)
                            for res in processed_results]
    else:
        raise "Unkown combine method; choose 'percentile' or 'mean'"

    # Iterator counts
    velocityIteratorData = [res["velocityIteratorCounts"] for res in processed_results]


if plotPositionConvergenceRates:
    # Lambdas
    if positionCombineMethod == "percentile":
        positionConvData = [np.percentile(res["positionLambdas"], positionPercentile, axis=0)
                            for res in processed_results]
        if positionAddQuantiles:
            positionQ1 = [np.percentile(res["positionLambdas"], 25, axis=0)
                          for res in processed_results]
            positionQ3 = [np.percentile(res["positionLambdas"], 75, axis=0)
                          for res in processed_results]
    elif positionCombineMethod == "mean":
        positionConvData = [np.mean(res["positionLambdas"], axis=0, dtype=np.float64)
                            for res in processed_results]
    else:
        raise "Unkown combine method; choose 'percentile' or 'mean'"

    # Iterator counts
    positionIteratorData = [res["positionIteratorCounts"] for res in processed_results]


if plotLambdaErrors:
    normalErrors = [smooth(pro_res["normalErrors"], errorSmoothingFactor) for pro_res in processed_results]
    tangentErrors = [smooth(pro_res["tangentErrors"], errorSmoothingFactor) for pro_res in processed_results]

    normalMin = max(10**-5, np.nanmin([np.nanmin(l) for l in normalErrors]))
    normalMax = max([np.nanmax(l) for l in normalErrors])

    tangentMin = max(10**-5, np.nanmin([np.nanmin(l) for l in tangentErrors]))
    tangentMax = max([np.nanmax(l) for l in tangentErrors])



# ----- Plot stuff -----
names  = [t[1] for t in models]
colors = [t[2] for t in models]


# --- Iteration plots ---
if plotIterationCounters:
    fig = plt.figure("Iterations")
    fig.suptitle("Iteration Counters")

    # Times
    ax1 = fig.add_subplot(221)
    for i in range(nModels):
        ax1.plot(
            smooth(processed_results[i]["stepTimes"],  smoothingFactor),
            ls="solid", c=colors[i], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Step time")
    ax1.set_title("Time taken for each step")

    # Contacts
    ax1 = fig.add_subplot(222)
    for i in range(nModels):
        ax1.plot(
            processed_results[i]["contactsSolved"],
            ls="solid", c=colors[i], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of contacts")
    ax1.set_title("Contact numbers for each step")

    # Velocity iterations
    ax1 = fig.add_subplot(223)
    for i in range(nModels):
        ax1.plot(
            smooth(processed_results[i]["totalVelocityIterations"], smoothingFactor),
            ls="solid", c=colors[i], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.set_title("Velocity iterations numbers")

    # Position iterations
    ax1 = fig.add_subplot(224)
    for i in range(nModels):
        ax1.plot(
            smooth(processed_results[i]["totalPositionIterations"], smoothingFactor),
            ls="solid", c=colors[i], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.set_title("Position iterations numbers")


# --- Convergence rate plots ---
if plotVelocityConvergenceRates:
    fig = plt.figure("Velocity Convergence")
    fig.suptitle("Velocity Convergence Rates")

    velocityYlim = [velocityThreshold/2, 10**1]

    velocityQuantiles = (velocityCombineMethod == "percentile") and velocityAddQuantiles

    # Full lambda convergence rates
    ax1 = fig.add_subplot(221)
    for i in range(nModels):
        ax1.semilogy(velocityConvData[i], ls="solid", c=colors[i], label=names[i])
        if velocityQuantiles:
            ax1.semilogy(velocityQ1[i], ls="dashed", c=colors[i])
            ax1.semilogy(velocityQ3[i], ls="dashed", c=colors[i])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Inf-Norm")
    ax1.set_title("Velocity Lambda Convergence Rate - All iterations, " + str(velocityPercentile) + "%")

    # Counters
    ax1 = fig.add_subplot(222)
    for i in range(nModels):
        ax1.plot(velocityIteratorData[i], ls="solid", c=colors[i], label=names[i])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited lambda inf-norm convergence rates
    ax1 = fig.add_subplot(223)
    for i in range(nModels):
        ax1.semilogy(velocityConvData[i], ls="solid", c=colors[i], label=names[i])
        if velocityQuantiles:
            ax1.semilogy(velocityQ1[i], ls="dashed", c=colors[i])
            ax1.semilogy(velocityQ3[i], ls="dashed", c=colors[i])
    ax1.set_xlim([0, velocityCutoff])
    ax1.set_ylim(velocityYlim)
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Inf-Norm")
    ax1.set_title("Velocity Lambda Convergence Rate - Cutoff, " + str(velocityPercentile) + "%")

    # Limited counters
    ax1 = fig.add_subplot(224)
    for i in range(nModels):
        ax1.plot(velocityIteratorData[i], ls="solid", c=colors[i], label=names[i])
    ax1.set_xlim([0, velocityCutoff])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")


if plotPositionConvergenceRates:
    fig = plt.figure("Position Convergence")
    fig.suptitle("Position Convergence Rates")

    positionYlim = [positionThreshold/2, 10**1]

    positionQuantiles = (positionCombineMethod == "percentile") and positionAddQuantiles

    # Full lambda convergence rates
    ax1 = fig.add_subplot(221)
    for i in range(nModels):
        ax1.semilogy(positionConvData[i], ls="solid", c=colors[i], label=names[i])
        if positionQuantiles:
            ax1.semilogy(positionQ1[i], ls="dashed", c=colors[i])
            ax1.semilogy(positionQ3[i], ls="dashed", c=colors[i])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda")
    ax1.set_title("Position Lambda Convergence Rate - All iterations, " + str(positionPercentile) + "%")

    # Counters
    ax1 = fig.add_subplot(222)
    for i in range(nModels):
        ax1.plot(positionIteratorData[i], ls="solid", c=colors[i], label=names[i])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited lambda convergence rates
    ax1 = fig.add_subplot(223)
    for i in range(nModels):
        ax1.semilogy(positionConvData[i], ls="solid", c=colors[i], label=names[i])
        if positionQuantiles:
            ax1.semilogy(positionQ1[i], ls="dashed", c=colors[i])
            ax1.semilogy(positionQ3[i], ls="dashed", c=colors[i])
    ax1.set_xlim([0, positionCutoff])
    ax1.set_ylim(positionYlim)
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda")
    ax1.set_title("Position Lambda Convergence Rate - Cutoff, " + str(positionPercentile) + "%")

    # Limited counters
    ax1 = fig.add_subplot(224)
    for i in range(nModels):
        ax1.plot(positionIteratorData[i], ls="solid", c=colors[i], label=names[i])
    ax1.set_xlim([0, positionCutoff])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")


# --- Lambda errors plots ---
if plotLambdaErrors:
    fig = plt.figure("Lambda Errors")
    fig.suptitle("Lambda Errors")

    # Normal Errors
    ax1 = fig.add_subplot(221)
    for i in range(nModels):
        ax1.semilogy(normalErrors[i], ls="solid", c=colors[i], label=names[i])
    ax1.set_xlim([0, steps])
    ax1.set_ylim([normalMin, normalMax])
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error")
    ax1.set_title("Normal Impulse Errors")

    # Tangent Errors
    ax1 = fig.add_subplot(222)
    for i in range(nModels):
        ax1.semilogy(tangentErrors[i], ls="solid", c=colors[i], label=names[i])
    ax1.set_xlim([0, steps])
    ax1.set_ylim([tangentMin, tangentMax])
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error")
    ax1.set_title("Tangent Impulse Errors")

    # Contacts
    ax1 = fig.add_subplot(223)
    for i in range(nModels):
        ax1.plot(processed_results[i]["contactsSolved"], ls="solid", c=colors[i], label=names[i])
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of contacts")
    ax1.set_title("Contact numbers for each step")


if plotVelocityConvergenceRates or plotPositionConvergenceRates or plotIterationCounters or plotLambdaErrors:
    plt.show()
