import time
import numpy as np

from ..gen_data.util import load_grid
from .peak import Peak
from .pressure import Pressure

# Disables stupid tensorflow warnings about cpu instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Training data
# Path is relative to the gen_data directory
train_path = "data/grid1515100_025_05/"
train_numbers = range(0, 100)
train_steps = 250

# Training parameters
train_params = {}
train_params["batch_size"] = 40
train_params["num_epochs"] = 2

# Evaluation data
# Path is relative to the gen_data directory
eval_path = train_path
eval_numbers = range(100, 110)
eval_steps = 300

# Decides whether to train
train = True
# Decides whether to evaluate
evaluate = True
# How often to run evaluation - don't pick 0
evalRate = 10


# Create the cnn
cnn = Peak()
#cnn = Pressure()

# Training
if train:
    for i in range(len(train_numbers)):
        n = train_numbers[i]
        print("Training on dataset %d of %d" % (i+1, len(train_numbers)))
        start = time.time()

        # We load the training data
        train_features, train_labels = load_grid(train_path, n)

        # We train the model
        cnn.train(train_features, train_labels, train_params)

        print("Training took: %d s" % (time.time()-start))


        # Evaluation
        if evaluate:
            if (n+1) % evalRate == 0:
                print("Running evaluation")
                start = time.time()

                eval_features = []
                eval_labels = []
                for i in range(len(eval_numbers)):
                    n = eval_numbers[i]

                    # We load the evaluation data
                    eval_f, eval_l = load_grid(eval_path, n)

                    if eval_features == []:
                        eval_features = eval_f
                        eval_labels = eval_l
                    else:
                        eval_features = np.concatenate((eval_features, eval_f), axis=0)
                        eval_labels = np.concatenate((eval_labels, eval_l), axis=0)

                # Evaluate the model and print results
                eval_results = cnn.evaluate(eval_features, eval_labels)

                print("Evaluation took: %d s" % (time.time()-start))
