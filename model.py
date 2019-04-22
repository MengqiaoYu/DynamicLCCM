import IOHMM
import logging
import pandas as pd
from os import listdir
import os
import numpy as np
import datetime
import json
import sys
sys.path.insert(0, '/Users/MengqiaoYu/Desktop/WholeTraveler')
from data_process import data_process


logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

"""Section 1: Load data"""
data_process.save_model_data(filepath='/Users/MengqiaoYu/Desktop/WholeTraveler/Data/young_data.csv', num_year=2000-1988)
data_dir = '/Users/MengqiaoYu/Desktop/WholeTraveler/Data/model/'
logger.info("Load the formatted files in dir: %s into model." %data_dir)
data_model = []
for f in listdir(data_dir):
    if f.startswith('.'):
        continue
    data_ind = pd.read_csv(data_dir + f)
    data_model.append(data_ind)

"""Section 2: Train the model"""
logger.info("Start training the model.")
### Set model parameters
model_policy = IOHMM.UnSupervisedIOHMM(num_states=3,
                                       max_EM_iter=200,
                                       EM_tol=1e-5)

# emission models are number of cars and mode choice
model_policy.set_models(
    model_emissions = [IOHMM.OLS(fit_intercept=True, est_stderr=True),
                       IOHMM.DiscreteMNL(solver='lbfgs', fit_intercept=True)],
    model_transition=IOHMM.CrossEntropyMNL(solver='lbfgs', fit_intercept=True),
    model_initial=IOHMM.CrossEntropyMNL(solver='lbfgs', fit_intercept=True))

model_policy.set_inputs(covariates_initial = [],
                covariates_transition = ['child', 'move', 'age'], # 'gas_price', 'ue_rate_scaled'
                covariates_emissions = [[],[],[],[]])

model_policy.set_outputs([['numcars'], ['used_own']])

model_policy.set_data(data_model)

model_policy.train()

### print the training results
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
logger.info("Here is the initial matrix:")
PI = np.exp(model_policy.model_initial.predict_log_proba(np.array([[]]))).reshape(model_policy.num_states,)
logger.info(np.exp(model_policy.model_initial.predict_log_proba(np.array([[]]))))

logger.info("Here is the transition matrix:")
A = {}
for i in range(model_policy.num_states):
    A[i] = model_policy.model_transition[i].coef
    logger.info(model_policy.model_transition[i].coef)

logger.info("Here is the car ownership model:")
for i in range(model_policy.num_states):
    logger.info(model_policy.model_emissions[i][0].coef)

logger.info("Here is the mode choice model (prob):")
for i in range(model_policy.num_states):
    logger.info(np.exp(model_policy.model_emissions[i][1].predict_log_proba(np.array([[]]))))

# Save the model results
model_results = {"Initial matrix": PI,
                 "Transition matrix": [model_policy.model_transition[i].coef
                                       for i in range(model_policy.num_states)],
                 "car ownership model": [model_policy.model_emissions[i][0].coef
                                        for i in range(model_policy.num_states)],
                 "Mode choice model": [model_policy.model_emissions[i][1].predict_log_proba(np.array([[]]))
                                       for i in range(model_policy.num_states)]}

result_dir = "/Users/MengqiaoYu/Desktop/WholeTraveler/Data/"
result_filename = "result_" + datetime.datetime.now().strftime('%m%d_%H%M') + ".json"
with open(os.path.join(result_dir, result_filename), 'w') as fp:
    json.dump(model_results, fp)