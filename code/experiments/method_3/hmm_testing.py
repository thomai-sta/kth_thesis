from hmmlearn import hmm
import numpy as np
from common_functions import iter_from_X_lengths
from common_functions import random_ints_with_sum
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])

model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))

for i in np.arange(1):
    obs, _ = model.sample(10000)
    model.fit(obs)
    N = 100

    obs, _ = model.sample(N)

    lengths = list(random_ints_with_sum(N))

    for i, j in iter_from_X_lengths(obs, lengths):
        observation = obs[i:j, :]
        logprob = model.score(observation)
        logprob1 = model.score(observation, [observation.shape[0]])
        print(logprob, logprob1)
        print(observation.shape[0], j - i)
