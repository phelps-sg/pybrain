__author__ = "Steve Phelps, sphelps@sphelps.net"

from scipy import array

from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer

import numpy as np

class SoftmaxExplorer(DiscreteExplorer):

    def __init__(self, probability_fn):
        DiscreteExplorer.__init__(self)
        self.state = None
        self.probability_fn = probability_fn

    def activate(self, state, action):
        self.state = state
        return DiscreteExplorer.activate(self, state, action)

    def _forwardImplementation(self, inbuf, outbuf):
        propensities = self.module.getActionValues(self.state)
        probabilities = self.probability_fn(propensities)
        outbuf[:] = np.random.choice(len(probabilities), 1, p=probabilities)