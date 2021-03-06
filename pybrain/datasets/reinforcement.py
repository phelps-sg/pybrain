__author__ = 'Steve Phelps, sphelps@sphelps.net'

from pybrain.datasets.sequential import SequentialDataSet
from pybrain.datasets.dataset import DataSet
from scipy import zeros

import pandas as pd

class ReinforcementListDataSet(object):

    def __init__(self, numStates, numActions):
        self.numStates = numStates
        self.numActions = numActions
        self.clear()

    def addSample(self, agent):
        self.lastSample = [agent.lastobs[0], agent.lastaction[0], agent.lastreward]
        self.dataset.append(self.lastSample + list(agent.module.getActionValues(0)[:]))
        self.advance_time()

    def getSample(self):
        return self.lastSample

    def advance_time(self):
        self.t = self.t + 1

    def clear(self):
        self.cols = ['state', 'action', 'reward']
        for s in range(self.numActions):
            self.cols.append('Q' + str(s))
        self.dataset = []
        self.t = 0

    def df(self):
        return pd.DataFrame(self.dataset, columns=self.cols)


    def newEpisode(self):
        pass
        #TODO


class ReinforcementDataFrameDataSet(object):

    def __init__(self, numStates, numActions):
        self.numStates = numStates
        self.numActions = numActions
        self.clear()

    def addSample(self, agent):
        self.lastSample = [agent.lastobs[0], agent.lastaction[0], agent.lastreward]
        self.df.loc[self.t] = self.lastSample + list(agent.module.getActionValues(0)[:])
        self.advance_time()

    def getSample(self):
        return self.lastSample

    def advance_time(self):
        self.t = self.t + 1

    def clear(self):
        cols = ['state', 'action', 'reward']
        for s in range(self.numActions):
            cols.append('Q' + str(s))
        self.df = pd.DataFrame(columns=cols)
        self.t = 0

    def newEpisode(self):
        pass
        #TODO


class ReinforcementDataSet(SequentialDataSet):
    def __init__(self, statedim, actiondim):
        """ initialize the reinforcement dataset, add the 3 fields state, action and
            reward, and create an index marker. This class is basically a wrapper function
            that renames the fields of SupervisedDataSet into the more common reinforcement
            learning names. Instead of 'episodes' though, we deal with 'sequences' here. """
        DataSet.__init__(self)
        # add 3 fields: input, target, importance
        self.addField('state', statedim)
        self.addField('action', actiondim)
        self.addField('reward', 1)
        # link these 3 fields
        self.linkFields(['state', 'action', 'reward'])
        # reset the index marker
        self.index = 0
        # add field that stores the beginning of a new episode
        self.addField('sequence_index', 1)
        self.append('sequence_index', 0)
        self.currentSeq = 0
        self.statedim = statedim
        self.actiondim = actiondim

        # the input and target dimensions (for compatibility)
        self.indim = self.statedim
        self.outdim = self.actiondim

    # def addSample(self, state, action, reward):
    def addSample(self, agent):
        """ adds a new sample consisting of state, action, reward.

            :key state: the current state of the world
            :key action: the executed action by the agent
            :key reward: the reward received for action in state """
        self.appendLinked(agent.lastobs, agent.lastaction, agent.lastreward)

    def getSumOverSequences(self, field):
        sums = zeros((self.getNumSequences(), self.getDimension(field)))
        for n in range(self.getNumSequences()):
            sums[n, :] = sum(self._getSequenceField(n, field), 0)
        return sums

    def __reduce__(self):
        # FIXME: This does actually not feel right: We have to use the DataSet
        # method here, although we inherit from sequential dataset.
        _, _, state, _, _ = DataSet.__reduce__(self)
        creator = self.__class__
        args = self.statedim, self.actiondim
        return creator, args, state, iter([]), iter({})


