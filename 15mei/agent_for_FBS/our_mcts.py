import logging
import math
import copy
import time
import random
import numpy as np
import pyspiel

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS(pyspiel.Bot):
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        pyspiel.Bot.__init__(self)
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        originalBoard = copy.deepcopy(canonicalBoard)

        for i in range(self.args.numMCTSSims):
            # make a copy of the board
            canonicalBoardCopy = copy.deepcopy(originalBoard)
            # print("canonicalBoardCopy: \n", canonicalBoardCopy)
            self.search(canonicalBoardCopy)

        canonicalBoard = copy.deepcopy(originalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]


        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            # print("terminal node")
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            
            self.Ps[s], v = self.nnet.predict(canonicalBoard, self.game)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            
            # pad valids with zeros to match the size of the policy vector of size 480
            # valids = np.pad(valids, (0, 480 - len(valids)), 'constant', constant_values=(0, 0))
            # valids = np.pad(valids, (0, 112 - len(valids)), 'constant', constant_values=(0, 0))

            # print(s)
            # print("valids: ", valids)
            # print("self.Ps[s]: ", self.Ps[s])

            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            # print("self.Ps[s] after masking: ", self.Ps[s])
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # print("leaf node")
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                
                # print("u: ", u)
                # print("cur_best: ", cur_best)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # print(f"{s=}")
        # print(f"{a=}")
        # print(canonicalBoard)
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        # print("s: ", s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
    
    def restart_at(self, state):
        pass


    def step_random(self, state):
        # return self.step_with_policy(state)[1]
        legal_actions = state.legal_actions()
        rand_idx = random.randint(0, len(legal_actions) - 1)
        action = legal_actions[rand_idx]
        return action
    
    def step(self, state):
        """
        return an action based on the highest probability action from getActionProb. The action must be a legal_action
        """
        return np.argmax(self.getActionProb(state))

