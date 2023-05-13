#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import random
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from our_mcts import MCTS
# import arena
from utils import dotdict
from NNet import NNetWrapper
import os
from dnbgame import Game
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.mcts import MCTSBot

logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')


args = dotdict({
    'numIters': 100,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOf'
    'Queue': 50,            # Number of game examples to train the neural networks.
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 5,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def get_agent_for_tournament_test(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    # my_player = Agent(player_id)
    game_azg = Game(7, 7)
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
    game_os = pyspiel.load_game(dotsandboxes_game_string)
    uct_c = 2 # apparently okay
    max_simulations = 10
    numMCTSSims = 5
    n1 = NNetWrapper(game_azg)
    n1.load_checkpoint(os.path.join('./', 'temp'), 'temp.pth.tar')

    args1 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
    # my_player = MCTS(game, n1, args1)
    evaluator = az_evaluator.AlphaZeroEvaluator(game_os, n1)
    my_player = MCTSBot(game_os, uct_c, max_simulations, evaluator)

    return my_player


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    # my_player = Agent(player_id)
    game = Game(7, 7)
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
    # game = pyspiel.load_game(dotsandboxes_game_string)
    
    numMCTSSims = 5
    n1 = NNetWrapper(game)
    n1.load_checkpoint(os.path.join('./', 'temp'), 'temp.pth.tar')
    args1 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
    my_player = MCTS(game, n1, args1)

    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id



    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        # Plays random action, change with your best strategy
        legal_actions = state.legal_actions()
        rand_idx = random.randint(0, len(legal_actions) - 1)
        action = legal_actions[rand_idx]
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
