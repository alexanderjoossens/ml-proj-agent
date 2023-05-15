import logging

# import coloredlogs

from coach import Coach
from utils import dotdict
# from neuralnet import NeuralNet as nn
from NNet import NNetWrapper as nn
from dnbgame import Game

log = logging.getLogger(__name__)

X_SIZE = 7
Y_SIZE = 7

# coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# args = dotdict({
#     'numIters': 1000,
#     'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
#     'tempThreshold': 15,        #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOf'
#     'Queue': 200000,            # Number of game examples to train the neural networks.
#     'numMCTSSims': 300,          # Number of games moves for MCTS to simulate.
#     'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
#     'cpuct': 1,

#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,

# })

args = dotdict({
    'numIters': 100,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOf'
    'Queue': 50,            # Number of game examples to train the neural networks.
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': f'./temp_fixed_{X_SIZE}x{Y_SIZE}/',
    'load_model': True,
    'load_folder_file': (f'./temp_fixed_{X_SIZE}x{Y_SIZE}/','temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

args['numIters'] = 5
args['numEps'] = 3
args['numMCTSSims'] = 10
args['arenaCompare'] = 3
args['Queue'] = 5


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(X_SIZE, Y_SIZE)
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()