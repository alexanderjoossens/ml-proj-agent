import numpy as np
import sys
import os
from utils import dotdict
from NeuralNet import NeuralNet
import tensorflow as tf

from DotsAndBoxesNNet import DotsAndBoxesNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})


def normalize_score(board):
    p1_score = board[:, 0, -1]
    p2_score = board[:, 1, -1]
    score = p1_score - p2_score

    n = board.shape[-1]-1

    max_score = n ** 2
    min_score = -max_score

    min_normalized, max_normalized = 0, 1
    normalized_score = ((score - max_score) / (min_score - max_score)) * (min_normalized - max_normalized) + max_normalized

    board[:, 0, -1] = normalized_score
    board[:, 1, -1] = 0

def ChangeBoardRepresentation(board, game,exp = False):
        """
        board: np array with board
        """

        horizontal_edges, vertical_edges = game.arrayRepresentation(board)
        input_board = np.concatenate([horizontal_edges, vertical_edges], axis=-1)
        if exp:
            input_board = np.expand_dims(input_board, axis=0)

        return input_board



class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        # input_boards = np.asarray(input_boards)
        # print(input_boards)
        
        updated_input_boards = [ChangeBoardRepresentation(b, self.game, exp=False) for b in input_boards]
        updated_input_boards = np.asarray(updated_input_boards)

        # normalize_score(input_boards)

        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        print(f'target_pis shape: {target_pis.shape}')
        print(f'target_vs shape: {target_vs.shape}')
        # print(f'input_boards shape: {input_boards.shape}')
        self.nnet.model.fit(x=updated_input_boards, y=[target_pis, target_vs], batch_size=args.batch_size, epochs=args.epochs)

    def predict(self, board, game):
        """
        board: np array with board
        # """

        # Create input tensor
        # horizontal_edges, vertical_edges = self.game.arrayRepresentation(board)
        # input_board = np.concatenate([horizontal_edges, vertical_edges], axis=-1)
        # input_board = np.expand_dims(input_board, axis=0)
        input_board = ChangeBoardRepresentation(board, game,exp=True)
        # input_board has shape (1,112)

        # Call predict method of the model
        pi, v = self.nnet.model.predict(input_board, verbose=False)

        return pi[0], v[0]


    def predict_old(self, board):
        """
        board: np array with board
        # """


        # Create input tensor

        horizontal_edges = self.game.arrayRepresentation(board)[0]
        vertical_edges = self.game.arrayRepresentation(board)[1]
        # input_tensor = np.concatenate([horizontal_edges, vertical_edges], axis=0)

        # # Print size and shape of input tensor
        # print("Size of input tensor:", input_tensor.size)
        # print("Shape of input tensor:", input_tensor.shape)

        # # Reshape input tensor
        # reshaped_tensor = np.reshape(input_tensor, (1, horizontal_edges.shape[0], vertical_edges.shape[0]*2))

        # # Check the shape of the reshaped tensor
        # print("Shape of reshaped tensor:", reshaped_tensor.shape)

        # # Check the shape of the reshaped tensor
        # print(reshaped_tensor.shape)

        # Create a TensorFlow constant with the batch size, horizontal edges, and vertical edges
        batch_size = 1
        tensor = tf.constant([batch_size, *horizontal_edges.shape, *vertical_edges.shape], dtype=tf.int64)

        # Print the tensor shape
        print("Tensor shape:", tensor.shape)
        print(tensor)


        # horizontal_edges = self.game.arrayRepresentation(board)[0]
        # vertical_edges = self.game.arrayRepresentation(board)[1]
        # # create an array in the shape (1, len(horizontal_edges), len(vertical_edges))
        # new_array = np.array([horizontal_edges, vertical_edges])
        # print(new_array)
        # new_array = new_array.reshape(1, len(horizontal_edges), len(vertical_edges))
        # print(self.game.arrayRepresentation(board))

        # board = np.copy(board)
        # board = board[np.newaxis, :, :]
        # normalize_score(board)
        # print(self.game.arrayRepresentation(board))
        # print(self.game.arrayRepresentation(board)[0].shape)
        # new_array = np.array(self.game.arrayRepresentation(board)[0], self.game.arrayRepresentation(board)[1])
        # print(new_array.shape)
        
        # input_array = self.game.arrayRepresentation(board)
        # input_tensor = np.array(input_array[0]).reshape((1, 56, 56))
        # print(input_array[0].shape)
        
        # input_tensor = np.array(1, input_array[0],  input_array[1])

        # input_tensor = np.concatenate((input_array[0], input_array[1]), axis=0)
        # print(input_array[0].shape)
        # input_tensor = input_tensor.reshape(1, len(input_array[0]),  len(input_array[1]))

        
        pi, v = self.nnet.model.predict(tensor, verbose=False)

        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath)