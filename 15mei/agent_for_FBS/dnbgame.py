import pyspiel
import numpy as np
import networkx as nx

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.game = pyspiel.load_game(f"dots_and_boxes(num_rows={rows},num_cols={cols},utility_margin=true)")
        self.board = self.getInitBoard()

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return self.game.new_initial_state()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.rows, self.cols

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.game.num_distinct_actions()

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # print("action: ", action)
        # print(self.board)
        board.apply_action(action)
        return board, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return [1 if action in board.legal_actions() else 0 for action in range(self.getActionSize())]

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        if board.is_terminal():
            if board.rewards()[0] > 0:
                return 1
            elif board.rewards()[0] < 0:
                return -1
            else:
                return 1e-4
        else:
            return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # dnb is always canonical
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass # we can add the symmetries from part 3 however gnns are invariant to symmetries so we also take it out of mcts

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.dbn_string()

    def arrayRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardArray: a quick conversion of board to a numpy array format.
                         Required by conversions to graphs.
        """

        # Get board as string because it easily contains the necessary information
        string = self.stringRepresentation(board)

        nb_horizontal_edges = (self.rows + 1) * (self.cols)

        # split string into horizontal and vertical edges
        horizontal_edges = string[:nb_horizontal_edges]
        vertical_edges = string[nb_horizontal_edges:]

        # convert to numpy array
        horizontal_edges = np.array([int(c) for c in horizontal_edges])
        vertical_edges = np.array([int(c) for c in vertical_edges])

        # return as array
        return [horizontal_edges, vertical_edges]

    def toGraph(self, board, dummy=True, self_loops=False):
        """
        Takes in a board and returns a graph representation of the board.
        The dots are the nodes and the edges are the edges, just like in the board.
        Nodes do not have any features, so we give them a dummy feature.
        If self loops is True, then the graph also connects edges to themselves.
        If dummy is True, then the graph also connects all edges to a dummy node. This is to allow information to spread more easily.
        This representation makes it difficult to predict a policy since we can't easily do edge classification.
        """
        # Get nb of nodes
        nodes = (self.rows + 1) * (self.cols + 1)

        # Create graph
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(range(nodes))

        edges = self.arrayRepresentation(board)

        # Add edges
        for i in range(self.cols):
            if edges[0][i]:
                offset = i // self.cols
                G.add_edge(i+offset, i+1+offset)
        for i in range(len(edges[1])):
            if edges[1][i]:
                G.add_edge(i, i+self.cols+1)

        # Add dummy node
        if dummy:
            G.add_node(nodes)
            for i in range(nodes):
                G.add_edge(i, nodes)
        
        # Add self loops
        if self_loops:
            G.add_edges_from([(i,i) for i in range(nodes)])
        
        features = np.zeros((nodes+1, 1))
        return G, features

    def toDualGraph(self, board, dummy=True, self_loops=False):
        """
        Takes in a board and returns a graph representation of the board by representing the squares as the nodes.
        Consider the dual representation of the coins and strings game.
        Nodes have a single feature, which is wether or not they are captured.
        If self loops is True, then the graph also connects edges to themselves.
        If dummy is True, then the graph also connects all edges to a dummy node. This is to allow information to spread more easily.
        Neither the wall node nor the dummy node can be captured.
        This representation reduces the number of nodes, but it has the same problems as the normal graph representation.
        It has the advantage that there is less nodes and also that we have direct information about the squares.
        """
        # Nodes is the number of squares + add wall nodes around to help with construction
        # All wall nodes will be collapsed into a single node afterwards
        nodes = (self.rows + 2) * (self.cols + 2)

        # Create graph
        G = nx.MultiGraph()

        # Add nodes
        G.add_nodes_from(range(nodes))

        edges = self.arrayRepresentation(board)

        # Add edges
        for i in range(len(edges[0])):
            offset = (i // (self.cols))*2 + 1
            if not edges[0][i]:
                G.add_edge(i+offset, i+self.cols+2+offset)
        for i in range(len(edges[1])):
            offset = (i // (self.cols+1)) + self.cols+2
            if not edges[1][i]:
                G.add_edge(i+offset, i+1+offset)

        # Collapse wall nodes
        # Collapse top
        for i in range(1, self.cols+2):
            G = nx.contracted_nodes(G, 0, i)

        # Collapse bottom
        for i in range(self.cols+2):
            G = nx.contracted_nodes(G, 0, (self.rows+2)*(self.rows+1)+i)
        
        # Collapse sides
        for i in range(self.rows):
            G = nx.contracted_nodes(G, 0, (self.cols+2)*(i+1))
            G = nx.contracted_nodes(G, 0, (self.cols+2)*(i+1)+self.cols+1)

        # Add dummy node
        if dummy:
            G.add_node(nodes)
            for i in G.nodes():
                G.add_edge(i, nodes)
        
        # Add self loops
        if self_loops:
            G.add_edges_from([(i,i) for i in G.nodes()])

        # features
        features = np.zeros((nodes, 1))
        features[0] = 1
        if dummy:
            features[-1] = 1
        
        squares = self.game.getSquares(edges)
        for i in range(len(squares)):
            if squares[i]:
                features[i+1] = 1
        return G, features

    def toLineGraph(self, board, dummy=True, self_loops=False, connect_opposites=False):
        """
        Takes in a board and returns a graph representation of the board by representing the edges as the nodes.
        This is so we can directly do node classification on the edges.
        The graph connects edges that share a dot (or vertex).
        Nodes have a single feature, which is wether or not they are drawn.
        If connect opposites is True, then the graph also connects opposite edges. This means edges that form a box also form a clique.
        If self loops is True, then the graph also connects edges to themselves.
        If dummy is True, then the graph also connects all edges to a dummy node. This is to allow information to spread more easily.
        The dummy node has a special feature value of -1.
        This representation makes it easy to predict a policy since we can easily do edge classification. However, it may be more costly as there are more nodes and edges.
        """
        # Get original graph from fully connected graph
        edges = [np.ones(self.rows), np.ones(self.cols)]

        # get fully connected graph
        G = self.toGraph(self.game.new_initial_state(f"{'1'*len(edges[0])+'1'*len(edges[1])}"), dummy=False, self_loops=False)[0]

        # Create line graph
        L = nx.line_graph(G)

        # for n in L.nodes:
        #     print(n)

        # split into horizontal and vertical edges
        horizontal_edges = [e for e in L.nodes(data=True) if e[0][1] - e[0][0] == 1]
        horizontal_edges = sorted(horizontal_edges, key=lambda x: x[0][0])
        vertical_edges = [e for e in L.nodes(data=True) if e[0][1] - e[0][0] == self.cols + 1]
        vertical_edges = sorted(vertical_edges, key=lambda x: x[0][0])

        # concatenate horizontal and vertical edges
        rearranged_nodes = horizontal_edges + vertical_edges

        H = nx.Graph()
        H.add_nodes_from(rearranged_nodes)
        H.add_edges_from(L.edges(data=True))


        if connect_opposites:
            for n in H.nodes:
                # get neighbors
                neighbors = list(H.neighbors(n))
                for m in H.nodes:
                    others = list(H.neighbors(m))
                    if n != m and len(set(neighbors).intersection(set(others))) == 2:
                        H.add_edge(n, m)

        # Add dummy node
        if dummy:
            H.add_node(len(H.nodes))
            for i in H.nodes:
                H.add_edge(i, len(H.nodes)-1)
        
        # Add self loops
        if self_loops:
            H.add_edges_from([(i,i) for i in range(len(H.nodes))])

        # Add features
        features = np.zeros((len(H.nodes), 1))

        # concatenate edges
        edges = self.arrayRepresentation(board)
        edges = np.concatenate((edges[0], edges[1]))
        for i in range(len(edges)):
            features[i] = edges[i]
        
        # Add dummy feature
        if dummy:
            features[-1] = -1

        return H, features
    
    def get_squares(self, edges):
        """
        The squares are numbered from left to right, top to bottom.
        The list has dimension size[0] * size[1]
        """
        # print("get_squares")
        squares = np.zeros((self.rows * self.cols), dtype=bool)
        for i in range(squares.shape[0]):
            r, c = i // self.cols, i % self.cols
            squares[i] = bool(edges[0][c + r*self.cols] and edges[0][c + (r+1)*self.cols] and edges[1][c + r*(self.cols+1)] and edges[1][c + 1 + r*(self.cols+1)])
                # squares[i] = 1
        return squares