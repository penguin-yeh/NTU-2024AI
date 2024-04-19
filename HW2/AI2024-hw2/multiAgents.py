# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        curPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        curFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # ghost appear
        for i in newGhostPositions:
            if i == newPos:
                return -99999
            elif util.manhattanDistance(i, newPos) <= 1:
                return -99999
            elif curFood[newPos[0]][newPos[1]] == True:
                return 99999
            
        # normal situation
        min = 99999
        for f in curFood.asList():
            # print(f)
            closet_dis = util.manhattanDistance(newPos, f)
            if closet_dis < min:
                min = closet_dis
                
        return 1000 - min
        
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
    
        # pacman first
        first_action = gameState.getLegalActions(0)
        max = -9999
        for action in first_action:
            # since below is min(ghost)
            tmp = self.min_function(gameState.generateSuccessor(0, action), 0, 1)
            if tmp > max:
                max = tmp
                max_action = action
        return max_action
            
    def max_function(self, gameState: GameState, curDepth):
        if gameState.isWin() or gameState.isLose() or self.depth == curDepth:
            return self.evaluationFunction(gameState)
        
        # below is always ghost => max(min())
        v = -99999
        for action in gameState.getLegalActions(0):
            v = max(v, self.min_function(gameState.generateSuccessor(0, action), curDepth, 1))
            
        return v
            
    def min_function(self, gameState: GameState, curDepth, curGhost):
        if gameState.isWin() or gameState.isLose() or self.depth == curDepth:
            return self.evaluationFunction(gameState)
        
        ghost_count = gameState.getNumAgents() - 1
        
        # below is also ghost => min(min())
        if curGhost < ghost_count:
            v = 99999
            for action in gameState.getLegalActions(curGhost):
                v = min(v, self.min_function(gameState.generateSuccessor(curGhost, action), curDepth, curGhost + 1))
        
        # last ghost => below is new depth, pacman => min(max())
        else:
            v = 99999
            for action in gameState.getLegalActions(curGhost):
                v = min(v, self.max_function(gameState.generateSuccessor(curGhost, action), curDepth + 1))
            
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
         # pacman first
        first_action = gameState.getLegalActions(0)
        max_ = -9999
        alpha = -9999
        beta = 9999
        for action in first_action:
            # since below is ghost => min()
            tmp = self.min_function(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if tmp > max_:
                max_ = tmp
                max_action = action
                alpha = max(alpha, tmp)
        return max_action
            
    def max_function(self, gameState: GameState, curDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or self.depth == curDepth:
            return self.evaluationFunction(gameState)
        
        # below is always ghost => max(min())
        v = -99999
        for action in gameState.getLegalActions(0):
            v = max(v, self.min_function(gameState.generateSuccessor(0, action), curDepth, 1, alpha, beta))
            # upper ghost's beta => want min
            if v > beta:
                return v
            # self alpha
            alpha = max(alpha, v)            
        return v
            
    def min_function(self, gameState: GameState, curDepth, curGhost, alpha, beta):
        if gameState.isWin() or gameState.isLose() or self.depth == curDepth:
            return self.evaluationFunction(gameState)
        
        ghost_count = gameState.getNumAgents() - 1
        
        # below is also ghost => min(min())
        if curGhost < ghost_count:
            v = 99999
            for action in gameState.getLegalActions(curGhost):
                v = min(v, self.min_function(gameState.generateSuccessor(curGhost, action), curDepth, curGhost + 1, alpha, beta))
                # upper pacman's alpha => want max
                if v < alpha:
                    return v
                # self beta
                beta = min(beta, v)
        # last ghost => below is new depth, pacman => min(max())
        else:
            v = 99999
            for action in gameState.getLegalActions(curGhost):
                v = min(v, self.max_function(gameState.generateSuccessor(curGhost, action), curDepth + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
