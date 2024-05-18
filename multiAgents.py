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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()

        ghostDistance = 999999
        foodDistance = 999999
        # capsuleDistance = 999999

        for ghost in successorGameState.getGhostPositions():
            ghostDistance = min(ghostDistance, manhattanDistance(newPos, ghost)) 

        for food in foodList:
            foodDistance = min(foodDistance, manhattanDistance(newPos, food)) 

        if successorGameState.isWin():
            return float('inf')
        if ghostDistance < 2:
            return float('-inf')
        if action == "STOP":
            return float('-inf')

        return successorGameState.getScore() + 10/foodDistance

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        maxVal = float("-inf")
        stopAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextVal = self.minimaxHelper(nextState, 0, 1)
            if nextVal > maxVal:
                maxVal = nextVal
                stopAction = action
        return stopAction
       
       # util.raiseNotDefined()
    
    """
    maximizer:
        Assumes we are the maximizer (Pacman) and 
        checks all legal actions with their respective
        evaluation score and returns maximum.
    """    
    def maximizer(self, gameState, depth): 
        maxVal = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.minimaxHelper(successor, depth, 1)
            maxVal = max(maxVal, val)
        return maxVal
    
    """
    minimizer:
        Assumes we are the minimizer (Ghosts) and 
        checks all legal actions with their respective
        evaluation score and returns minimum.
    """   
    def minimizer(self, gameState, depth, agent): 
        minVal = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
                val = self.minimaxHelper(successor, depth + 1, 0)
                minVal = min(minVal, val)
            else:
                val = self.minimaxHelper(successor, depth, agent + 1)
                minVal = min(minVal, val)
        return minVal
    
    """
    minimaxHelper:
        Checks the state of the game and returns
        evaluation score if a condition is meet.
        If not then go further.
    """   
    def minimaxHelper(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        elif agent == 0:
            return self.maximizer(gameState, depth)
        else:
            return self.minimizer(gameState, depth, agent)        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        maxVal = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        stopAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextVal = self.alphabetaHelper(nextState, 0, 1, alpha, beta)
            if nextVal > maxVal:
                maxVal = nextVal
                stopAction = action
            alpha = max(maxVal, alpha)
        return stopAction
    
        # util.raiseNotDefined()
    
    """
    maximizer:
        Assumes we are the maximizer (Pacman) and 
        checks all legal actions with their respective
        evaluation score that accounts alpha and beta 
        values and returns maximum.
    """ 
    def maximizer(self, gameState, depth, alpha, beta):
        maxVal = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.alphabetaHelper(successor, depth, 1, alpha, beta)
            maxVal = max(maxVal, val)
            if maxVal > beta:
                return maxVal
            alpha = max(maxVal, alpha)
        return maxVal
    
    """
    minimizer:
        Assumes we are the minimizer (Ghosts) and 
        checks all legal actions with their respective
        evaluation score that accounts alpha and beta
        values and returns minimum.
    """  
    def minimizer(self, gameState, depth, agent, alpha, beta):
        minVal = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
                val = self.alphabetaHelper(successor, depth + 1, 0, alpha, beta)
                minVal = min(minVal, val)
            else:
                val = self.alphabetaHelper(successor, depth, agent + 1, alpha, beta)
                minVal = min(minVal, val)
                
            if minVal < alpha:
                return minVal
            
            beta = min(minVal, beta)
        return minVal
    
    """
    alphabetaHelper:
        Checks the state of the game and returns
        evaluation score if a condition is meet.
        If not then go further.
    """   
    def alphabetaHelper(self, gameState, depth, agent, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agent == 0:
            return self.maximizer(gameState, depth, alpha, beta)
        else:
            return self.minimizer(gameState, depth, agent, alpha, beta)
        
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        maxVal = float("-inf")
        stopAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextVal = self.expectimaxHelper(nextState, 0, 1)
            if nextVal > maxVal:
                maxVal = nextVal
                stopAction = action
        return stopAction 

    """
    maximizer:
        Assumes we are the maximizer (Pacman) and 
        checks all legal actions with their respective
        evaluation and returns maximum.
    """ 
    def maximizer(self, gameState, depth):
        maxVal = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.expectimaxHelper(successor, depth, 1)
            maxVal = max(maxVal, val)
        return maxVal

    """
    expectVal:
        Checks all legal actions of the respective agent
        and search further with the associated path. 
        return average value.
    """
    def expectVal(self, gameState, depth, agent):
        expVal = 0.0
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
                val = self.expectimaxHelper(successor, depth + 1, 0)
                expVal += val
            else:
                val = self.expectimaxHelper(successor, depth, agent + 1)
                expVal += val
        return expVal
    
    """
    expectimaxHelper:
        Checks the state of the game and returns
        evaluation score if a condition is meet.
        If not then search further.
    """ 
    def expectimaxHelper(self, gameState, depth, agent):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agent == 0:
            return self.maximizer(gameState, depth)
        else:
            return self.expectVal(gameState, depth, agent)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
