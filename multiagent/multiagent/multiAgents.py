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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        # xử lý food
        foodList = newFood.asList()
        
        currentFoodCount = currentGameState.getFood().count()
        newFoodCount = len(foodList)
        if newFoodCount < currentFoodCount:
            score += 100  
        
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # sử dụng reciprocal để food gần = điểm cao
            score += 1.0 / (minFoodDistance + 1) 
        
        # xử lý ghost
        ghostDistances = []
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            ghostDistances.append(distance)
            
            if newScaredTimes[i] > 0:
                # ghost đang scared - có thể đuổi theo(chỉ đuổi khi còn đủ thời gian)
                if newScaredTimes[i] > distance:
                    score += 200.0 / (distance + 1)
            else:
                if distance <= 1:
                    return -999999  
                elif distance <= 3:
                    score -= 200.0 / distance
                elif distance <= 5:
                    score -= 50.0 / distance
        
        if action == Directions.STOP:
            score -= 50
        
        if ghostDistances and min(ghostDistances) > 0:
            minGhostDistance = min(ghostDistances)
            if minGhostDistance > 3:
                score += 10
        
        return score
        #return successorGameState.getScore()

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
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            if nextAgent >= numAgents:
                nextAgent = 0
                nextDepth += 1
                
            if agentIndex == 0:
                actions = state.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(state)
                return max(minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in actions)
            
            else:
                actions = state.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(state)
                return min(minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in actions)

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')
        
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = minimax(successorState, 0, 1)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
                
        return bestAction
        "util.raiseNotDefined()"

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
                
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
                
            if agentIndex == 0:  
                v = float("-inf")
                for action in actions:
                    v = max(v, alphaBeta(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent, alpha, beta))
                    if v > beta:  
                        return v
                    alpha = max(alpha, v)
                return v
            else:  
                v = float("inf")
                for action in actions:
                    v = min(v, alphaBeta(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent, alpha, beta))
                    if v < alpha:  
                        return v
                    beta = min(beta, v)
                return v

        
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        bestValue = float("-inf")
        
        legalActions = gameState.getLegalActions(0)
        nextAgent = 1 % gameState.getNumAgents()
        nextDepth = 1 if nextAgent == 0 else 0
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
            
        return bestAction
        util.raiseNotDefined()

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
        def expectimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():                       
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                value = -float("inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max (value, expectimax (successor, depth, 1))
                return value
            else:
                if agentIndex < state.getNumAgents() -1:
                    nextAgent = agentIndex +1
                    nextDepth = depth
                else:
                    nextAgent = 0
                    nextDepth = depth +1
                total = 0
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    total += expectimax( successor, nextDepth, nextAgent)
                return total / len(actions)

        bestScore = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0,action)
            score = expectimax(successor,0,1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
            
                
        """util.raiseNotDefined()"""

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    foodList = foodGrid.asList()
    if foodList:
        minFoodDist = min([util.manhattanDistance(pacmanPos,food) for food in foodList])
        score +=1.0/minFoodDist
    score -= 4.0 * len(foodList)
    score -= 20.0 * len(capsules)
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = util.manhattanDistance( pacmanPos, ghostPos)
        if ghostState.ScaredTimer >0:
            score +=20.0 /(ghostDist +1)
        else:
            if ghostDist <=1:
                score -=1000.0
            else:
                score -= 5.0/ (ghostDist +1)
    return score
    """
    util.raiseNotDefined()
    """

# Abbreviation
better = betterEvaluationFunction
