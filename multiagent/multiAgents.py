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

        newFood=newFood.asList()
        ghostPos=[(G.getPosition()[0],G.getPosition()[1]) for G in newGhostStates]
        scared=min(newScaredTimes)>0

        if not scared and (newPos in ghostPos):
            return -1,0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist=sorted(newFood,key=lambda fDist:util.manhattanDistance(fDist,newPos))
        closestGhostDist=sorted(ghostPos,key=lambda gDist:util.manhattanDistance(gDist,newPos))

        fd=lambda fDis:util.manhattanDistance(fDis,newPos)
        gd=lambda gDis:util.manhattanDistance(gDis,newPos)

        return 1.0/fd(closestFoodDist[0])-1.0/gd(closestGhostDist[0])
#        return successorGameState.getScore()

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
        GhostIndex=[i for i in range(1,gameState.getNumAgents())]

        def term(state,d):
            return state.isWin() or state.isLose() or d==self.depth

        def min_value(state,d,ghost): #minimizer

            if term(state,d):
                return self.evaluationFunction(state)

            v=10000000000000000
            for action in state.getLegalActions(ghost):
                if ghost==GhostIndex[-1]:
                    v=min(v,max_value(state.generateSuccessor(ghost,action),d+1))
                else:
                    v=min(v,min_value(state.generateSuccessor(ghost,action),d,ghost+1))
            return v

        def max_value(state,d):
            if term(state,d):
                return self.evaluationFunction(state)
            v=-10000000000000000
            for action in state.getLegalActions(0):
                v=max(v,min_value(state.generateSuccessor(0,action),d,1))
            return v

        res=[(action,min_value(gameState.generateSuccessor(0,action),0,1)) for action in gameState.getLegalActions(0)]
        res.sort(key=lambda k:k[1])

        return res[-1][0]
            
                    
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabetaSearch(gameState,0,self.depth,-1e9,1e9)[1]

    def alphabetaSearch(self,gameState,agentIndex,depth,alpha,beta):
        if depth==0 or gameState.isWin() or gameState.isLose():
            ret=self.evaluationFunction(gameState),Directions.STOP
        elif agentIndex==0:
            ret=self.alphasearch(gameState,agentIndex,depth,alpha,beta)
        else:
            ret=self.betasearch(gameState,agentIndex,depth,alpha,beta)
        return ret

    def alphasearch(self,gameState,agentIndex,depth,alpha,beta):
        actions=gameState.getLegalActions(agentIndex)
        if agentIndex==gameState.getNumAgents()-1:
            next_agent,next_depth=0,depth-1
        else:
            next_agent,next_depth=agentIndex+1,depth
        max_score,max_action=-1e9,Directions.STOP
        for action in actions:
            successor_game_state=gameState.generateSuccessor(agentIndex,action)
            new_score=self.alphabetaSearch(successor_game_state,next_agent,next_depth,alpha,beta)[0]
            if new_score>max_score:
                max_score,max_action=new_score,action
            if new_score>beta:
                return new_score,action
            alpha=max(alpha,max_score)
        return max_score,max_action

    def betasearch(self,gameState,agentIndex,depth,alpha,beta):
        actions=gameState.getLegalActions(agentIndex)
        if agentIndex==gameState.getNumAgents()-1:
            next_agent,next_depth=0,depth-1
        else:
            next_agent,next_depth=agentIndex+1,depth
        min_score,min_action=1e9,Directions.STOP
        for action in actions:
            successor_game_state=gameState.generateSuccessor(agentIndex,action)
            new_score=self.alphabetaSearch(successor_game_state,next_agent,next_depth,alpha,beta)[0]

            if new_score<min_score:
                min_score,min_action=new_score,action
            if new_score<alpha:
                return new_score,action
            beta=min(beta,min_score)
        return min_score,min_action
                                
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
        "*** YOUR CODE HERE ***"
        return self.expectimaxsearch(gameState,0,self.depth)[1]

    def expectimaxsearch(self,game_state,agent_index,depth):
        if depth==0 or game_state.isWin() or game_state.isLose():
            ret=self.evaluationFunction(game_state),Directions.STOP
        elif agent_index==0:
            ret=self.maximizer(game_state,agent_index,depth)
        else:
            ret=self.expectation(game_state,agent_index,depth)
        return ret

    def maximizer(self,game_state,agent_index,depth):
        actions=game_state.getLegalActions(agent_index)
        if agent_index==game_state.getNumAgents()-1:
            next_agent,next_depth=0,depth-1
        else:
            next_agent,next_depth=agent_index+1,depth
        max_score,max_action=-1e9,Directions.STOP
        for action in actions:
            successor_game_state=game_state.generateSuccessor(agent_index,action)
            new_score=self.expectimaxsearch(successor_game_state,next_agent,next_depth)[0]
            if new_score>max_score:
                max_score,max_action=new_score,action
        return max_score,max_action

    def expectation(self,game_state,agent_index,depth):
        actions=game_state.getLegalActions(agent_index)
        if agent_index==game_state.getNumAgents()-1:
            next_agent,next_depth=0,depth-1
        else:
            next_agent,next_depth=agent_index+1,depth
        exp_score,exp_action=0,Directions.STOP
        for action in actions:
            successor_game_state=game_state.generateSuccessor(agent_index,action)
            exp_score+=self.expectimaxsearch(successor_game_state,next_agent,next_depth)[0]
        exp_score/=len(actions)
        return exp_score,exp_action
       # util.raiseNotDefined()

def betterEvaluationFunction(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:The heuristic function is simple, i.e manhattanDistance.
    Nearest ghost, nearest food dots, current score and scared time are taken
    into account. Nearer the ghost the worse, nearer the food the better; hence,
    the former is put in the numerator, while the latter in the denominator, wit
    h a negative sign.
    """
    "*** YOUR CODE HERE ***"
    pacman_pos=current_game_state.getPacmanPosition()
    food=current_game_state.getFood()
    foods=food.asList()
    ghost_states=current_game_state.getGhostStates()
    scared_times=[ghost_state.scaredTimer for ghost_state in ghost_states]

    nearest_ghost_dis=1e9
    for ghost_state in ghost_states:
        ghost_x,ghost_y=ghost_state.getPosition()
        ghost_x=int(ghost_x)
        ghost_y=int(ghost_y)
        if ghost_state.scaredTimer==0:
            nearest_ghost_dis=min(nearest_ghost_dis,manhattanDistance((ghost_x,ghost_y),pacman_pos))
        else:
            nearest_ghost_dis=-10
    nearest_food_dis=1e9
    for food in foods:
        nearest_food_dis=min(nearest_food_dis,manhattanDistance(food,pacman_pos))
    if not foods:
        nearest_food_dis=0
    return current_game_state.getScore()-7/(nearest_ghost_dis+1)-nearest_food_dis/3
   # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
