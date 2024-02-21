from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from wekaI import Weka
from builtins import range
from builtins import object
import util
from astar import astar
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from learningAgents import ReinforcementAgent

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.weka = Weka()
        self.weka.start_jvm()

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        move = self.chooseMove(gameState, legal)
        line = self.printLineData(gameState, move)

        x = self.convertStrToList(line)
        x.pop(6)
        print(x)
        print(len(x))
        a = self.weka.predict("./rf.model", x, r".\data\sin_muros\training_tutorial1_filter.arff")
        print(a)
        print(legal)
        if ( a == Directions.WEST ) and Directions.WEST in legal:  move = Directions.WEST
        elif ( a == Directions.EAST ) and Directions.EAST in legal: move = Directions.EAST
        elif ( a == Directions.NORTH ) and Directions.NORTH in legal:   move = Directions.NORTH
        elif ( a == Directions.SOUTH ) and Directions.SOUTH in legal: move = Directions.SOUTH
        else:
            n = random.randint(0, len(legal)-1)
            move = legal[n]
        return move

    @staticmethod
    def convertStrToList(str):
        li = list(str.split(", "))
        li.pop()
        for i in range(len(li)):
            li[i] = int(li[i])
        li = li
        return li

    def printLineData(self, gameState, move):
        line = ""
        line += str(gameState.getPacmanPosition()) + ", "
        line += self.legal_pacman(gameState) + ", "
        line += self.living_ghosts(gameState) + ", "
        line += str(gameState.getGhostPositions()) + ", "
        # line += self.ghost_distances(gameState) + ", "
        # pacman = gameState.getPacmanPosition()
        # mapa = gameState.getWalls()
        # for i in range(1, 6):
        #     j = 0
        #     position = [pacman[0] - i, pacman[1] + i]
        #     right_up_corner = [pacman[0] + i, pacman[1] + i]
        #     right_down_corner = [pacman[0] + i, pacman[1] - i]
        #     left_down_corner = [pacman[0] - i, pacman[1] - i]
        #     for new_position in range(i*8):  # [(0, i), (i, i), (i, 0), (i, -i), (0, -i), (-i, -i), (-i, 0), (-i, i)]:
        #         if position[0] >= gameState.data.layout.width or position[0] < 0 or position[1] >= gameState.data.layout.height or position[1] < 0:
        #             line += "1, "
        #         elif mapa[position[0]][position[1]]:
        #             line += "1, "
        #         else:
        #             line += "0, "
        #         if position == right_up_corner or position == right_down_corner or position == left_down_corner:
        #             j += 1
        #         if j == 0:
        #             position[0] += 1
        #         elif j == 1:
        #             position[1] -= 1
        #         elif j == 2:
        #             position[0] -= 1
        #         else:
        #             position[1] += 1

        line += str(gameState.getScore()) + ", "
        line += str(move) + " "
        line = self.replaceData(line)
        # print(lineholahola)
        # f = open("data/training_tutorial1.arff", "a+")
        # if self.countActions != 0:
        #     f.write(str(gameState.getScore()))
        # f.write("\n" + line)
        # f.close()
        return line

    def replaceData(self, line):
        char_to_replace = {"(": "",
                           ")": "",
                           "[": "",
                           "]": ""}
        for key, value in char_to_replace.items():
            line = line.replace(key, value)
        return line

    def legal_pacman(self, gameState):
        aux = []
        directions = ['North', 'South', 'East', 'West', 'Stop']
        legal = gameState.getLegalPacmanActions()
        for direction in directions:
            if direction in legal:
                aux.append(1)
            else:
                aux.append(0)
        return str(aux)

    def living_ghosts(self, gameState):
        aux = []
        ghosts = gameState.getLivingGhosts()
        ghosts.pop(0)
        for ghost in ghosts:
            if ghost is True:
                aux.append(1)
            else:
                aux.append(0)
        return str(aux)

    def ghost_distances(self, gameState):
        aux = []
        distances = gameState.data.ghostDistances
        for distance in distances:
            if distance is None:
                aux.append(0)
            else:
                aux.append(distance)
        return str(aux)

    def chooseMove(self, gameState, legal):
        distances = gameState.data.ghostDistances
        #Calcula el fantasma mas cercano
        ghost = self.min_ghost(distances)
        path = gameState.path
        if len(path) < 1:
            path = astar(gameState.getWalls(), gameState.getPacmanPosition(), gameState.getGhostPositions()[ghost])
            path.pop(0)
            self.addPath(gameState, path)
        # Direccion a la que se va a mover
        direction_x, direction_y = path[0]
        pacman_x, pacman_y = gameState.getPacmanPosition()
        gameState.path.pop(0)
        # Cuando esta en la misma vertical se movera horizontalmente
        if direction_y == pacman_y:
            if direction_x < pacman_x and Directions.WEST in legal:
                return Directions.WEST
            elif direction_x > pacman_x and Directions.EAST in legal:
                return Directions.EAST
        # Cuando esta en la misma horizontal se movera verticalmente
        elif direction_x == pacman_x:
            if direction_y > pacman_y and Directions.NORTH in legal:
                return Directions.NORTH
            elif direction_y < pacman_y and Directions.SOUTH in legal:
                return Directions.SOUTH

    def min_ghost(self, distances):
        min_dist = 999999
        ghost = 0
        for i in range(len(distances)):
            if distances[i] is not None and distances[i] < min_dist:
                min_dist = distances[i]
                ghost = i
        return ghost

    def addPath(self, gameState, path):
        for pos in path:
            gameState.path.append(pos)


class QLearningAgent:

    def __init__(self, **args):
        "Initialize Q-values"
        # ReinforcementAgent.__init__(self, **args)

        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3, "Stop": 4}
        self.table_file = open("qtable.txt", "r+")
        #        self.table_file_csv = open("qtable.csv", "r+")
        self.q_table = self.readQtable()
        self.epsilon = 0.3
        self.alpha = 0.5
        self.discount = 0.3
        self.count = 0
        self.prev = None
        self.prev_action = None
        self.prev_score = 0
        self.reward = 1

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

        #         self.table_file_csv.seek(0)
        #         self.table_file_csv.truncate()
        #         for line in self.q_table:
        #             for item in line[:-1]:
        #                 self.table_file_csv.write(str(item)+", ")
        #             self.table_file_csv.write(str(line[-1]))
        #             self.table_file_csv.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        distances = state.data.ghostDistances
        # Calcula el fantasma mas cercano
        ghost, distance = self.min_ghost(distances)
        pacman_x, pacman_y = state.getPacmanPosition()
        ghost_x, ghost_y = state.getGhostPositions()[ghost]
        num_pos_rel = {"IZQUIERDA": 0, "DERECHA": 1, "ABAJO": 2, "ARRIBA": 3,
                       "IZQUIERDA_ABAJO": 4, "IZQUIERDA_ARRIBA": 5, "DERECHA_ABAJO": 6, "DERECHA_ARRIBA": 7}
        pos_rel = ""

        if (pacman_x < ghost_x) and (pacman_y == ghost_y): # Caso izquierda
            pos_rel = "IZQUIERDA"
        elif (pacman_x > ghost_x) and (pacman_y == ghost_y): # Caso derecha
            pos_rel = "DERECHA"
        elif (pacman_x == ghost_x) and (pacman_y > ghost_y): # Caso abajo
            pos_rel = "ABAJO"
        elif (pacman_x == ghost_x) and (pacman_y < ghost_y): # Caso arriba
            pos_rel = "ARRIBA"
        elif (pacman_x < ghost_x) and (pacman_y > ghost_y): # Caso izquierda_abajo
            pos_rel = "IZQUIERDA_ABAJO"
        elif (pacman_x < ghost_x) and (pacman_y < ghost_y): # Caso izquierda_arriba
            pos_rel = "IZQUIERDA_ARRIBA"
        elif (pacman_x > ghost_x) and (pacman_y > ghost_y): # Caso derecha_abajo
            pos_rel = "DERECHA_ABAJO"
        elif (pacman_x > ghost_x) and (pacman_y < ghost_y): # Caso derecha_arriba
            pos_rel = "DERECHA_ARRIBA"

        # print(num_pos_rel[pos_rel], distance)
        return num_pos_rel[pos_rel] * 18 + distance

    def min_ghost(self, distances):
        min_dist = 999999
        ghost = 0
        for i in range(len(distances)):
            if distances[i] is not None and distances[i] < min_dist:
                min_dist = distances[i]
                ghost = i
        return ghost, min_dist

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        # print(position, action)
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """


        legalActions = state.getLegalPacmanActions()
        action = None

        if len(legalActions) == 0:
            # self.update(self.prev, self.prev_action, state, 100)
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            legalActions.remove("Stop")
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        # score = state.getScore()
        # reward = 1
        # print(self.prev, self.prev_action)
        # if self.prev == None:
        #     self.prev = state
        # print("Update: ", self.prev, action, state, reward)
        # self.update(self.prev, action, state, reward)
        # self.prev = state
        # self.prev_action = action
        # self.prev_score = score
        # self.count += 1
        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
        # print("Update Q-table with transition: ", state, action, nextState, reward)
        position = self.computePosition(state)
        action_column = self.actions[action]
        # print("Corresponding Q-table cell to update:", position, action_column)

        "*** YOUR CODE HERE ***"
        terminal_state = (nextState.isWin())
        if terminal_state:
            # position = self.computePosition(state)
            # action_column = self.actions[action]
            qvalue = (1 - self.alpha) * self.q_table[position][
                action_column] + self.alpha * reward
        else:
            # position = self.computePosition(state)
            # action_column = self.actions[action]
            best_next_action = self.computeActionFromQValues(nextState)
            # print(reward, self.discount, self.getQValue(nextState, best_next_action))
            qvalue = (1 - self.alpha) * self.q_table[position][
                action_column] + self.alpha * (reward + self.discount * self.getQValue(nextState, best_next_action))
        self.q_table[position][action_column] = qvalue
        print("QValue", qvalue)
        print("Reward", reward)
        # print("SIIIIIIIIIII")
        self.writeQtable()
        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        # print("Q-table:")
        # self.printQtable()

    def getPolicy(self, state):
        """Return the best action in the qtable for a given state"""
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        """Return the highest q value for a given state"""
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextState):
        reward = 0
        if nextState.isWin():
            return 10
        if sum(state.getLivingGhosts()) > sum(nextState.getLivingGhosts()):
            return 10
        if state.getNumFood() > nextState.getNumFood():
            reward += 5
        # print("Distance state:", state.data.ghostDistances)
        # print("Distance state:", self.min_ghost(state.data.ghostDistances)[1])
        # print("Distance nextState:", nextState.data.ghostDistances)
        # print("Distance nextState:", self.min_ghost(nextState.data.ghostDistances)[1])
        if self.min_ghost(state.data.ghostDistances)[1] > self.min_ghost(nextState.data.ghostDistances)[1]:
            reward += 3
        else:
            reward -= 1
        # print(sum(nextState.getLivingGhosts()), self.getPolicy(state), self.getPolicy(nextState))
        # print(game.isWin())
        return reward



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action