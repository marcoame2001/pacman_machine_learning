# keyboardAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Directions
from game import GameStateData
import random


##Isa
import sys


class KeyboardAgent(Agent):
   # NOTE: Arrow keys also work.
    WEST_KEY  = 'a'
    EAST_KEY  = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY = 'q'

    def __init__( self, index = 0 ):

        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []
        self.countActions = 0

    def getAction( self, state):
        from graphicsUtils import keys_waiting
        from graphicsUtils import keys_pressed
        keys = keys_waiting() + keys_pressed()
        if keys != []:
            self.keys = keys

        legal = state.getLegalActions(self.index)
        move = self.getMove(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.lastMove in legal:
                move = self.lastMove

        if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

        if move not in legal:
            move = random.choice(legal)

        self.lastMove = move
        self.printLineData(state, move)
        self.countActions += 1
        return move

    def printLineData(self, gameState, move):
        line = ""
        line += str(gameState.getPacmanPosition()) + ", "
        line += self.legal_pacman(gameState) + ", "
        line += self.living_ghosts(gameState) + ", "
        line += str(gameState.getGhostPositions()) + ", "
        line += self.ghost_distances(gameState) + ", "
        pacman = gameState.getPacmanPosition()
        mapa = gameState.getWalls()
        for i in range(1, 6):
            j = 0
            position = [pacman[0] - i, pacman[1] + i]
            right_up_corner = [pacman[0] + i, pacman[1] + i]
            right_down_corner = [pacman[0] + i, pacman[1] - i]
            left_down_corner = [pacman[0] - i, pacman[1] - i]
            for new_position in range(i * 8):  # [(0, i), (i, i), (i, 0), (i, -i), (0, -i), (-i, -i), (-i, 0), (-i, i)]:
                if position[0] >= gameState.data.layout.width or position[0] < 0 or position[1] >= gameState.data.layout.height or position[1] < 0:
                    line += "1, "
                elif mapa[position[0]][position[1]]:
                    line += "1, "
                else:
                    line += "0, "
                if position == right_up_corner or position == right_down_corner or position == left_down_corner:
                    j += 1
                if j == 0:
                    position[0] += 1
                elif j == 1:
                    position[1] -= 1
                elif j == 2:
                    position[0] -= 1
                else:
                    position[1] += 1

        line += str(gameState.getScore()) + ", "
        line += str(move) + " "
        line = self.replaceData(line)
        f = open("data/training_keyboard.arff", "a+")
        if self.countActions != 0:
            f.write(str(gameState.getScore()))
        f.write("\n" + line)
        f.close()

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

    def getMove(self, legal):
        move = Directions.STOP
        if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
        if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
        if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
        if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move        

        
