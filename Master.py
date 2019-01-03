from Initialize import cells, intializeInfra
import numpy as np
import random as rand
import time
import sys
from lane_start_index import lane_start_index
import pickle
from infractions import detect_collision
# for gui
import tkinter
from tkinter import *

# for image creation
from PIL import Image, ImageTk
from scipy.misc import imresize
import xlwt
from xlwt import Workbook

intializeInfra(cells)

possible_actions = ['GRRR', 'RGRR', 'RRGR', 'RRRG']

# Q learning parameters, to be tuned
iepsilon = 2
ilamda = 0.5
igamma = 0.5
ialpha = 0.2

# Road image layout
w, h = 74, 74
rgbArray = np.zeros((h, w, 3), dtype=np.uint8)

# initialization for Q-learning
"""
noOfstates = 32
noOfActions = 4
max_min = -sys.maxsize - 1
Q1 = np.ones((noOfstates, noOfActions))
Q1[1, 3] = Q1[2, 2] = Q1[3, 3] = Q1[4, 2] = Q1[5, 1] = Q1[6, 3] = Q1[8, 2] = Q1[9, 1] = Q1[11, 1] = Q1[12, 3] = Q1[
    13, 2] = Q1[14, 1] = Q1[17, 1] = max_min
Q1[18, 2] = Q1[19, 3] = Q1[20, 1] = Q1[22, 1] = Q1[23, 2] = Q1[25, 3] = Q1[26, 1] = Q1[27, 2] = Q1[28, 3] = Q1[29, 2] = \
    Q1[30, 3] = max_min

Q2 = np.ones((noOfstates, noOfActions))
Q2[1, 3] = Q2[2, 2] = Q2[3, 3] = Q2[4, 2] = Q2[5, 1] = Q2[6, 3] = Q2[8, 2] = Q2[9, 1] = Q2[11, 1] = Q2[12, 3] = Q2[
    13, 2] = Q2[14, 1] = Q2[17, 1] = max_min
Q2[18, 2] = Q2[19, 3] = Q2[20, 1] = Q2[22, 1] = Q2[23, 2] = Q2[25, 3] = Q2[26, 1] = Q2[27, 2] = Q2[28, 3] = Q2[29, 2] = \
    Q2[30, 3] = max_min

Q3 = np.ones((noOfstates, noOfActions))
Q3[1, 3] = Q3[2, 2] = Q3[3, 3] = Q3[4, 2] = Q3[5, 1] = Q3[6, 3] = Q3[8, 2] = Q3[9, 1] = Q3[11, 1] = Q3[12, 3] = Q3[
    13, 2] = Q3[14, 1] = Q3[17, 1] = max_min
Q3[18, 2] = Q3[19, 3] = Q3[20, 1] = Q3[22, 1] = Q3[23, 2] = Q3[25, 3] = Q3[26, 1] = Q3[27, 2] = Q3[28, 3] = Q3[29, 2] = \
    Q3[30, 3] = max_min

Q4 = np.ones((noOfstates, noOfActions))
Q4[1, 3] = Q4[2, 2] = Q4[3, 3] = Q4[4, 2] = Q4[5, 1] = Q4[6, 3] = Q4[8, 2] = Q4[9, 1] = Q4[11, 1] = Q4[12, 3] = Q4[
    13, 2] = Q4[14, 1] = Q4[17, 1] = max_min
Q4[18, 2] = Q4[19, 3] = Q4[20, 1] = Q4[22, 1] = Q4[23, 2] = Q4[25, 3] = Q4[26, 1] = Q4[27, 2] = Q4[28, 3] = Q4[
    29, 2] = \
    Q4[30, 3] = max_min

for index in range(noOfstates):
    Q1[index, 0] = max_min
    Q2[index, 0] = max_min
    Q3[index, 0] = max_min
    Q4[index, 0] = max_min

# Q2 = Q1
# Q3 = Q1
# Q4 = Q1

Q1[13,1] = Q1[5, 3] = Q1[22,2] = Q1[29, 3] = max_min
Q1[18,1] = Q1[26,3] = Q1[9,3] = Q1[2,1] = max_min
Q2[2,1] = Q2[9,3] = Q2[5,3] = Q2[13,1] = max_min
Q2[22,3] = Q2[29,1] = Q2[18,3] = Q2[26,2] = max_min
Q3[9,3] = Q3[2,1] = Q3[5,2] = Q3[13,3] = max_min
Q3[18,1] = Q3[26,3] = Q3[29,1] = Q3[22,3] = max_min
Q4[9,2] = Q4[2,3] = Q4[13,1] = Q4[5,3] = max_min
Q4[22,3] = Q4[29,1] = Q4[26,3] = Q4[18,1] = max_min
Q = [Q1, Q2, Q3, Q4]
"""
# loads Q matrix values from file and initializes itself
with open('outfile1', 'rb') as fp:
    Q = pickle.load(fp)

#print("Q matrix for start position 1: \n",Q[0])
#print("Q matrix for start position 2: \n",Q[1])
#print("Q matrix for start position 3: \n",Q[2])
#print("Q matrix for start position 4: \n",Q[3])
#exit()

def get_juncn_num(xpos, ypos):
    juncn_num = np.zeros(2)
    if (xpos == 4 and ypos == 2):
        juncn_num[0] = 1
        juncn_num[1] = 3
    elif (xpos == 3 and ypos == 5):
        juncn_num[0] = 1
        juncn_num[1] = 1
    elif (xpos == 5 and ypos == 4):
        juncn_num[0] = 1
        juncn_num[1] = 2

    elif (xpos == 4 and ypos == 35):
        juncn_num[0] = 2
        juncn_num[1] = 3
    elif (xpos == 3 and ypos == 38):
        juncn_num[0] = 2
        juncn_num[1] = 1
    elif (xpos == 5 and ypos == 37):
        juncn_num[0] = 2
        juncn_num[1] = 2

    elif (xpos == 4 and ypos == 68):
        juncn_num[0] = 3
        juncn_num[1] = 3
    elif (xpos == 2 and ypos == 69):
        juncn_num[0] = 3
        juncn_num[1] = 0
    elif (xpos == 5 and ypos == 70):
        juncn_num[0] = 3
        juncn_num[1] = 2

    elif (xpos == 35 and ypos == 3):
        juncn_num[0] = 4
        juncn_num[1] = 0
    elif (xpos == 36 and ypos == 5):
        juncn_num[0] = 4
        juncn_num[1] = 1
    elif (xpos == 38 and ypos == 4):
        juncn_num[0] = 4
        juncn_num[1] = 2

    elif (xpos == 35 and ypos == 36):
        juncn_num[0] = 5
        juncn_num[1] = 0
    elif (xpos == 36 and ypos == 38):
        juncn_num[0] = 5
        juncn_num[1] = 1
    elif (xpos == 38 and ypos == 37):
        juncn_num[0] = 5
        juncn_num[1] = 2
    elif (xpos == 37 and ypos == 35):
        juncn_num[0] = 5
        juncn_num[1] = 3

    elif (xpos == 35 and ypos == 69):
        juncn_num[0] = 6
        juncn_num[1] = 0
    elif (xpos == 37 and ypos == 68):
        juncn_num[0] = 6
        juncn_num[1] = 3
    elif (xpos == 38 and ypos == 70):
        juncn_num[0] = 6
        juncn_num[1] = 2

    elif (xpos == 68 and ypos == 3):
        juncn_num[0] = 7
        juncn_num[1] = 0
    elif (xpos == 69 and ypos == 5):
        juncn_num[0] = 7
        juncn_num[1] = 1
    elif (xpos == 71 and ypos == 4):
        juncn_num[0] = 7
        juncn_num[1] = 2

    elif (xpos == 68 and ypos == 36):
        juncn_num[0] = 8
        juncn_num[1] = 0
    elif (xpos == 69 and ypos == 38):
        juncn_num[0] = 8
        juncn_num[1] = 1
    elif (xpos == 70 and ypos == 35):
        juncn_num[0] = 8
        juncn_num[1] = 3

    elif (xpos == 68 and ypos == 69):
        juncn_num[0] = 9
        juncn_num[1] = 0
    elif (xpos == 69 and ypos == 71):
        juncn_num[0] = 9
        juncn_num[1] = 1
    elif (xpos == 70 and ypos == 68):
        juncn_num[0] = 9
        juncn_num[1] = 3
    return juncn_num


class Car:
    startTime = None
    dir = None
    pos = np.zeros(2)
    destination = np.zeros(2)
    dest_node = None
    start_id = None
    s = None
    penalty = 0
    lastpos = np.zeros(2)
    epsilon = 0.05
    def __init__(self, start_position):
        self.startTime = time.time()
        self.pos = start_position
        self.lastpos = np.array([0, 0])
        cells[self.pos[0]][self.pos[1]].occupied = True
        self.Id = str(start_position) + "T" + str(time.time())

        if self.pos[0] == 73 and self.pos[1] == 4:
            self.destination = np.array([0, 70])
            self.dir = "up"
            self.dest_node = 3
            self.start_id = 2
            self.s = 26

        elif self.pos[0] == 4 and self.pos[1] == 0:
            self.destination = np.array([70, 73])
            self.dir = "right"
            self.dest_node = 9
            self.start_id = 0
            self.s = 2

        elif self.pos[0] == 0 and self.pos[1] == 69:
            self.destination = np.array([73, 3])
            self.dir = "down"
            self.dest_node = 7
            self.start_id = 1
            self.s = 7

        elif self.pos[0] == 69 and self.pos[1] == 73:
            self.destination = np.array([3, 0])
            self.dir = "left"
            self.dest_node = 1
            self.start_id = 3
            self.s = 31

    def __del__(self):
        print(end="")

    #        print("Car reached its Destination deleted object")

    def is_frontclear(self):

        if self.dir == "right":
            return False if cells[self.pos[0]][self.pos[1] + 1].occupied else True
        elif self.dir == "left":
            return False if cells[self.pos[0]][self.pos[1] - 1].occupied else True
        elif self.dir == "up":
            return False if cells[self.pos[0] - 1][self.pos[1]].occupied else True
        elif self.dir == "down":
            return False if cells[self.pos[0] + 1][self.pos[1]].occupied else True
        else:
            return False

    def is_junction(self):
        if self.dir == "right":
            return True if cells[self.pos[0]][self.pos[1] + 1].type == "junction" else False
        elif self.dir == "left":
            return True if cells[self.pos[0]][self.pos[1] - 1].type == "junction" else False
        elif self.dir == "up":
            return True if cells[self.pos[0] - 1][self.pos[1]].type == "junction" else False
        elif self.dir == "down":
            return True if cells[self.pos[0] + 1][self.pos[1]].type == "junction" else False
        else:
            return True

    def is_injunction(self):
        if self.dir == "right":
            return True if cells[self.pos[0]][self.pos[1]].type == "junction" and cells[self.pos[0]][
                self.pos[1] + 1].type == "junction" else False
        elif self.dir == "left":
            return True if cells[self.pos[0]][self.pos[1]].type == "junction" and cells[self.pos[0]][
                self.pos[1] - 1].type == "junction" else False
        elif self.dir == "up":
            return True if cells[self.pos[0]][self.pos[1]].type == "junction" and cells[self.pos[0] - 1][
                self.pos[1]].type == "junction" else False
        elif self.dir == "down":
            return True if cells[self.pos[0]][self.pos[1]].type == "junction" and cells[self.pos[0] + 1][
                self.pos[1]].type == "junction" else False
        else:
            return False

    def is_destination(self):
        if cells[self.pos[0]][self.pos[1]] == self.destination:
            return True
        else:
            return False

    def turn_left(self):
        if self.dir == "right":
            self.pos[0] -= 2
            self.pos[1] += 1
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] + 2][self.pos[1] - 1].occupied = False
            self.dir = "up"

        elif self.dir == "left":
            self.pos[0] += 2
            self.pos[1] -= 1
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] - 2][self.pos[1] + 1].occupied = False
            self.dir = "down"

        elif self.dir == "up":
            self.pos[0] -= 1
            self.pos[1] -= 2
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] + 1][self.pos[1] + 2].occupied = False
            self.dir = "left"

        elif self.dir == "down":
            self.pos[0] += 1
            self.pos[1] += 2
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] - 1][self.pos[1] - 2].occupied = False
            self.dir = "right"

    def turn_right(self):
        if self.dir == "right":
            self.pos[0] += 1
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] - 1][self.pos[1]].occupied = False
            self.dir = "down"

        elif self.dir == "left":
            self.pos[0] -= 1
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] + 1][self.pos[1]].occupied = False
            self.dir = "up"

        elif self.dir == "up":
            self.pos[1] += 1
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0]][self.pos[1] - 1].occupied = False
            self.dir = "right"

        elif self.dir == "down":
            self.pos[1] -= 1
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0]][self.pos[1] + 1].occupied = False
            self.dir = "left"

    def go_forward(self):
        if self.dir == "right":
            self.pos[1] += 2
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0]][self.pos[1] - 2].occupied = False

        elif self.dir == "left":
            self.pos[1] -= 2
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0]][self.pos[1] + 2].occupied = False

        elif self.dir == "up":
            self.pos[0] -= 2
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] + 2][self.pos[1]].occupied = False

        elif self.dir == "down":
            self.pos[0] += 2
            cells[self.pos[0]][self.pos[1]].occupied = True
            # cells[self.pos[0] - 2][self.pos[1]].occupied = False

    def isfree_cellof_action(self,action):
        if self.dir == "right":
            if action == 1:
                return False if (cells[self.pos[0]][self.pos[1]+3].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]+1][self.pos[1]+1].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]-2][self.pos[1]+2].occupied is True) else True
            else:
                return False

        elif self.dir == "left":
            if action == 1:
                return False if (cells[self.pos[0]][self.pos[1]-3].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]-1][self.pos[1]-1].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]+2][self.pos[1]-2].occupied is True) else True
            else:
                return False

        elif self.dir == "up":
            if action == 1:
                return False if (cells[self.pos[0]-3][self.pos[1]].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]-1][self.pos[1]+1].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]-2][self.pos[1]-2].occupied is True) else True
            else:
                return False

        elif self.dir == "down":
            if action == 1:
                return False if (cells[self.pos[0]+3][self.pos[1]].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]+1][self.pos[1]-1].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]+2][self.pos[1]+2].occupied is True) else True
            else:
                return False

    def isfree_injunc(self, action):
        if self.dir == "right":
            if action == 1:
                return False if (cells[self.pos[0]][self.pos[1]+2].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]+1][self.pos[1]].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]-2][self.pos[1]+1].occupied is True) else True
            else:
                return False

        elif self.dir == "left":
            if action == 1:
                return False if (cells[self.pos[0]][self.pos[1]-2].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]-1][self.pos[1]].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]+2][self.pos[1]-1].occupied is True) else True
            else:
                return False

        elif self.dir == "up":
            if action == 1:
                return False if (cells[self.pos[0]-2][self.pos[1]].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]][self.pos[1]+1].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]-1][self.pos[1]-2].occupied is True) else True
            else:
                return False

        elif self.dir == "down":
            if action == 1:
                return False if (cells[self.pos[0]+2][self.pos[1]].occupied is True) else True
            if action == 2:
                return False if (cells[self.pos[0]][self.pos[1]-1].occupied is True) else True
            if action == 3:
                return False if (cells[self.pos[0]+1][self.pos[1]+2].occupied is True) else True
            else:
                return False

    def valid_action(self, state):
        if(self.destination == np.array([0,70])).all():
            if(state in [6,3,19,30]):
                acn = [3]
            elif(state in [10,27,23]):
                acn = [2]
            elif(state in [14,1,8,25,32]):
                acn = [1]
            elif(state in [2,4,7,13,20,26,31,29]):
                acn = [1,2]
            elif(state in [5,9,24,28]):
                acn = [1,3]
            elif(state in [12,15,18,21]):
                acn = [2,3]
            elif(state in [11,16,17,22]):
                acn = [1,2,3]
            return acn

        elif(self.destination == np.array([70,73])).all():
            if(state in  [23,14,3,19]):
                acn = [3]
            elif(state in  [6,10,27]):
                acn = [2]
            elif(state in [30,1,8,25,32]):
                acn = [1]
            elif(state in [2,4,7,13,20,26,31,29]):
                acn = [1,2]
            elif(state in [5,9,24,28]):
                acn = [1,3]
            elif(state in [12,15,18,21]):
                acn = [2,3]
            elif(state in [11,16,17,22]):
                acn = [1,2,3]
            return acn

        elif(self.destination == np.array([73,3])).all():
            if(state in  [27,3,30,14]):
                acn = [3]
            elif(state in [10,6,23]):
                acn = [2]
            elif(state in [19,1,8,25,32]):
                acn = [1]
            elif(state in [2,4,7,13,20,26,31,29]):
                acn = [1,2]
            elif(state in [5,9,24,28]):
                acn = [1,3]
            elif(state in [12,15,18,21]):
                acn = [2,3]
            elif(state in [11,16,17,22]):
                acn = [1,2,3]
            return acn

        elif(self.destination == np.array([3,0])).all():
            if(state in  [10,19,30,14]):
                acn = [3]
            elif(state in [27,23,6]):
                acn = [2]
            elif(state in [3,1,8,25,32]):
                acn = [1]
            elif(state in [2,4,7,13,20,26,31,29]):
                acn = [1,2]
            elif(state in [5,9,24,28]):
                acn = [1,3]
            elif(state in [12,15,18,21]):
                acn = [2,3]
            elif(state in [11,16,17,22]):
                acn = [1,2,3]
            return acn

    def QLearning(self, act):
        state = self.s - 1
        alpha = 0.6
        gama = 0.5
        #        actions = ["stop", "forward", "right", "left"]
        #        def choose_action_epsilongreedy(current_state):
        #            if random.uniform(0, 1) < epsilon:
        #                return random.randint(0, noOfActions - 1)
        #            else:
        #                return Q[self.start_id][current_state, :].argmax()
        a = act
        # a = Q[self.start_id][state, :].argmax()
        #        print("acton: ", a, "State: ", self.s)
        s_dash, r = self.get_next_state_and_reward(a, self.s)
        s_dash -= 1
        r -= self.penalty
        delta = r + gama * Q[self.start_id][s_dash, :].max() - Q[self.start_id][state, a]
        dQ = alpha * delta
        Q[self.start_id][state, a] = Q[self.start_id][state, a] + dQ
        self.s = s_dash + 1


    def get_next_state_and_reward(self, action, pr_state):
        rwd = 0
        if action == 0:
            next_state = pr_state
        elif action == 1:
            if pr_state in [20, 22, 24]:
                next_state = pr_state - 10
            elif pr_state in [14, 26]:
                next_state = pr_state - 6
                if pr_state == 14 and (self.destination == np.array([0, 70])).all():
                    rwd = 10
            elif pr_state in [3, 5, 17, 29, 31]:
                next_state = pr_state - 2
                if pr_state == 3 and (self.destination == np.array([3, 0])).all():
                    rwd = 10
            elif pr_state in [2, 4, 16, 28, 30]:
                next_state = pr_state + 2
                if pr_state == 30 and (self.destination == np.array([70, 73])).all():
                    rwd = 10
            elif pr_state in [7, 19]:
                next_state = pr_state + 6
                if pr_state == 19 and (self.destination == np.array([73, 3])).all():
                    rwd = 10
            elif pr_state in [9, 11, 13]:
                next_state = pr_state + 10
            else:
                next_state = pr_state
        elif action == 2:
            if pr_state in [27, 29, 31]:
                next_state = pr_state - 7
            elif pr_state in [10, 12]:
                next_state = pr_state - 6
            elif pr_state in [15, 17]:
                next_state = pr_state - 5
            elif pr_state in [20, 22]:
                next_state = pr_state - 4
            elif pr_state in [7]:
                next_state = pr_state - 2
            elif pr_state in [26]:
                next_state = pr_state + 2
            elif pr_state in [11, 13]:
                next_state = pr_state + 4
            elif pr_state in [16, 18]:
                next_state = pr_state + 5
            elif pr_state in [21, 23]:
                next_state = pr_state + 6
            elif pr_state in [2, 4, 6]:
                next_state = pr_state + 7
            else:
                next_state = pr_state
        elif action == 3:
            if pr_state in [10, 12, 14]:
                next_state = pr_state - 9
                if pr_state == 10 and (self.destination == np.array([3, 0])).all():
                    rwd = 10
            elif pr_state in [22, 24]:
                next_state = pr_state - 7
            elif pr_state in [28, 30]:
                next_state = pr_state - 6
            elif pr_state in [16, 18]:
                next_state = pr_state - 4
            elif pr_state in [27]:
                next_state = pr_state - 2
                if (self.destination == np.array([73, 3])).all():
                    rwd = 10
            elif pr_state in [6]:
                next_state = pr_state + 2
                if (self.destination == np.array([0, 70])).all():
                    rwd = 10
            elif pr_state in [15, 17]:
                next_state = pr_state + 4
            elif pr_state in [3, 5]:
                next_state = pr_state + 6
            elif pr_state in [9, 11]:
                next_state = pr_state + 7
            elif pr_state in [19, 21, 23]:
                next_state = pr_state + 9
                if pr_state == 23 and (self.destination == np.array([70, 73])).all():
                    rwd = 10
            else:
                next_state = pr_state
        return next_state, rwd

    def choose_action_epsilongreedy(self):
            if rand.uniform(0, 1) < self.epsilon:
                actn = self.valid_action(self.s)
                index = rand.randint(0, len(actn) - 1)
                act = actn[index]
                # print("action: ", act[index],"   at state..: ",self.s)
            #            print("exploring with action  :",act,"  at position",self.s)
            else:
                act = Q[self.start_id][self.s - 1, :].argmax()
            #           print("exploitation action: ",act,"got from Q ",self.start_id,self.s -1)
            return act

    @property
    def drive(self):
        if not(self.lastpos == self.pos).all():
            #print(self.lastpos)
            cells[self.lastpos[0]][self.lastpos[1]].occupied = False
        self.lastpos = np.array([self.pos[0], self.pos[1]])
        if (self.pos == self.destination).all():
            #            print("reached Destination")
            return 1
        else:
            #action = Q[self.start_id][self.s - 1, :].argmax()
            action = self.choose_action_epsilongreedy()
            if self.is_junction() and not self.is_injunction():

                junction_number = get_juncn_num(self.pos[0], self.pos[1])
                light = junction[int(junction_number[0]) - 1].lights[int(junction_number[1])]
                if (light == 'G') and self.isfree_cellof_action(action):
                    if self.dir == "right":
                        self.pos[1] += 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0]][self.pos[1] - 1].occupied = False

                    elif self.dir == "left":
                        self.pos[1] -= 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0]][self.pos[1] + 1].occupied = False

                    elif self.dir == "up":
                        self.pos[0] -= 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0] + 1][self.pos[1]].occupied = False

                    elif self.dir == "down":
                        self.pos[0] += 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0] - 1][self.pos[1]].occupied = False
                    if action == 2 :
                            self.turn_right()
                            self.QLearning(action)
                            self.penalty = 0
                            return 0
                    elif action == 3 :
                            self.turn_left()
                            self.QLearning(action)
                            self.penalty = 0
                            return 0
                    elif action == 1 :
                            self.go_forward()
                            self.QLearning(action)
                            self.penalty = 0
                            return 0
                else:
                    self.pos = self.pos
                    #                    print("waiting for green light")
                    self.penalty += 0.1
                    return 0
            if not self.is_junction():
                if self.is_frontclear():
                    if self.dir == "right":
                        self.pos[1] += 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0]][self.pos[1] - 1].occupied = False

                    elif self.dir == "left":
                        self.pos[1] -= 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0]][self.pos[1] + 1].occupied = False

                    elif self.dir == "up":
                        self.pos[0] -= 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0] + 1][self.pos[1]].occupied = False

                    elif self.dir == "down":
                        self.pos[0] += 1
                        cells[self.pos[0]][self.pos[1]].occupied = True
                        # cells[self.pos[0] - 1][self.pos[1]].occupied = False
                else:
                    self.penalty += 0.1
                # print("new pos: ", self.pos)
            return 0

# Junction class
# This has the basic parameters for each junction in the road infrastructure
class Junction:
    wait_time = [0, 0, 0, 0]  # This variable has the waiting time count of the cars in each of the lanes
    congestion_count = [0, 0, 0, 0]  # Stores the congestion of each lane at a junction
    valid_directions = [0, 0, 0, 0]  # indicates the valid lane for each junction, set when creating the junction object
    junction_name = ""  # has the name of the junction (J1-J9), display purpose
    junction_index = 0  # junction index number (0-8)
    lights = ''  # Road traffic signal of each lane of the junction, in clockwise

    # Constructor
    def __init__(self, junction_name, index, valid_directions_for_junction):
        self.valid_directions = valid_directions_for_junction
        self.junction_name = junction_name
        self.junction_index = index
        self.wait_time = [0, 0, 0, 0]
        self.congestion_count = [0, 0, 0, 0]
        self.lights = ''

    # Calculates the wait time of the cars waiting in the lane of a junction with Red Signal
    def wait_time_counter(self):
        for i in range(4):
            lane_idx = [lane_start_index[self.junction_index][2 * i], lane_start_index[self.junction_index][2 * i + 1]]
            if self.valid_directions[i] == 1:
                if self.lights[i] == 'R' and cells[lane_idx[0]][lane_idx[1]].occupied == 1:
                    self.wait_time[i] = self.wait_time[i] + 1
                else:
                    self.wait_time[i] = 0
            else:
                self.wait_time[i] = 0

    # Finds the congestion in each lane
    def congestion_finder(self):

        for i in range(4):
            # i = 0 North and so on
            # invalid direction congestion count = 0 and continue to next loop
            if self.valid_directions[i] == 0:
                self.congestion_count[i] = 0
                continue

            # Select action on each row and column
            if i == 0:  # N
                cr_inc = -1
                cl_inc = 0
            elif i == 1:  # E
                cr_inc = 0
                cl_inc = 1
            elif i == 2:  # S
                cr_inc = 1
                cl_inc = 0
            else:  # W
                cr_inc = 0
                cl_inc = -1

            self.congestion_count[i] = 0
            cr = lane_start_index[self.junction_index][2 * i]  # row index
            cl = lane_start_index[self.junction_index][2 * i + 1]  # col index

            for j in range(28):
                if cells[cr][cl] is None:
                    break
                elif cr == 73 or cl == 73:
                    break
                if cells[cr][cl].occupied:
                    self.congestion_count[i] += 1
                    cr += cr_inc
                    cl += cl_inc
                else:
                    break


# Create Junction
junction = np.empty(9, dtype=Junction)

junction[0] = Junction("J1", 0, [0, 1, 1, 1])
junction[1] = Junction("J2", 1, [0, 1, 1, 1])
junction[2] = Junction("J3", 2, [1, 0, 1, 1])
junction[3] = Junction("J4", 3, [1, 1, 1, 0])
junction[4] = Junction("J5", 4, [1, 1, 1, 1])
junction[5] = Junction("J6", 5, [1, 0, 1, 1])
junction[6] = Junction("J7", 6, [1, 1, 1, 0])
junction[7] = Junction("J8", 7, [1, 1, 0, 1])
junction[8] = Junction("J9", 8, [1, 1, 0, 1])

junc_pos_array = [3, 4, 36, 37, 69, 70]


def find_features(junction):
    feature = np.zeros([9, 4, 6])
    for n in range(9):
        for i in range(4):

            if junction[n].valid_directions[i] == 0:
                continue
            if cells[lane_start_index[n][2*i]][lane_start_index[n][2*i + 1]].occupied is True:
            #if junction[n].congestion_count[i] >= 2:
                feature[n][i][0] = 1
            if junction[n].congestion_count[i] > 5:
                feature[n][i][1] = 1
            if junction[n].congestion_count[i] > 15:
                feature[n][i][2] = 1
            if junction[n].wait_time[i] > 1:
                feature[n][i][3] = 1
            if junction[n].wait_time[i] > 4:
                feature[n][i][4] = 1
            if junction[n].wait_time[i] > 10:
                feature[n][i][5] = 1
    return feature


# epsilon greedy explore function
def explore(n, junction):
    random = rand.randint(0, 3)
    if junction[n].valid_directions[random] == 0:
        if random == 3:
            return rand.randint(0, 2)
        elif random == 0:
            return rand.randint(1, 3)
        elif random == 1:
            seq = [0, 2, 3]
            return rand.choice(seq)
        elif random == 2:
            seq = [0, 1, 3]
            return rand.choice(seq)
    else:
        return random


# epsilon greedy exploit function
def exploit(qvalue, n, junction):
#    qval = [abs(number) for number in qvalue[n]]
    qval = qvalue[n]
    maxq = np.argmax(qval)
    if junction[n].valid_directions[maxq] == 0:
        if maxq == 3:
            new_q = [qval[0], qval[1], qval[2]]
            max = np.argmax(new_q)
        if maxq == 2:
            new_q = [qval[0], qval[1], qval[3]]
            max = np.argmax(new_q)
            if max == 2:
                max = 3
        if maxq == 1:
            new_q = [qval[0], qval[2], qval[3]]
            max = np.argmax(new_q)
            if max == 1:
                max =2
        if maxq == 0:
            new_q = [qval[1], qval[2], qval[3]]
            max = np.argmax(new_q)
            if max == 0:
                max =1

        maxq = max

    return maxq


# Choose action based on epsilon greedy
def choose_action(qvalue, e, junction):
    x = rand.randint(0, 10) < iepsilon

    action = np.zeros(9)
    if x:
        for n in range(9):
            for i in range(len(e[n])):
                e[n][i] = 0
            action[n] = explore(n, junction)

    else:
        for n in range(9):
            for i in range(len(e[n])):
                e[n][i] = igamma * ilamda * e[n][i]
            action[n] = exploit(qvalue, n, junction)

    return action, e


def update_junction_lights(action_i_team):
    for n in range(9):
        junction[n].lights = str(possible_actions[int(action_i_team[n])])


def check_throughput(cars):
    output = 0

    for car in cars:
        if (car.pos == car.destination).all():
            output = output + 1
    #    print("output is: ", output)
    return output


def inject_cars_probability(car,th):
    if(th >1):
      prob = 0.18
      if not cells[4][0].occupied and rand.uniform(0, 1) < prob:
          car = np.append(car, Car([4, 0]))
      if not cells[0][69].occupied and rand.uniform(0, 1) < prob:
          car = np.append(car, Car([0, 69]))
      if not cells[69][73].occupied and rand.uniform(0, 1) < prob:
          car = np.append(car, Car([69, 73]))
      if not cells[73][4].occupied and rand.uniform(0, 1) < prob:
          car = np.append(car, Car([73, 4]))
      return car
    else:
      if cells[4][0].occupied == False:
         car = np.append(car, Car([4, 0]))
      if cells[0][69].occupied == False:
         car = np.append(car, Car([0, 69]))
      if cells[69][73].occupied == False:
         car = np.append(car, Car([69, 73]))
      if cells[73][4].occupied == False:
         car = np.append(car, Car([73, 4]))
      return car


def inject_cars_congestionbased(car):
    conjestioncount = 0
    for n in range(9):
        conjestioncount += sum(junction[n].congestion_count)
    if(conjestioncount < 71):
        if cells[4][0].occupied is False:
            car = np.append(car, Car([4, 0]))
        if cells[0][69].occupied is False:
            car = np.append(car, Car([0, 69]))
        if cells[69][73].occupied is False:
            car = np.append(car, Car([69, 73]))
        if cells[73][4].occupied is False:
            car = np.append(car, Car([73, 4]))
    else:
        return car
    return car



def master(num_episodes, num_steps):

    wb = Workbook()
    sheet1 = wb.add_sheet('Theta sheet')
    xlsCol = 0
    xlsRow = 0

    # Tracker GUI
    track_gui = Tk()
    # create canvas for GUI
    road_img = Canvas(track_gui, width = 1900, height = 1900)
    road_img.pack(expand=YES, fill=BOTH)
    rgbArray_road = imresize(rgbArray,(74*10, 74*10), interp='nearest')
    img = Image.fromarray(rgbArray_road, 'RGB')
    road_cars_img = ImageTk.PhotoImage(img)
    created_canvas = road_img.create_image(0,0,image=road_cars_img, anchor="nw")

    collided_cars = {}
    infraction_cars = {}
    direction_violators = {}

    theta_i_team = np.ones([9, 6])
    step_no = 0
    # V team Q values definition

    throughput = 0

    cars = np.array([], dtype=Car)
    if cells[4][0].occupied == False:
        cars = np.append(cars, Car([4, 0]))
    if cells[0][69].occupied == False:
        cars = np.append(cars, Car([0, 69]))
    if cells[69][73].occupied == False:
        cars = np.append(cars, Car([69, 73]))
    if cells[73][4].occupied == False:
        cars = np.append(cars, Car([73, 4]))

    for episode in range(num_episodes):

        #sheet1.write( xlsRow, 0, " Q Value : ")
        #sheet2.write( xlsRow, 0, " Theta : ")

        #sheet1.write( xlsRow, 1, episode)
        #sheet2.write( xlsRow, 1, episode)

        #xlsRow = xlsRow + 1

        e_i_team = np.zeros([9, 6])
        qvalue_i_team = np.zeros([9, 4])

        action_i_team, e_i_team = choose_action(qvalue_i_team, e_i_team, junction)
        update_junction_lights(action_i_team)

        for update in range(len(junction)):
            junction[update].congestion_finder()
            junction[update].wait_time_counter()

        feature = find_features(junction)


        for i in range(9):
            for j in range(4):
                for k in range(6):
                    qvalue_i_team[i][j] += (feature[i][j][k] * theta_i_team[i][k])

        for i in range(9):
            for j in range(4):
                xlsCol = i*4 + j
                sheet1.write( xlsRow, xlsCol, theta_i_team[i][j])

        xlsRow = xlsRow + 1
        for step in range(num_steps):

            action_i_team, e_i_team = choose_action(qvalue_i_team, e_i_team, junction)
            update_junction_lights(action_i_team)

            # create image here
            # call the gui function take input cars
            car_tracker(cars, road_img, created_canvas, secs=0.0)


            # i team updates each junction object with its sequence of lights
            cars_dash = np.array([], dtype=Car)
            for car in cars:
                if not((car.pos == car.destination).all()):

                    if (car.pos == np.array([0, 70])).all() or (car.pos == np.array([70, 73])).all() or (
                            car.pos == np.array([73, 3])).all() or (car.pos == np.array([3, 0])).all():
                        print("wrong desination",car.Id)
                        del (car)
                    else:

                        # if (car.pos == np.array([0, 70])).all() or (car.pos==np.array([70, 73])).all() or (car.pos==np.array([73, 3])).all() or (car.pos==np.array([3, 0])).all():
                        #    del (car)
                        # else:
                        cars_dash = np.append(cars_dash, car)
                else:
                    #                    print(car.Id, " reached destination")
                    throughput += 1
                    print("throughput is", throughput)
                    cells[car.lastpos[0]][car.lastpos[1]].occupied = False
                    cells[car.pos[0]][car.pos[1]].occupied = False
                    del (car)
            cars = cars_dash

            ## injection of cars
            #if cells[4][0].occupied == False:
            #    cars = np.append(cars, Car([4, 0]))
            #if cells[0][69].occupied == False:
            #    cars = np.append(cars, Car([0, 69]))
            #if cells[69][73].occupied == False:
            #    cars = np.append(cars, Car([69, 73]))
            #if cells[73][4].occupied == False:
            #    cars = np.append(cars, Car([73, 4]))

            cars = inject_cars_congestionbased(cars)
            #cars = inject_cars_probability(cars,throughput)

            for car in cars:
                car.drive
            #                print(car.pos, end="   ")

            if throughput > 0:
                print("cars reached is: ", throughput)
                step_no += 1
                print("Step number: ", step_no)


            #if step + 150 * episode != 0 and throughput > 0:
#              print("cars reached proper destination per step: ", throughput / (step + 150 * episode))

            # V team can now take an action according to the junction lights and update the environment
            for update in range(len(junction)):
                junction[update].congestion_finder()
                junction[update].wait_time_counter()


            X = detect_collision(cars, junction, cells)
            collided_cars.update(X[0])
            infraction_cars.update(X[1])
            direction_violators.update(X[2])


            qvalue_i_team, theta_i_team, e_i_team = i_team_learn(feature, action_i_team, qvalue_i_team, e_i_team, theta_i_team, cars)

            for i in range(9):
                for j in range(6):
                    xlsCol = i*6 + j
                    sheet1.write( xlsRow, xlsCol, theta_i_team[i][j])

            xlsRow = xlsRow + 1

#            print("congestion count: ", junction[0].congestion_count)
#            print()
            #print()

            # v_team_learn()3
    #            print("Qvalue is: ", qvalue_i_team)
    #            print()
    #        print("Theta is: ", theta_i_team)
    #        print()
    #            print()
    print("The no of collisions are:", len(collided_cars))
    print("The no of red light infractions are:", len(infraction_cars))
    print("The no of direction violators are:", len(direction_violators))

    wb.save('Q_learning_values.xls')
    track_gui.mainloop()

curr_car = np.empty((9, 4), dtype=object)
pre_car = np.empty((9, 4), dtype=object)

def i_team_learn(feature, action_i_team, qvalue_i_team, e_i_team, theta_i_team, cars):
    for n in range(9):
        for k in range(6):
            if feature[n][int(action_i_team[n])][k] == 1:
                e_i_team[n][k] = e_i_team[n][k] + 1

    for update in range(len(junction)):
        junction[update].congestion_finder()
        junction[update].wait_time_counter()

    penalty = np.zeros(9)


    for i in range(9):
        for j in range(4):
            if junction[i].valid_directions[j]==1:
                lane_index = tuple([lane_start_index[i][j * 2], lane_start_index[i][(j * 2) + 1]])
                if cells[lane_index[0]][lane_index[1]].occupied is False:
                    curr_car[i][j] = None
                else:
                    for car in cars:
                        if car.pos == [lane_index[0], lane_index[1]]:
                            curr_car[i][j] = car.Id
                if pre_car[i][j] is not None and curr_car[i][j] != pre_car[i][j]:
                    penalty[i] += 1

                pre_car[i][j] = curr_car[i][j]

        """
        if sum(junction[i].wait_time) == 0:
            penalty[i] += 1
        """


    print()

    """
    for k in range(9):
        penalty[k] = +(sum(junction[k].congestion_count) / 120)
        penalty[k] += (sum(junction[k].wait_time) / 200)
    """

    delta = np.zeros(9)
    for n in range(9):
        delta[n] = penalty[n] - qvalue_i_team[n][int(action_i_team[n])]

    feature = find_features(junction)
#    print("feature is: ", feature[0])
#    print()

    qvalue_i_team = np.zeros([9, 4])
    for n in range(9):
        for k in range(4):
            for m in range(6):
                qvalue_i_team[n][k] += (feature[n][k][m] * theta_i_team[n][m])




    for n in range(9):
        delta[n] = delta[n] + (igamma * max(qvalue_i_team[n]))

    for n in range(9):
        for k in range(6):
            theta_i_team[n][k] = theta_i_team[n][k] + (ialpha * delta[n] * e_i_team[n][k])

    return qvalue_i_team, theta_i_team, e_i_team


def base_road(cells):
    rgbValue = np.uint8([255, 255, 255])
    for row in range(h):
        for col in range(w):
            if (cells[row][col].type == "None"):
                rgbValue = [255, 255, 255]
            elif (cells[row][col].type == "start" or cells[row][col].type == "end"):
                rgbValue = [0, 115, 0]
            elif (cells[row][col].type == "road"):
                rgbValue = [0, 0, 0]
            elif (cells[row][col].type == "junction"):
                rgbValue = [255, 0, 0]

            rgbArray[row][col] = rgbValue


def car_tracker(cars, road_img, created_canvas, secs):
    # assign base road layout
    base_road(cells)
    rgbArray_car = rgbArray
    # access each object
    for car in cars:
        car_pos = tuple(car.pos)
        car_mod_time = int ((car.startTime *1000) % 256)
        #car_color = [abs(car_mod_time*25 - 50 *car.start_id)%256,
        #             abs(car_mod_time * car.start_id * 50) % 256,
        #             abs(car_mod_time * 100 + 50 * car.start_id) % 256]
        if (car.start_id == 0):
            car_color = [ 200, 100, 0]
        elif (car.start_id == 1):
            car_color = [ 125, 0, 125]
        elif (car.start_id == 2):
            car_color = [ 255, 255, 125]
        elif (car.start_id == 3):
            car_color = [ 120, 0, 255]

        rgbArray_car[car_pos[0], car_pos[1]] = car_color

    for junc in range(len(junction)):

        junc_var = junction[junc]  # junction object
        junc_color = [255, 0, 0]  # initialize to red color

        row_sel = int(junc / 3)
        col_sel = junc % 3

        junc_row_pair = [junc_pos_array[2 * row_sel], junc_pos_array[2 * row_sel] + 1]
        junc_col_pair = [junc_pos_array[2 * col_sel], junc_pos_array[2 * col_sel] + 1]

        for junc_points in range(4):  # loop over all junction points

            junc_row = int(junc_points / 2)
            if (junc_points == 0 or junc_points == 3):
                junc_col = 0
            else:
                junc_col = 1

            if (junc_var.valid_directions[junc_points] == 1):
                if (junc_var.lights[junc_points] == 'R'):  # red
                    junc_color = [255, 0, 0]
                elif (junc_var.lights[junc_points] == 'G'):  # green
                    junc_color = [50, 205, 50]
            else:
                junc_color = [255, 0, 0]

            rgbArray_car[junc_row_pair[junc_row], junc_col_pair[junc_col]] = junc_color

    rgbArray_scale = imresize(rgbArray_car, (74 * 10, 74 * 10), interp='nearest')
    img = Image.fromarray(rgbArray_scale, 'RGB')
    road_cars_img = ImageTk.PhotoImage(img)
    road_img.itemconfig(created_canvas, image=road_cars_img)
    road_img.update()
    time.sleep(secs)


# Base road
base_road(cells)
## Main function call
master(num_episodes=15, num_steps=150)

print("terminated")
# this stores the updated Q matrices to output file
with open('outfile1', 'wb') as fp:
    pickle.dump(Q, fp)
