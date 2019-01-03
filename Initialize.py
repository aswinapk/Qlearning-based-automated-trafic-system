import numpy as np


class Environment:
    def __init__(self, type, occupied):
        self.type = type
        self.occupied = occupied


rows = 74
cols = 74
cells = np.empty((rows, cols), dtype=Environment)


def intializeInfra(cells):
    roads = [3, 4, 36, 37, 69, 70]
    junction = [3, 36, 69]
    for i in roads:
        for j in range(5, 36):
            cells[i][j] = Environment("road", False)
            cells[j][i] = Environment("road", False)
        for j in range(38, 69):
            cells[i][j] = Environment("road", False)
            cells[j][i] = Environment("road", False)

    for i in junction:
        for j in junction:
            cells[i][j] = Environment("junction", False)
            cells[i][j + 1] = Environment("junction", False)
            cells[j + 1][i] = Environment("junction", False)
            cells[j + 1][i + 1] = Environment("junction", False)

    cells[1][69] = Environment("road", False)
    cells[1][70] = Environment("road", False)
    cells[2][69] = Environment("road", False)
    cells[2][70] = Environment("road", False)
    cells[3][1] = Environment("road", False)
    cells[3][2] = Environment("road", False)
    cells[4][1] = Environment("road", False)
    cells[4][2] = Environment("road", False)
    cells[71][3] = Environment("road", False)
    cells[71][4] = Environment("road", False)
    cells[72][3] = Environment("road", False)
    cells[72][4] = Environment("road", False)
    cells[69][71] = Environment("road", False)
    cells[69][72] = Environment("road", False)
    cells[70][71] = Environment("road", False)
    cells[70][72] = Environment("road", False)

    cells[3][0] = Environment("end", False)
    cells[0][70] = Environment("end", False)
    cells[70][73] = Environment("end", False)
    cells[73][3] = Environment("end", False)

    cells[4][0] = Environment("start", False)
    cells[0][69] = Environment("start", False)
    cells[69][73] = Environment("start", False)
    cells[73][4] = Environment("start", False)

    for i in range(rows):
        for j in range(cols):
            if cells[i][j] is None:
                cells[i][j] = Environment("None", False)



def printInfra(cells):
    for i in range(rows):
        for j in range(cols):
            print(cells[i][j].type, end=" || ")
        print()



