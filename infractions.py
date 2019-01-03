import numpy as np
from lane_start_index import lane_start_index

# Variable for red-light violation detection
current_car = np.empty((9, 4), dtype=object)
prev_car = np.empty((9, 4), dtype=object)


# Direction Violators, this function determines the current direction of motion of the vehicle, and the allowed legal
# direction for the lane and determines if the vehicle is going in the wrong direction
def direction_violation(cars, cells):
    no_of_violation = 0
    direction_violators = {}
    no_of_u_turns = 0
    u_turns = {}

    for n in range(len(cars)):
        x = cars[n].pos[0]
        y = cars[n].pos[1]

        if cells[x][y].type != "junction":
            if x == 4 or x == 37 or x == 70:
                if cars[n].dir == "left":
                    direction_violators.update({cars[n].Id: (x, y)})
                    no_of_violation = no_of_violation + 1

                if cars[n].dir == "up":
                    u_turns.update({cars[n].Id: (x, y)})
                    no_of_u_turns = no_of_u_turns + 1

            if x == 3 or x == 36 or x == 69:
                if cars[n].dir == "right":
                    direction_violators.update({cars[n].Id: (x, y)})
                    no_of_violation = no_of_violation + 1

                if cars[n].dir == "down":
                    u_turns.update({cars[n].Id: (x, y)})
                    no_of_u_turns = no_of_u_turns + 1

            if y == 4 or y == 37 or y == 70:
                if cars[n].dir == "down":
                    direction_violators.update({cars[n].Id: (x, y)})
                    no_of_violation = no_of_violation + 1

                if cars[n].dir == "left":
                    u_turns.update({cars[n].Id: (x, y)})
                    no_of_u_turns = no_of_u_turns + 1

            if y == 3 or y == 36 or y == 69:
                if cars[n].dir == "up":
                    direction_violators.update({cars[n].Id: (x, y)})
                    no_of_violation = no_of_violation + 1

                if cars[n].dir == "right":
                    u_turns.update({cars[n].Id: (x, y)})
                    no_of_u_turns = no_of_u_turns + 1

    return direction_violators, u_turns


# Red light violation
# The function helps to find the vehicles violating red signal and adds the vehicles to the list of traffic violators
# On a red light at a junction, we store the car if any that is stopped at the junction, if on the next time step the
# the signal is still red and car has changes the position then we identify it as a violator
def red_light_infraction(junction, car_dictionary, cells):
    no_of_infractions = 0
    infraction_cars = {}
    for n in range(len(junction)):
        for i in range(4):
            if junction[n].lights[i] == 'G' and junction[n].valid_directions[i] == 1:
                prev_car[n][i] = None
            if junction[n].lights[i] == 'R' and junction[n].valid_directions[i] == 1:
                lane_index = tuple([lane_start_index[n][i * 2], lane_start_index[n][(i * 2) + 1]])
                if lane_index in car_dictionary:
                    current_car[n][i] = car_dictionary[lane_index]
                else:
                    current_car[n][i] = None

                if prev_car[n][i] is not None and current_car[n][i] != prev_car[n][i]:
                    infraction_cars.update({prev_car[n][i]: lane_index})
                    no_of_infractions = no_of_infractions + 1

                prev_car[n][i] = current_car[n][i]

        return infraction_cars


# Collision Detection
# Cars have collided if two or more cars are at the same location
def detect_collision(cars, junction, cells):
    car_dictionary = {}
    no_of_collisions = 0
    collided_cars = {}
    
    for i in range(len(cars)):
        position = tuple(cars[i].pos)
        Id = cars[i].Id
        if position in car_dictionary:
            collided_cars.update({car_dictionary[position]: position})
            #print(position)
            no_of_collisions = no_of_collisions + 1

        else:
            car_dictionary.update({position: Id})


    infraction_cars = red_light_infraction(junction, car_dictionary, cells)
    direction_violators = direction_violation(cars, cells)[0]
    u_turns = direction_violation(cars, cells)[1]

    return collided_cars, infraction_cars, direction_violators, u_turns