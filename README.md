# Qlearning-based-automated-trafic-system
Automated traffic/cars that find optimal path based on the destination and congestion data working in hand with automated traffic signals that also learns from traffic congestion. Both cars and traffic lights learn separately and work together to increase throughput.  

INTELLIGENT TRAFFIC SYSTEM

Problem Formulation:

vehicle aim is to navigate the cars and train them to take the best possible path from its start location to reach its destination along with meeting all the constraint requirements in order to achieve maximum throughput.
The problem statement can be viewed in terms of the following subtasks
1)	Injection of Cars :- The amount of cars injected from the 4 start positions at each time step so that the system will be able to handle it and get maximum throughput.
2)	Driving the Car :- The Car must be able to recognize the roads and the junction and decide on its next position. The Car also should be able to take valid actions at junction and set right direction
3)	Learning :- Along with taking the action the car must also be able to learn from the actions it takes at different lanes. It should be able to use the congestion information so that it will take both the shortest and less congested path to reach the destination
4)	Meeting the Constraints :- We should ensure that there should be no more than one car in a particular cell at any given time (No collision). The Car must stop if red light is present at the junction. The Car must not go in wrong direction.

Reinforcement Learning Structure
Environment:
The entire environment is modelled into a 74x74 matrix which as shown in the diagram below. Each cell in the matrix is an object of data type Class Environment which has 2 attributes. First one is “Type” that indicates weather it is a road, junction or terminal points. And the second “occupied” indicate if that particular cell is occupied by a car or not. This setup is used as Environment for the Cars. At every time step each car object will update the “occupied” attribute of environment cells based on its position.

 
Agent : 
The Class Car has the following attributes.
“Id” - A unique number having the start position and the time at which the Car started
“dir” - Indicates the current direction in which the Car is moving
“destination” – Stores the destination the car is headed to
“pos” – Indicates the current position of the Car 
The cars are injected into the environment by creating objects of Car class. Each cars acts as agents which makes decisions based on Qlearning and takes actions.
Every time an object of class Car is created it will have its position (pos), direction (dir), start position (start_pos), destination, Car Id (Id), as its variables which can be accessed by both infrastucture and the vehicle in order to know the details of the car.
States:
Each lane between 2 junctions is considered as a state. The entire system has 32 states which are numbered as shown in the figure. The agent updates its current state based on its real time position in the environment.
Actions:
There are 4 actions the car can take at each state namely stay, move_forward, turn_left and turn_right, which are mapped to integer values 0,1,2,3 respectively for computation purpose. The functions go_forward(), turn_left(), turn_right() updates the position of the car and the environment cells based on its current position, action it is taking and the direction it is moving in.  When the car is in a particular lane it moves forward if the cell in front of it is not occupied which is verified using the function is_frontclear(). The car understands it is in junction using the function is_injunction() is_junction(). At the junction it takes any of the possible actions based on the Q values. 
Q values:
There are 4 sets of Q value matrices one for each of the Cars starting from a particular start_position. Each Q matrix has Q values for all possible state action pair that is 32*4 = 128 Q(s,a) pairs. For invalid state,action pair the Q value was initialized to a high negative value for easy convergence. The function choose_action_epsilongreedy() gives the action the car has to take.At the junction the car either explores with probability of Epsilon or exploits by taking the best possible action based on the greedy method of selecting action with maximum Qvalue. valid_action() function called inside choose_action_epsilongreedy() during exploration makes sure that only valid actions are being chosen from during exploration by returning set of valid actions based or current state of the agent.The Q value is updated using the function QLearning() which is called after the Car takes its action.
Example structure: Q = [Q1,Q2,Q3,Q4] , where Q1 will be the Q matrix used by cars starting from start position1 and respectively other Q matrices for other start positions.

Rewards:
When the car reaches its correct destination a reward of +10 is given. If the car reaches the wrong destination a negative reward of -10 is given. And if the car stays in the same position for more than one time instant waiting for the green signal a negative reward of -0.1 is given which reduces the Q value for that particular state so that the Car learns to take alternate routes which has less congestion or greater Q value. The function get_next_state_and_reward() gives the next state and immediate reward based on current state and action taken.

Function Drive ( )

Drive is the function which is called for each car object at each time step. The function implements the following 
1)	If the Car has reached the destination it returns with indicator set.
2)	If the Car has not reached the destination then an action is chosen based on epsilon_greedy method
3)	If it is just before the junction slot the car fetches the green light for the particular lane it is in and enters the junction only if the traffic light is green
4)	Car takes the action and learns from it. Car position is updated. Environment is updated
5)	Else if Car is not in the junction, it will navigate on the straight lane by checking if the position in front of it is not occupied. Car position and  Environment is updated

Main function – drive() used to navigate the car based on its position

Master Code 

1)	GUI image is created 
2)	Injection of Cars 
3)	Infrastucture module will update the junction lights
4) If the car has reached the destination throughput value is incremented. Else car.drive() is called for all the car objects present in the array of cars. With this all the cars navigate and update the environment and also learns 
5) Infrastucture module will learn based on cars new positions

Data Exchange
Car information from vehicle to infrastructure
An array of Class car objects is passed on to both the Infrastucture in the master code. Every time a new car is injected into the system the array is appended with this new car. Once the Car reaches its destination, that particular car/agent is deleted by calling the class destructor. Therefore, in general, the array consists of all the cars present in the system at that particular time instant. The Infrastucture can use this information to get the position, car-id, direction and destination attributes of each Car.
In general all the information is exchanged via function call and variable call as code of both the modules are packaged into a single file and merged in master code.
In our code “cars” is the array of objects which is appended every time we inject new car

cars = np.array([], dtype=Car)
if cells[4][0].occupied == False:
    cars = np.append(cars, Car([4, 0]))
if cells[0][69].occupied == False:
    cars = np.append(cars, Car([0, 69]))
if cells[69][73].occupied == False:
    cars = np.append(cars, Car([69, 73]))
if cells[73][4].occupied == False:
    cars = np.append(cars, Car([73, 4]))
for car in cars:
    if not((car.pos == car.destination).all()):

        if (car.pos == np.array([0, 70])).all() or (car.pos == np.array([70, 73])).all() or (
                car.pos == np.array([73, 3])).all() or (car.pos == np.array([3, 0])).all():
            print("wrong desination",car.Id)
            del (car)
        else:

            cars_dash = np.append(cars_dash, car)
     else:
        #                    print(car.Id, " reached destination")
        throughput += 1
        print("throughput is", throughput)
        cells[car.lastpos[0]][car.lastpos[1]].occupied = False
        cells[car.pos[0]][car.pos[1]].occupied = False
        del (car)
The car objected is deleted if it reaches the destination. The same can be observed in the else part of the code 


Collision Detection:
Collision detection is done by iterating through the cars array and verifying if any two cars has same value for "pos" attribute. In case two cars have same "pos" attribute that means they are in the same position and we report that as a collision.

Red Light violation:
Red light violation is verified based on the position value of car just before the red light. If the position attribute of car at junction just before red light has changed in subsequent time step then it has violated red light and that is reported.

Wrong direction/ U-turn detection: 
in case the "dir" attribute of any car changes to the opposite direction in the next time step then it is a direction violation/U-turn and this is reported.

Injection of Cars

Congestion based injection: The Infrastucture module has a congestion_count parameter associated with each junction which accounts for the real-time congestion at each of the junctions. This is being used to monitor the overall system congestion and regulate the injection rate. The threshold for congestion parameter was set by trial and error by tweaking the value a bit and the best result was found at 150. The injection function also checks if the injection point is clear of any cars and only injects if the start point is clear to be injected.
