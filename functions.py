# def Nagel_Schreckenberg(L, N, v_max, p, t_max, max_brake = 1, max_acceleration = 1, seed = 2024):
#     """
#     asdfhjdsf
    
#     The function outputs: 
#         cells_in_clusters returns a list of the number of cells in a cluster at each timestep
#         cluster_count returns a list of the number of clusters at each timestep
#         average_velocity returns a single value for the average velocity over all time
#     """
#     import random
#     import numpy as np

#     random.seed(seed)

#     # Initialises the first row. Cars are denoted with 0 and empty spaces with -1.
#     initial = N*[0] + (L-N)*[-1]
#     random.shuffle(initial)
#     positions = [initial]
    
#     # Lists for cluster size and amount
#     cells_in_clusters = np.zeros(t_max)
#     cluster_count = np.zeros(t_max)

#     # Variables for velocity statistics
#     average_velocity = 0
#     total_velocity = 0

#     # Variables for measuring flow through cell 4
#     flow_single_cell = 0
#     density = 0

#     # For each timestep we update the car positions and values
#     for t in range(t_max):
#         previous, current = positions[-1], L * [-1]
        
#         # Determine if there is a car in this cell
#         if previous[4] > -1:
#             density += 1

#         # For each timestep reset the velocity count to zero
#         one_time_total_velocity = 0
#         cluster_state = False # Set this to false for counting cells in clusters later

#         for pos in range(L):
#             if previous[pos] > -1: # Check if there is a car in this cell
#                 distance_ahead = 1
#                 v_prev = previous[pos]
#                 while previous[(pos + distance_ahead) % L] < 0: # Check how many spaces ahead are free
#                     distance_ahead += 1 
#                 v_temp = min(v_prev + random.randint(1, max_acceleration), distance_ahead - 1, v_max) # Accelerating
#                 if random.uniform(0,1) < p: # Random braking
#                     v = max(v_temp-random.randint(1, max_brake), 0)
#                 else:
#                     v = v_temp
#                 current[(pos+v)%L] = v # Updates the cars position
                
#                 # Update the car velocity count
#                 one_time_total_velocity += v

#             # Count the number of cells in a cluster
#             if (current[pos-1] > -1) and (current[pos-2] > -1): # if two cells to the left are cars
#                 cells_in_clusters[t] += 1
#                 cluster_state = True
#             if (current[pos-1] == -1) and (current[pos-2] > -1) and cluster_state:
#                 cells_in_clusters[t] += 1 # there's a delayed aspect in this way of counting, so we add an extra count
#                 cluster_state = False

#             # Update the number of clusters: if i'm not a car and i have 2 cars to my left: +1
#             if (current[pos] == -1) and (current[pos-1] > -1) and (current[pos-2] > -1):
#                 cluster_count[t] += 1

#         # Update flow count between neighboring cells (only cells 4 and 5)
#         for i in range(v_max):
#             if current[(4-i)] > i:
#                 flow_single_cell += 1

#         positions.append(current)

#         # Get the average velocity for this time step and add it to the total velocity
#         one_time_total_velocity = one_time_total_velocity/N 
#         total_velocity += one_time_total_velocity

#     # Average the total velocity over the number of timesteps
#     average_velocity = total_velocity/t_max

#     return positions, cells_in_clusters, density, flow_single_cell, average_velocity, cluster_count





import random
import numpy as np

class Nagel_Schreckenberg():
    """
    The function outputs: 

    """
    def __init__(self, L, N, v_max, p, t_max, max_brake = 1, max_acceleration = 1, seed = 2024):
        
        self.L = L
        #Allows for density in percentages as well as calculated from absolute values
        if N < 1:
            self.N = int(N * L)
        else:
            self.N = N
        self.v_max = v_max
        self.p = p
        self.t_max = t_max
        self.max_brake = max_brake
        self.max_acceleration = max_acceleration
        self.seed = seed
        self.positions = []


    def run_model(self):
        random.seed(self.seed)

        # Initialises the first row. Cars are denoted with 0 and empty spaces with -1.
        initial = self.N*[0] + (self.L-self.N)*[-1]
        random.shuffle(initial)
        positions = [initial]

        # For each timestep we update the car positions and values
        for t in range(self.t_max):
            previous, current = positions[-1], self.L * [-1]

            for pos in range(self.L):
                if previous[pos] > -1: # Check if there is a car in this cell
                    distance_ahead = 1
                    v_prev = previous[pos]
                    while previous[(pos + distance_ahead) % self.L] < 0: # Check how many spaces ahead are free
                        distance_ahead += 1 
                    v_temp = min(v_prev + random.randint(1, self.max_acceleration), distance_ahead - 1, self.v_max) # Accelerating
                    if random.uniform(0,1) < self.p: # Random braking
                        v = max(v_temp-random.randint(1, self.max_brake), 0)
                    else:
                        v = v_temp
                    current[(pos+v)%self.L] = v # Updates the cars position
                    
            positions.append(current)

        self.positions = positions
        
        return positions
    

    def flow(self):
        flow_single_cell = 0
        for t in range(self.t_max):
            for i in range(self.v_max):
                if self.positions[t][4-i] > i:
                    flow_single_cell += 1
                    
        return flow_single_cell


    def average_velocity(self):
        velocities_sum = 0
        for t in range(self.t_max):
            for i in range(self.L):
                if self.positions[t][i] > -1:
                    velocities_sum += self.positions[t][i]

        return velocities_sum/(self.N * self.t_max)
    

    def clusters(self):
        cells_in_clusters = np.zeros(self.t_max)
        cluster_count = np.zeros(self.t_max)
        for t in range(self.t_max):
            cluster_exists = False
            for i in range(self.L):
                # Count the number of cells in a cluster
                if (self.positions[t][i-1] > -1) and (self.positions[t][i-2] > -1): # if two cells to the left are cars
                    cells_in_clusters[t] += 1
                    cluster_exists = True
                if (self.positions[t][i-1] == -1) and (self.positions[t][i-2] > -1) and cluster_exists:
                    cells_in_clusters[t] += 1 # there's a delayed aspect in this way of counting, so we add an extra count
                    cluster_exists = False

                # Update the number of clusters: if i'm not a car and i have 2 cars to my left: +1
                if (self.positions[t][i] == -1) and (self.positions[t][i-1] > -1) and (self.positions[t][i-2] > -1):
                    cluster_count[t] += 1

        return cells_in_clusters, cluster_count


    def density(self):
        density = 0
        for t in range(self.t_max):
            # Determine if there is a car in this cell
            if self.positions[t][4] > -1:
                density += 1
        return density


def plot_simulation(simulation):
    '''
    Plots a grid of the car velocities
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    timesteps, L = len(simulation), len(simulation[0])
    a = np.empty(shape=(timesteps, L), dtype=object)

    # Make array of velocities and white spaces
    for i in range(L):
        for j in range(timesteps):
            a[j, i] = str(int(simulation[j][i])) if simulation[j][i] > -1 else ''

    fig, ax = plt.subplots(figsize = (L/7.5, timesteps/7.5))
    ax.set_xticks(np.arange(L))
    ax.set_yticks(np.arange(timesteps))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()

    for i in range(timesteps):
        for j in range(L):
            text = ax.text(j, i, a[i, j], ha="center", va="center")

    plt.xlabel('Position')
    plt.ylabel('Time Step')
    plt.title('Traffic Simulation')
    plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.show()

    
# Calculate the average flow over the whole timespan for a specific density
def calculate_time_averaged_flow(flow_counts, t_max):
    total_flow = sum(flow_counts)
    time_averaged_flow = total_flow / t_max
    return time_averaged_flow




# Two lanes function

def Two_Lane_CA(L, N, v_max, p, t_max, switch_trigger, min_front_gap, min_back_gap, max_brake=1, max_acceleration = 1):
    """ Here I count average velocity using the previous saved velocities, but over a long time it shouldn't make much difference
    should maybe try to coordinate the one lane and two lane so they count the same way
    """
    
    import random
    import numpy as np

    # Analysis counts
    cells_in_clusters = np.zeros(t_max)
    cluster_count = np.zeros(t_max)
    average_velocity = 0
    total_velocity = 0



    # Initializing the first two rows (left and right lane). Cars are denoted with 0 and no car with -1.
    left_lane = N*[0] + (L-N)*[-1]
    random.shuffle(left_lane)
    left_lane = [left_lane]

    right_lane = N*[0] + (L-N)*[-1]
    random.shuffle(right_lane)
    right_lane = [right_lane]


    # For each timestep we update the car positions and values
    for t in range(t_max):
        previous_left_road, current_left_road = left_lane[-1], L * [-1]
        previous_right_road, current_right_road = right_lane[-1], L * [-1]

        # For each timestep reset the velocity count to zero
        one_time_total_velocity = 0
        # Set this boolean to false for counting cells in clusters later
        left_cluster_state = False 
        right_cluster_state = False 

        # Go through all the cells in the two rows
        for pos in range(L):

            # Check the LEFT lane

            if previous_left_road[pos] > -1: # Check if there is a car in this cell
                d = 1              # Set the distance to the car ahead as 1 for now
                other_lane_front_distance = 0

                vi = previous_left_road[pos] # Store the velocity of the car in the previous timestep 
                one_time_total_velocity += vi # Add this velocity to get the average velocity
                
                while previous_left_road[(pos + d) % L] < 0: # Check how many spaces ahead are free
                    d += 1 
                while previous_right_road[(pos + other_lane_front_distance)%L]<0:
                    other_lane_front_distance += 1 # Count the distance to a car 'infront' on the other lane

                # If the car is too close to a car ahead, and if there is enough space on the other lane, we switch
                if switch_trigger > d and min_front_gap < other_lane_front_distance:
                    # The car only changes lane and doesn't advance. It keeps its previous speed
                    current_right_road[pos] = vi

                else: # Otherwise we didn't switch, but can advance in our own lane
                    vtemp = min(vi + random.randint(1,max_acceleration), d - 1, v_max) # Accelerating
                    if random.uniform(0,1) < p: # Random braking
                        v = max(vtemp-random.randint(1, max_brake), 0)
                    else:
                        v = vtemp
                    current_left_road[(pos+v)%L] = v # Updates the cars position

            # Count the number of cells in a cluster in the left lane
            if (current_left_road[pos-1] > -1) and (current_left_road[pos-2] > -1): # if two cells to the left are cars
                cells_in_clusters[t] += 1
                left_cluster_state = True
            if (current_left_road[pos-1] == -1) and (current_left_road[pos-2] > -1) and left_cluster_state:
                cells_in_clusters[t] += 1 # there's a delayed aspect in this way of counting, so we add an extra count
                left_cluster_state = False
            


            # Check the RIGHT lane
            
            if previous_right_road[pos] > -1: # Check if there is a car in this cell
                d = 1              # Set the distance to the car ahead as 1 for now
                other_lane_front_distance = 0

                vi = previous_right_road[pos] # Store the velocity of the car in the previous timestep 
                one_time_total_velocity += vi # Add this velocity to get the average velocity
                while previous_right_road[(pos + d) % L] < 0: # Check how many spaces ahead are free
                    d += 1 
                while previous_left_road[(pos + other_lane_front_distance)%L]<0:
                    other_lane_front_distance += 1 # Count the distance to a car 'infront' on the other lane

                # If the car is too close to a car ahead, and if there is enough space on the other lane, we switch
                if switch_trigger > d and min_front_gap < other_lane_front_distance:
                    # The car only changes lane and doesn't advance. It keeps its previous speed
                    current_left_road[pos] = vi

                else: # Otherwise we didn't switch, but can advance in our own lane
                    vtemp = min(vi + random.randint(1,max_acceleration), d - 1, v_max) # Accelerating
                    if random.uniform(0,1) < p: # Random braking
                        v = max(vtemp-random.randint(1, max_brake), 0)
                    else:
                        v = vtemp
                    current_right_road[(pos+v)%L] = v # Updates the cars position
            
            # Count the number of cells in a cluster in the right lane (but add to the same count as the left one)
            if (current_right_road[pos-1] > -1) and (current_right_road[pos-2] > -1): # if two cells to the left are cars
                cells_in_clusters[t] += 1
                right_cluster_state = True
            if (current_left_road[pos-1] == -1) and (current_left_road[pos-2] > -1) and right_cluster_state:
                cells_in_clusters[t] += 1 # there's a delayed aspect in this way of counting, so we add an extra count
                right_cluster_state = False

        left_lane.append(current_left_road)
        right_lane.append(current_right_road)

        # Get the average velocity for this time step and add it to the total velocity
        one_time_total_velocity = one_time_total_velocity/N 
        total_velocity += one_time_total_velocity

    # Average the total velocity over the number of timesteps
    average_velocity = total_velocity/t_max

    return left_lane, right_lane, average_velocity, cells_in_clusters





# Plotting the two lanes

def plot_simulation(simulation_left, simulation_right):
    import numpy as np
    import matplotlib.pyplot as plt

    # Set the timestep and the length of the road (although they should be the same!)
    timesteps_left, L_left = len(simulation_left), len(simulation_left[0])
    timesteps_right, L_right = len(simulation_right), len(simulation_right[0])
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6.5))  # Create subplots for left and right lanes


    # Plot the LEFT lane

    # a_left holds the data for the left lane
    a_left = np.empty(shape=(timesteps_left + 1, L_left + 1), dtype=object)
    for i in range(L_left): # Go through the length of the road
        for j in range(timesteps_left): # Go through the time steps
            # Set the value of this cell if it has a car
            a_left[j, i] = str(int(simulation_left[j][i])) if simulation_left[j][i] > -1 else ''
    # Create the grid and assign the car values
    for i in range(timesteps_left + 1):
        for j in range(L_left + 1):
            text = ax_left.text(j + 0.5, i + 0.5, a_left[i, j], ha="center", va="center", fontsize=7)


    # Plot the RIGHT lane
            
    a_right = np.empty(shape=(timesteps_right+ 1, L_right+ 1), dtype=object)

    for i in range(L_right):
        for j in range(timesteps_right):
            a_right[j, i] = str(int(simulation_right[j][i])) if simulation_right[j][i] > -1 else ''
    for i in range(timesteps_right + 1):
        for j in range(L_right + 1):
            text = ax_right.text(j + 0.5, i + 0.5, a_right[i, j], ha="center", va="center", fontsize=7)


    # Set the grid and make time go down instead of up    
    ax_left.set_xticks(np.arange(L_left + 1))
    ax_left.set_yticks(np.arange(timesteps_left+1))
    ax_left.invert_yaxis()
    
    ax_right.set_xticks(np.arange(L_right + 1))
    ax_right.set_yticks(np.arange(timesteps_right+1))
    ax_right.invert_yaxis()

    # Adjust the size of the timestep numbers on the y-axis
    ax_left.tick_params(axis='y', labelsize=7)
    

    # Remove the cell numbers on the x-axes and remove the timestep numbers on the right plot 
    ax_left.set_xticklabels([])
    ax_right.set_xticklabels([])
    ax_right.set_yticklabels([])


    # Apply grids and labels to both subplots
    ax_left.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_right.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    ax_left.set_xlabel('Left lane cells')
    ax_left.set_ylabel('Time steps')
    ax_right.set_xlabel('Right lane cells')
    
    plt.suptitle('Two lanes simulation')
    plt.show()


    


    

