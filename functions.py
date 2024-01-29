def Nagel_Schreckenberg(L, N, v_max, p, t_max, max_brake = 1, max_acceleration = 1, seed = 2024):
    """
    asdfhjdsf
    """
    import random
    import numpy as np

    random.seed(seed)

    # Initialises the first row. Cars are denoted with 0 and empty spaces with -1.
    initial = N*[0] + (L-N)*[-1]
    random.shuffle(initial)
    positions = [initial]
    
    # Lists for cluster size and amount
    cells_in_clusters = np.zeros(t_max)
    cluster_count = np.zeros(t_max)

    # Variables for velocity statistics
    average_velocity = 0
    total_velocity = 0

    # Variables for measuring flow through cell 4
    flow_single_cell = 0
    density = 0

    # For each timestep we update the car positions and values
    for t in range(t_max):
        previous, current = positions[-1], L * [-1]
        
        # Determine if there is a car in this cell
        if previous[4] > -1:
            density += 1

        # For each timestep reset the velocity count to zero
        one_time_total_velocity = 0
        cluster_state = False # Set this to false for counting cells in clusters later

        for pos in range(L):
            if previous[pos] > -1: # Check if there is a car in this cell
                distance_ahead = 1
                v_prev = previous[pos]
                while previous[(pos + distance_ahead) % L] < 0: # Check how many spaces ahead are free
                    distance_ahead += 1 
                v_temp = min(v_prev + random.randint(1, max_acceleration), distance_ahead - 1, v_max) # Accelerating
                if random.uniform(0,1) < p: # Random braking
                    v = max(v_temp-random.randint(1, max_brake), 0)
                else:
                    v = v_temp
                current[(pos+v)%L] = v # Updates the cars position
                
                # Update the car velocity count
                one_time_total_velocity += v

            # Count the number of cells in a cluster
            if (current[pos-1] > -1) and (current[pos-2] > -1): # if two cells to the left are cars
                cells_in_clusters[t] += 1
                cluster_state = True
            if (current[pos-1] == -1) and (current[pos-2] > -1) and cluster_state:
                cells_in_clusters[t] += 1 # there's a delayed aspect in this way of counting, so we add an extra count
                cluster_state = False

            # Update the number of clusters: if i'm not a car and i have 2 cars to my left: +1
            if (current[pos] == -1) and (current[pos-1] > -1) and (current[pos-2] > -1):
                cluster_count[t] += 1

        # Update flow count between neighboring cells (only cells 4 and 5)
        for i in range(v_max):
            if current[(4-i)] > i:
                flow_single_cell += 1

        positions.append(current)

        # Get the average velocity for this time step and add it to the total velocity
        one_time_total_velocity = one_time_total_velocity/N 
        total_velocity += one_time_total_velocity

    # Average the total velocity over the number of timesteps
    average_velocity = total_velocity/t_max

    return positions, cells_in_clusters, density, flow_single_cell, average_velocity, cluster_count



def plot_simulation(simulation):
    '''
    Plots a grid of the velocities
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

