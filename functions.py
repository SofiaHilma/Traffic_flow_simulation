def Nagel_Schreckenberg(L, N, v_max, p, t_max, max_brake=1):
    import random
    import numpy as np

    # Initializing the first row. Cars are denoted with 0 and no car with -1.
    positions = N*[0] + (L-N)*[-1]
    random.shuffle(positions)
    positions = [positions]

    # To store the cluster count for each timestep
    cluster_count = np.zeros(t_max)

    # average velocity of all cars per time step
    average_velocity = 0
    total_velocity = 0
    

    # To store the flow count for a fixed cell (4) over all time t_max
    flow_single_cell = 0
    density = 0

    # For each timestep we update the car positions and values
    for t in range(t_max):
        previous, current = positions[-1], L * [-1]
        
        # Determine if there is a car in this cell
        if previous[4] > -1:
            density += 1

        one_time_total_velocity = 0

        # Go through all the cells in the row
        for pos in range(L):
            if previous[pos] > -1: # Check if there is a car in this cell
                d = 1              # Set the distance to the car ahead as 1 for now
                vi = previous[pos] # Store the velocity of the car in the previous timestep 
                while previous[(pos + d) % L] < 0: # Check how many spaces ahead are free
                    d += 1 
                vtemp = min(vi + 1, d - 1, v_max) # Accelerating
                if random.uniform(0,1) < p: # Random braking
                    v = max(vtemp-random.randint(1, max_brake), 0)
                else:
                    v = vtemp
                current[(pos+v)%L] = v # Updates the cars position

                # Update flow count between neighboring cells (only cells 4 and 5)
                if pos == 4:
                    if v > 0: 
                        flow_single_cell += 1 
                if pos == 3: 
                    if v > 1: 
                        flow_single_cell += 1 
                if pos == 2: 
                    if v > 2: 
                        flow_single_cell += 1 
                if pos == 1:
                    if v > 3: 
                        flow_single_cell += 1 
                if pos == 0: 
                    if v > 4: 
                        flow_single_cell += 1 

                # Update the cluster count
                if (current[(pos-1)%L] > -1) or (current[(pos+1)%L] > -1):
                    cluster_count[t] += 1

                # Update the car velocity count
                one_time_total_velocity += v

        positions.append(current)
        one_time_total_velocity = one_time_total_velocity/N
        total_velocity += one_time_total_velocity
    average_velocity = total_velocity/t_max
    return positions, cluster_count, density, flow_single_cell, average_velocity



def plot_simulation(simulation):
    import numpy as np
    import matplotlib.pyplot as plt
    timesteps, L = len(simulation), len(simulation[0])
    a = np.empty(shape=(timesteps, L), dtype=object)

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
    total_flow = sum(flow_counts)  # Sum the entire array
    time_averaged_flow = total_flow / t_max  # Calculate the average
    return time_averaged_flow

