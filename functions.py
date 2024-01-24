def Nagel_Schreckenberg(L, N, v_max, p, t_max):
    import random
    import numpy as np

    # Initializing the first row. Cars are denoted with 0 and no car with -1.
    positions = N*[0] + (L-N)*[-1]
    random.shuffle(positions)
    positions = [positions]

    # To store the flow counts for each timestep
    flow_counts = np.zeros(t_max)  

    # For each timestep we update the car positions
    for t in range(t_max):
        previous, current = positions[-1], L * [-1]

        # Go through all the cells in the row
        for pos in range(L):
            if previous[pos] > -1: # If there is a car in this cell, we change its position and value
                d = 1
                vi = previous[pos]
                while previous[(pos + d) % L] < 0: # Check how many spaces ahead are free
                    d += 1
                vtemp = min(vi + 1, d - 1, v_max) # Accelerating
                if random.uniform(0,1) < p: # Braking
                    v = max(vtemp-1, 0)
                else:
                    v = vtemp
                current[(pos+v)%L] = v # Moving

                # Update flow count between neighboring cells
                if v > 0:
                    flow_counts[t] += 1

        positions.append(current)
    return positions, flow_counts


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