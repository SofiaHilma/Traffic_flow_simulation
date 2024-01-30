import random
import numpy as np

class Nagel_Schreckenberg():
    """
    Need to add description
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