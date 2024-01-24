def Nagel_Schreckenberg(L, N, v_max, p, t_max):
    import random
    positions = N*[0] + (L-N)*[-1]
    random.shuffle(positions)
    positions = [positions]
    for t in range(t_max):
        previous, current = positions[-1], L * [-1]
        for pos in range(L):
            if previous[pos] > -1:
                d = 1
                vi = previous[pos]
                while previous[(pos + d) % L] < 0: #Check how many spaces ahead are free
                    d += 1
                vtemp = min(vi + 1, d - 1, v_max) #Accelerating
                if random.uniform(0,1) < p: #Braking
                    v = max(vtemp-1, 0)
                else:
                    v = vtemp
                current[(pos+v)%L] = v #Moving
        positions.append(current)
    return positions

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