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