import numpy as np
import copy
import sys


### PSO
class Particle:
    def __init__(self, fitness, dim, minx, maxx, minv, maxv):
        
        self.position = np.random.uniform(minx, maxx)
            
        self.velocity = np.random.uniform(minv, maxv)
        
        while fitness(self.position) == sys.float_info.max:
            self.position = np.random.uniform(minx, maxx)
        
        self.fitness = fitness(self.position)
        
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness
    


def PSO(fitness, max_iter, n, dim, minx, maxx, minv, maxv, a1 = 1.49445, a2 = 1.49445):
    SWRM = []
    BP = []
    BSF = []
    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, minv, maxv) for _ in range(n)]    
    # compute best_pos & best_fit
    best_swarm_pos = np.zeros(dim)
    
    best_swarm_fitnessVal = sys.float_info.max
    
    for iteration in range(max_iter):
        for i in range(n):
            

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

        
            swarm[i].velocity = (
                (swarm[i].velocity) + 
                (a1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) + 
                (a2 * r2 * (best_swarm_pos - swarm[i].position))
            )
            
            swarm[i].velocity = np.clip(swarm[i].velocity, minv, maxv)
                
            swarm[i].position += swarm[i].velocity
            swarm[i].position = np.clip(swarm[i].position, minx, maxx)
            
            for k in range(dim):
                if swarm[i].position[k] < minx[k]:
                    swarm[i].position[k] = minx[k] + abs(swarm[i].position[k] - minx[k])
                    swarm[i].velocity[k] *= -1
                elif swarm[i].position[k] > maxx[k]:
                    swarm[i].position[k] = maxx[k] + abs(swarm[i].position[k] - maxx[k])
                    swarm[i].velocity[k] *= -1
            
            swarm[i].fitness = fitness(swarm[i].position)
            
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)
            
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)
        SWRM.append(np.array([p.position for p in swarm]))
        BP.append(best_swarm_pos)
        BSF.append(copy.copy(best_swarm_fitnessVal))
    
    return BSF, BP, SWRM