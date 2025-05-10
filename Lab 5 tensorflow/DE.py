import numpy as np

def differential_evolution(func, bounds, mut=0.8, crossp=0.7, popsize=10, max_iter=100):
    dimensions = len(bounds)
    # Ініціалізація популяції
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(max_b - min_b)
    pop_denorm = np.round(min_b + pop * diff)
    fitness = np.asarray([func(ind) for ind in pop_denorm])
    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    best_history_value = []
    best_history_position = []
    
    for i in range(max_iter):
        for j in range(popsize):
            # Вибір 3 різних векторів, не включаючи j
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            
            # Мутація
            mutant = np.clip(a + mut * (b - c), 0, 1)
            
            # Кросовер
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            
            # Декодування
            trial_denorm = min_b + trial * diff
            f = func(trial_denorm)
            
            # Відбір
            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        best_history_value.append(fitness[best_idx])
        best_history_position.append(best)
    return best_history_position, best_history_value 