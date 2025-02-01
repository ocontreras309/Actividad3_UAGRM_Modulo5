# Paralelismo con autores a través de PyKKA

from itertools import cycle
import random
import time
import numpy as np
import pykka

from pykka.messages import ProxyCall

from util import MonitoringThread, show_chessboard

BOARD_SIZE = 8

class PopulationActor(pykka.ThreadingActor):
    def generate_population(self, n_individuals, n_genes):
        indices = np.arange(n_genes)
        population = np.zeros((n_individuals, n_genes), dtype=np.float64)
        for i in range(n_individuals):
            np.random.shuffle(indices)
            population[i] = indices.copy()
        return population

class FitnessActor(pykka.ThreadingActor):
    def fitness(self, individual):
        fitness_value = len(individual)
        for i in range(len(individual)):
            row1 = individual[i]
            for k in range(i + 1, len(individual)):
                row2 = individual[k]
                row_diff = abs(row1 - row2)
                col_diff = abs(i - k)
                if row_diff == col_diff or row1 == row2 or i == k:
                    fitness_value -= 1
                if fitness_value == 0:
                    return 0
        return fitness_value

class SelectionActor(pykka.ThreadingActor):
    def select_individuals(self, population, fitnesses, probability):
        max_index = int(probability * len(population))
        indices = np.argsort(-fitnesses)  # Descending order
        selected_indices = indices[:max_index]
        return population[selected_indices]

class CrossoverActor(pykka.ThreadingActor):
    def crossover(self, individual1, individual2):
        crossover_point = random.randint(1, len(individual1)-1)
        result = np.zeros_like(individual1)
        result[:crossover_point] = individual1[:crossover_point]
        result[crossover_point:] = individual2[crossover_point:]
        return result

class MutationActor(pykka.ThreadingActor):
    def mutate(self, individual, mutation_rate):
        result = individual.copy()
        for i in range(len(result)):
            if random.random() < mutation_rate:
                result[i] = random.randint(0, BOARD_SIZE - 1)
        return result



def train_ga(population_size=1000, iterations=1000):
    n_population_actors = 10
    n_fitness_actors = 10

    population_actors = [PopulationActor.start() for _ in range(n_population_actors)]
    fitness_actors = [FitnessActor.start() for _ in range(n_fitness_actors)]
    selection_actor = SelectionActor.start()
    crossover_actor = CrossoverActor.start()
    mutation_actor = MutationActor.start()

    population = []
    population_futures = []

    for idx, actor in enumerate(population_actors):
        future = actor.ask(ProxyCall(
                    attr_path=['generate_population'],
                    args=[population_size // n_population_actors, BOARD_SIZE],
                    kwargs={}
                ), block=False)
        population_futures.append((idx, future))

    for idx, future in population_futures:
        population.extend(future.get(timeout=10))

    population = np.array(population)

    fitness_history = np.zeros(iterations, dtype=np.float64)
    best_found = np.zeros(BOARD_SIZE, dtype=np.float64)

    try:
        for i in range(iterations):
            # Cálculo de la función de fitness para cada individuo
            fitnesses = np.zeros(len(population))
            futures = []
            
            # Calcular cada fitness
            for idx, (actor, ind) in enumerate(zip(cycle(fitness_actors), population)):
                future = actor.ask(ProxyCall(
                    attr_path=['fitness'],
                    args=[ind],
                    kwargs={}
                ), block=False)
                futures.append((idx, future))
            
            # Obtención de resultados
            for idx, future in futures:
                fitnesses[idx] = future.get(timeout=10)

            if len(fitnesses) == 0:
                continue

            mean_fitness = np.mean(fitnesses)
            fitness_history[i] = mean_fitness
            max_fitness = np.max(fitnesses)

            if max_fitness == BOARD_SIZE:
                best_found = population[np.argmax(fitnesses)].copy()
                break

            # Selección de individuos
            selection_future = selection_actor.ask(
                ProxyCall(
                    attr_path=['select_individuals'],
                    args=[population, fitnesses, 0.65],
                    kwargs={}
                ),
                block=False
            )
            
            selected_population = selection_future.get(timeout=10)
            
            if selected_population is None or len(selected_population) == 0:
                continue
                
            population = selected_population
            n_parents = len(population)
            n_pairs = n_parents // 2

            # Resultados del cruzamiento
            crossover_futures = []
            offspring = np.zeros((n_pairs, BOARD_SIZE), dtype=np.float64)
            
            for j in range(n_pairs):
                idx1, idx2 = random.randint(0, n_parents-1), random.randint(0, n_parents-1)
                future = crossover_actor.ask(
                    ProxyCall(
                        attr_path=['crossover'],
                        args=[population[idx1], population[idx2]],
                        kwargs={}
                    ),
                    block=False
                )
                crossover_futures.append((j, future))
            
            for j, future in crossover_futures:
                offspring[j] = future.get(timeout=10)

            population = np.vstack([population, offspring])

            # Mutación
            mutation_futures = []
            for j in range(len(population)):
                future = mutation_actor.ask(
                    ProxyCall(
                        attr_path=['mutate'],
                        args=[population[j].copy(), 0.02],
                        kwargs={}
                    ),
                    block=False
                )
                mutation_futures.append((j, future))
            
            for j, future in mutation_futures:
                population[j] = future.get(timeout=10)

    finally:
        for actor in fitness_actors:
            actor.stop()
        selection_actor.stop()
        crossover_actor.stop()
        mutation_actor.stop()

    return best_found, fitness_history

if __name__ == '__main__':
    start = time.time()
    t = MonitoringThread()
    t.start()
    solution, _ = train_ga(10000, 1000)
    print("Solution found:", solution)
    end = time.time()
    print('Total execution time:', end - start)

    t.stop()

    if len(solution) > 0:
        show_chessboard(solution)
