# First step: Create the first population set
import numpy as np
import random
from datetime import datetime

# Parameters
# n_cities = 20

# n_population = 100

# mutation_rate = 0.3


# Generating a list of coordenades representing each city
#coordinates_list = [[x, y] for x, y in
                    #zip(np.random.randint(0, 100, n_cities), np.random.randint(0, 100, n_cities))]


# names_list = np.array(
# ['Berlin', 'London', 'Moscow', 'Barcelona', 'Rome', 'Paris', 'Vienna', 'Munich', 'Istanbul', 'Kyiv',
# 'Bucharest',
# 'Minsk', 'Warsaw', 'Budapest', 'Milan', 'Prague', 'Sofia', 'Birmingham', 'Brussels', 'Amsterdam'])

# cities_dict = {x: y for x, y in zip(names_list, coordinates_list)}

# Function to compute the distance between two points
def compute_city_distance_coordinates(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_city_distance_names(city_a, city_b, cities_dict):
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])


def genesis(city_list, n_population, n_cities):
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        sol_i = city_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)]
        population_set.append(sol_i)
    return np.array(population_set)


# population_set = genesis(names_list, n_population)


# 2. Evaluation of the fitness

# individual solution
def fitness_eval(city_list, cities_dict, n_cities):
    total = 0
    for i in range(n_cities - 1):
        a = city_list[i]
        b = city_list[i + 1]
        total += compute_city_distance_names(a, b, cities_dict)
    return total


# All solutions
def get_all_fitnes(population_set, cities_dict, n_population, n_cities):
    fitnes_list = np.zeros(n_population)

    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], cities_dict, n_cities)

    return fitnes_list


# fitnes_list = get_all_fitnes(population_set, cities_dict)


# 3. Selecting the progenitors
def progenitor_selection(population_set, fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list / total_fit

    # Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)

    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]

    return np.array([progenitor_list_a, progenitor_list_b])


# progenitor_list = progenitor_selection(population_set, fitnes_list)


# Pairs crossover
def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]


    for city in prog_b:

        if not city in offspring:
            print(offspring)
            print([city])
            offspring = np.concatenate((offspring, [city]))

    return offspring


# Finding pairs of mates
def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)

    return new_population_set


# new_population_set = mate_population(progenitor_list)


# Offspring production
def mutate_offspring(offspring, n_cities, mutation_rate):
    for q in range(int(n_cities * mutation_rate)):
        a = np.random.randint(0, n_cities)
        b = np.random.randint(0, n_cities)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring


# New populaiton generation
def mutate_population(new_population_set, n_cities, mutation_rate):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring, n_cities, mutation_rate))
    return mutated_pop


# mutated_pop = mutate_population(new_population_set)


# Begin
def list_coordinates_and_names(list_stations):
    # Parameters

    n_cities = len(list_stations)

    n_population = 100

    mutation_rate = 0.3

    # cities_dict = list_stations
    names_list = []
    for i in list_stations:
        names_list.append(i[0])
    names_list = np.array(names_list)

    coordinates_list = []
    for i in list_stations:
        coordinates_list.append([i[1], i[2]])

    cities_dict = {x: y for x, y in zip(names_list, coordinates_list)}

    # Use algo

    population_set = genesis(names_list, n_population, n_cities)
    fitnes_list = get_all_fitnes(population_set, cities_dict, n_population, n_cities)
    progenitor_list = progenitor_selection(population_set, fitnes_list)
    new_population_set = mate_population(progenitor_list)
    mutated_pop = mutate_population(new_population_set, n_cities, mutation_rate)

    # Everything put together
    best_solution = [-1, np.inf, np.array([])]
    for i in range(1000):
        if i % 100 == 0: print(i, fitnes_list.min(), fitnes_list.mean(), datetime.now().strftime("%d/%m/%y %H:%M"))
        fitnes_list = get_all_fitnes(mutated_pop, cities_dict, n_population, n_cities)

        # Saving the best solution
        if fitnes_list.min() < best_solution[1]:
            best_solution[0] = i
            best_solution[1] = fitnes_list.min()
            best_solution[2] = np.array(mutated_pop)[fitnes_list.min() == fitnes_list]

        progenitor_list = progenitor_selection(population_set, fitnes_list)
        new_population_set = mate_population(progenitor_list)

        mutated_pop = mutate_population(new_population_set, n_cities, mutation_rate)

    #print(best_solution)

    return best_solution[2][0]

