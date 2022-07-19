# First step: Create the first population set
import numpy as np
import random
from datetime import datetime
import math


# Parameters
# n_stations = 20

# n_population = 100

# mutation_rate = 0.3


# Generating a list of coordenades representing each city
# coordinates_list = [[x, y] for x, y in
# zip(np.random.randint(0, 100, n_stations), np.random.randint(0, 100, n_stations))]


# names_list = np.array(
# ['Berlin', 'London', 'Moscow', 'Barcelona', 'Rome', 'Paris', 'Vienna', 'Munich', 'Istanbul', 'Kyiv',
# 'Bucharest',
# 'Minsk', 'Warsaw', 'Budapest', 'Milan', 'Prague', 'Sofia', 'Birmingham', 'Brussels', 'Amsterdam'])

# stations_dict = {x: y for x, y in zip(names_list, coordinates_list)}

# Function to compute the distance between two points
def compute_city_distance_coordinates(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_city_distance_names(city_a, city_b, stations_dict):
    return compute_city_distance_coordinates(stations_dict[city_a], stations_dict[city_b])


def genesis(station_list, n_population, n_stations, nb_lines):
    line = []
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        for j in range(nb_lines):
            line.append(station_list[np.random.choice(list(range(n_stations)),
                                                      np.random.randint(math.ceil(n_stations * 0.75 / nb_lines),
                                                                        math.ceil(n_stations * 2 / nb_lines)),
                                                      replace=False)])
        population_set.append(line)
        line = []
    return np.array(population_set)


# population_set = genesis(names_list, n_population)


# 2. Evaluation of the fitness

# individual solution

def fitness_eval(lines_list, stations_dict):
    # print(line_list)
    total = 0
    for station_list in lines_list:
        for i in range(len(station_list) - 1):
            a = station_list[i]
            b = station_list[i + 1]
            total += compute_city_distance_names(a, b, stations_dict)
    # for station in stations_dict:
    # for line_stations in lines_list:
    # if station not in line_stations:
    # total += 10000
    line = {}
    for station in stations_dict:
        for i in range(len(lines_list)):
            if station in lines_list[i]:
                if station in line:
                    line[station].append(i)
                else:
                    line[station] = []
                    line[station].append(i)
    if len(line) != len(stations_dict):
        total += 10

    # boucle_list = []
    # p = list(line.keys())
    # for i in range(len(lines_list)):
    #     try:
    #         boucle_list.append(line[p[0]][i])
    #     except:
    #         pass
    #
    # print(line)

    return total


# All solutions
def get_all_fitnes(population_set, stations_dict, n_population):
    fitnes_list = np.zeros(n_population)

    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], stations_dict)

    return fitnes_list


# fitnes_list = get_all_fitnes(population_set, stations_dict)


# 3. Selecting the progenitors
def progenitor_selection(population_set, fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list / total_fit
    #print(prob_list)

    # Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)

    #print(progenitor_list_a)
    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]
    #print(progenitor_list_b)

    return np.array([progenitor_list_a, progenitor_list_b])


# progenitor_list = progenitor_selection(population_set, fitnes_list)


# Pairs crossover
def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:2]

    for line in prog_b:
        print([line])
        print(offspring)
        if not line in offspring:
            offspring = np.concatenate((offspring, [line]))

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
def mutate_offspring(offspring, n_stations, mutation_rate):
    for q in range(int(n_stations * mutation_rate)):
        a = np.random.randint(0, n_stations)
        b = np.random.randint(0, n_stations)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring


# New populaiton generation
def mutate_population(new_population_set, n_stations, mutation_rate):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring, n_stations, mutation_rate))
    return mutated_pop


# mutated_pop = mutate_population(new_population_set)


# Begin
def list_coordinates_and_names(list_stations, nb_lines):
    # Parameters

    n_stations = len(list_stations)

    n_population = 100

    mutation_rate = 0.3

    # stations_dict = list_stations
    names_list = []
    for i in list_stations:
        names_list.append(i[0])
    names_list = np.array(names_list)

    coordinates_list = []
    for i in list_stations:
        coordinates_list.append([i[1], i[2]])

    stations_dict = {x: y for x, y in zip(names_list, coordinates_list)}

    # Use algo

    population_set = genesis(names_list, n_population, n_stations, nb_lines)
    fitnes_list = get_all_fitnes(population_set, stations_dict, n_population)
    progenitor_list = progenitor_selection(population_set, fitnes_list)
    new_population_set = mate_population(progenitor_list)
    mutated_pop = mutate_population(new_population_set, n_stations, mutation_rate)

    # Everything put together
    best_solution = [-1, np.inf, np.array([])]
    for i in range(1000):
        if i % 100 == 0: print(i, fitnes_list.min(), fitnes_list.mean(), datetime.now().strftime("%d/%m/%y %H:%M"))
        fitnes_list = get_all_fitnes(mutated_pop, stations_dict, n_population)

        # Saving the best solution
        if fitnes_list.min() < best_solution[1]:
            best_solution[0] = i
            best_solution[1] = fitnes_list.min()
            best_solution[2] = np.array(mutated_pop)[fitnes_list.min() == fitnes_list]

        progenitor_list = progenitor_selection(population_set, fitnes_list)
        new_population_set = mate_population(progenitor_list)

        mutated_pop = mutate_population(new_population_set, n_stations, mutation_rate)

    print(best_solution)

    return best_solution[2][0]
