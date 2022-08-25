import numpy as np
from pyhealgo import Ga

CHROMOSOME_SIZE = 32
POPULATION_SIZE = 100
GENE_LOW = 0
GENE_HIGH = 2  # high is exclusive
MUTATION_RATE = 0.00002
CROSSOVER_RATE = 0.5
NUMBER_OF_GENERATIONS = 100

WEIGHTS = np.random.randint(1, 15, size=CHROMOSOME_SIZE)
VALUES = np.random.randint(10, 750, size=CHROMOSOME_SIZE)
THRESHOLD = 0.75 * np.sum(WEIGHTS)


def fitness_function(solution_idx, solution):
    sum_value = np.sum(solution * VALUES)
    sum_weight = np.sum(solution * WEIGHTS)
    if sum_weight <= THRESHOLD:
        return sum_value
    else:
        return 0


def validate(solution):
    sum_value = np.sum(solution * VALUES)
    sum_weight = np.sum(solution * WEIGHTS)
    return sum_weight, sum_value


def on_generation_callback(ga_instance):
    print("Generation: ", ga_instance.current_generation)
    print("Best solution {0} Fitness {1}".format(ga_instance.best_solution, ga_instance.best_fitness))
    return 0


if __name__ == '__main__':
    res = Ga(
        chromosome_size=CHROMOSOME_SIZE,
        gene_low=GENE_LOW,
        gene_high=GENE_HIGH,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        fitness_func=fitness_function,
        number_of_generations=NUMBER_OF_GENERATIONS,
        crossover_type="2-point",
        selection_type="roulette-wheel-selection",
        stop_criteria_saturate = 5,
        on_generation_cbk=on_generation_callback
    )

    solution, fitness = res.run()
    weight, value = validate(solution)
    print("SOLUTION: {0}, WEIGHT:{1}, VALUE:{2}, FITNESS:{3}".format(solution, weight, value, fitness))

    res.visualize()
    pass
