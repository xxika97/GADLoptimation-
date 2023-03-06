# GADLoptimation-
GADLoptimation python

import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

# Define the hyperparameter space
hyperparameter_space = {
    'learning_rate': np.linspace(0.001, 0.1, 10),
    'batch_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'num_neurons': [32, 64, 128]
}

# Define the fitness function to evaluate each solution
def fitness_function(solution):
    # Create a deep neural network with the given hyperparameters
    model = Sequential()
    for i in range(solution['num_layers']):
        model.add(Dense(solution['num_neurons'], activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=solution['learning_rate']),
                  metrics=['accuracy'])

    # Train the model on the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model.fit(x_train, y_train, batch_size=solution['batch_size'], epochs=5)

    # Evaluate the model on the test set and return the accuracy
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy

# Define the genetic algorithm parameters
population_size = 20
mutation_rate = 0.1
generations = 10

# Initialize the population with random solutions
population = []
for i in range(population_size):
    solution = {}
    for hyperparameter, values in hyperparameter_space.items():
        solution[hyperparameter] = random.choice(values)
    population.append(solution)

# Evolve the population for the specified number of generations
for generation in range(generations):
    print('Generation', generation+1)

    # Evaluate each solution in the population
    fitness_scores = []
    for solution in population:
        fitness_scores.append(fitness_function(solution))

    # Select the top solutions to become parents
    parents = []
    for i in range(int(population_size/2)):
        parent1 = population[fitness_scores.index(max(fitness_scores))]
        fitness_scores[fitness_scores.index(max(fitness_scores))] = -1
        parent2 = population[fitness_scores.index(max(fitness_scores))]
        fitness_scores[fitness_scores.index(max(fitness_scores))] = -1
        parents.append((parent1, parent2))

    # Create offspring by crossover and mutation
    offspring = []
    for parent1, parent2 in parents:
        child = {}
        for hyperparameter in hyperparameter_space.keys():
            if random.random() < mutation_rate:
                child[hyperparameter] = random.choice(hyperparameter_space[hyperparameter])
            else:
                child[hyperparameter] = random.choice([parent1[hyperparameter], parent2[hyperparameter]])
        offspring.append(child)

    # Replace the old population with the new offspring
    population = offspring

# Evaluate the final population and select the best solution
fitness_scores = []
for solution in population:
    fitness_scores.append(fitness_function(solution))
best_solution = population[fitness_scores.index(max(fitness_scores))]

print('Best solution:', best_solution)
