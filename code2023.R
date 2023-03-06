library(keras)
library(GA)
install.packages(GA)
library(MASS)

# Define the hyperparameter space
hyperparameter_space <- list(
  learning_rate = seq(0.001, 0.1, length.out = 10),
  batch_size = c(32, 64, 128),
  num_layers = c(1, 2, 3),
  num_neurons = c(32, 64, 128)
)

# Define the fitness function to evaluate each solution
fitness_function <- function(solution) {
  # Create a deep neural network with the given hyperparameters
  model <- keras_model_sequential()
  for (i in 1:solution$num_layers) {
    model %>% layer_dense(units = solution$num_neurons, activation = "relu")
  }
  model %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = solution$learning_rate),
    metrics = "accuracy"
  )
  
  # Train the model on the MNIST dataset
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  y_train <- mnist$train$y
  x_test <- mnist$test$x
  y_test <- mnist$test$y
  x_train <- array_reshape(x_train, c(nrow(x_train), 784)) / 255
  x_test <- array_reshape(x_test, c(nrow(x_test), 784)) / 255
  y_train <- to_categorical(y_train, 10)
  y_test <- to_categorical(y_test, 10)
  model %>% fit(
    x_train, y_train,
    batch_size = solution$batch_size,
    epochs = 5,
    verbose = 0
  )
  
  # Evaluate the model on the test set and return the accuracy
  result <- model %>% evaluate(x_test, y_test, verbose = 0)
  accuracy <- result[[2]]
  return(accuracy)
}

# Define the genetic algorithm parameters
population_size <- 20
mutation_rate <- 0.1
generations <- 10

# Initialize the population with random solutions
population <- NULL
for (i in 1:population_size) {
  solution <- list()
  for (hyperparameter in names(hyperparameter_space)) {
    solution[[hyperparameter]] <- sample(hyperparameter_space[[hyperparameter]], 1)
  }
  population <- c(population, solution)
}

# Define the GA control parameters
ga_control <- gaControl(
  fitness = fitness_function,
  popSize = population_size,
  maxiter = generations,
  pcrossover = 0.8,
  pmutation = mutation_rate,
  elitism = TRUE
)

# Evolve the population for the specified number of generations
ga_results <- ga(type = "real-valued", 
                 lower = rep(1, length(hyperparameter_space)),
                 upper = rep(length(hyperparameter_space), length(hyperparameter_space)),
                 pop = population,
                 gaControl = ga_control)

# Evaluate the final population and select the best solution
best_solution <- ga_results$population[ga_results$which.min]

print(paste("Best solution:", best_solution))

