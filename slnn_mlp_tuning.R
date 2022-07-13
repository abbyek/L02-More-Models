# Single Layer Neural Network (multilayer perceptron — mlp) tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load("data/wildfires_folds.rda")
load("data/wildfires_recipe.rda")
load("data/wildfires_split.rda")

# recipe adjustments
wildfires_recipe <- wildfires_recipe %>% 
  step_dummy(all_nominal(), -wlf) %>% 
  step_normalize(all_predictors()) %>% 
  step_zv(all_predictors())


# Define model ----
slnn_mlp_model <-  mlp(
  mode = "classification",
  hidden_units = tune(),
  penalty = tune()) %>%
  set_engine("nnet")

# set-up tuning grid ----
# check parameters
parameters(slnn_mlp_model)


slnn_mlp_params <- parameters(slnn_mlp_model) 

# define tuning grid
slnn_mlp_grid <- grid_regular(slnn_mlp_params, levels = 5)

# workflow ----
slnn_mlp_workflow <- workflow() %>% 
  add_model(slnn_mlp_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Single Layer Neural Network (multilayer perceptron)")
slnn_mlp_tune <- slnn_mlp_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = slnn_mlp_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
slnn_mlp_tictoc <- tic.log(format = TRUE)

# Write out results & workflow
save(slnn_mlp_tune, slnn_mlp_workflow, slnn_mlp_tictoc, file = "model_info/slnn_mlp_tune.rda")



