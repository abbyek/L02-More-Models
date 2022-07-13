# nearest neighbors tuning ----

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
nn_model <- nearest_neighbor(
  mode = "classification", 
  neighbors = tune()
) %>% 
  set_engine("kknn")

# set-up tuning grid ----
# check parameters
# parameters(nn_model)


nn_params <- parameters(nn_model) 


# define tuning grid
nn_grid <- grid_regular(nn_params, levels = 5)

# workflow ----
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Nearest Neighbor")
nn_tune <- nn_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = nn_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
nn_tictoc <- tic.log(format = TRUE)

# Write out results & workflow

save(nn_tune, nn_workflow, nn_tictoc, file = "model_info/nn_tune.rda")