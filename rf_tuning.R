# random forest tuning ----

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
rf_model <-  rand_forest(
  mode = "classification", 
  mtry = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger", importance = "impurity")

# set-up tuning grid ----
# check parameters
parameters(rf_model)


rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 9)))


# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Random Forest")
rf_tune <- rf_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = rf_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
rf_tictoc <- tic.log(format = TRUE)

# Write out results & workflow

save(rf_tune, rf_workflow, rf_tictoc, file = "model_info/rf_tune.rda")