# Multivariate adaptive regression splines (MARS) tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(earth)

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
mars_model <-  mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()) %>%
  set_engine("earth")

# set-up tuning grid ----
# check parameters
parameters(mars_model)


mars_params <- parameters(mars_model) %>% 
  update(num_terms = num_terms(range = c(1,10)))

# define tuning grid
mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Single Layer Neural Network (multilayer perceptron)")
mars_tune <- mars_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = mars_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
mars_tictoc <- tic.log(format = TRUE)

# Write out results & workflow
save(mars_tune, mars_workflow, mars_tictoc, file = "model_info/mars_tune.rda")



