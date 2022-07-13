# Support vector machine (radial basis function) tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(kernlab)

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
svm_rbf_model <-  svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()
) %>% 
  set_engine("kernlab")

# set-up tuning grid ----
# check parameters
parameters(svm_rbf_model)


svm_rbf_params <- parameters(svm_rbf_model) 

# define tuning grid
svm_rbf_grid <- grid_regular(svm_rbf_params, levels = 3)

# workflow ----
svm_rbf_workflow <- workflow() %>% 
  add_model(svm_rbf_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Support Vector Machine (radial basis function)")
svm_rbf_tune <- svm_rbf_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = svm_rbf_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
svm_rbf_tictoc <- tic.log(format = TRUE)

# Write out results & workflow
save(svm_rbf_tune, svm_rbf_workflow, svm_rbf_tictoc, file = "model_info/svm_rbf_tune.rda")



