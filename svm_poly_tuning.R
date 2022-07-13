# Support vector machine (polynomial) tuning ----

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
svm_poly_model <-  svm_poly(
  mode = "classification", 
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>% 
  set_engine("kernlab")

# set-up tuning grid ----
# check parameters
parameters(svm_poly_model)


svm_poly_params <- parameters(svm_poly_model) 

# define tuning grid
svm_poly_grid <- grid_regular(svm_poly_params, levels = 3)

# workflow ----
svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Support Vector Machine (polynomial)")
svm_poly_tune <- svm_poly_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = svm_poly_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
svm_poly_tictoc <- tic.log(format = TRUE)

# Write out results & workflow
save(svm_poly_tune, svm_poly_workflow, svm_poly_tictoc, file = "model_info/svm_poly_tune.rda")



