# boosted tree tuning ----

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
bt_model <-  boost_tree(
  mode = "classification", 
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>% 
  set_engine("xgboost", importance = "impurity")

# set-up tuning grid ----
# check parameters
parameters(bt_model)


bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(c(2, 9)), learn_rate = learn_rate(range = c(-5, -0.2)))


# define tuning grid
bt_grid <- grid_regular(bt_params, levels = 5)

# workflow ----
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Boosted Tree")
bt_tune <- bt_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = bt_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
bt_tictoc <- tic.log(format = TRUE)

# Write out results & workflow
save(bt_tune, bt_workflow, bt_tictoc, file = "model_info/bt_tune.rda")



