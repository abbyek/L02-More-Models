# elastic net tuning ----

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
  step_interact(wlf ~ .*.) %>% 
  step_zv(all_predictors())


# Define model ----
en_model <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

# set-up tuning grid ----
# check parameters
# parameters(en_model)

en_params <- parameters(en_model) 


# define tuning grid
en_grid <- grid_regular(en_params, levels = 10)

# workflow ----
en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
tic("Elastic Net")
en_tune <- en_workflow %>% 
  tune_grid(
    resample = wildfire_folds,
    grid = en_grid
  )
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
en_tictoc <- tic.log(format = TRUE)

# Write out results & workflow

save(en_tune, en_workflow, en_tictoc, file = "model_info/en_tune.rda")