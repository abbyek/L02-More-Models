## L02 More Models ---------------------------------------------------------

# Load package(s)
library(tidymodels)
library(tidyverse)
library(splitstackshape)
# Seed
set.seed(3013)

## load data ----
wildfires_dat <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-burned)




## inspect dataset ----

# overall
naniar::miss_var_summary(wildfires_dat)
skimr::skim_without_charts(wildfires_dat)

# check target for imbalance
wildfires_dat %>% 
  count(wlf) %>% 
  mutate(
    pct = 100 * n / sum(n)
  )

ggplot(wildfires_dat, aes(wlf)) +
  geom_bar()




## splitting data ----
# initial split
wildfires_split <- initial_split(wildfires_dat, prop = 0.8, strata = wlf)

wildfires_train <- wildfires_split %>% training()
wildfires_test <- wildfires_split %>% testing()

# cross validation
wildfire_folds <- vfold_cv(wildfires_train, v = 5, repeats = 3, strata = wlf)



## write out objects ----
save(wildfires_split, file = "data/wildfires_split.rda")
save(wildfires_train, file = "data/wildfires_train.rda")
save(wildfires_test, file = "data/wildfires_test.rda")
save(wildfire_folds, file = "data/wildfires_folds.rda")




## Feature engineering ----
# set up base recipe
wildfires_recipe <- recipe(wlf ~ ., wildfires_train) 

save(wildfires_recipe, file = "data/wildfires_recipe.rda")







## Compare models ----
load(file = "model_info/en_tune.rda")
load(file = "model_info/nn_tune.rda")
load(file = "model_info/rf_tune.rda")
load(file = "model_info/bt_tune.rda")
load(file = "model_info/svm_poly_tune.rda")
load(file = "model_info/svm_rbf_tune.rda")
load(file = "model_info/mars_tune.rda")
load(file = "model_info/slnn_mlp_tune.rda")


# create accuracy table
results <- tibble(
  model_type = c("Elastic Net", "Nearest Neighbor", "Random Forest", 
                 "Boosted Tree", "Support Vector Machine (polynomial)",
                 "Support Vector Machine (radial basis function)",
                 "Multivariate Adaptive Regression Splines (MARS)",
                 "Single Layer Neural Network (multilayer perceptron)"),
  tune_info = list(en_tune, nn_tune, rf_tune, bt_tune, svm_poly_tune, svm_rbf_tune, mars_tune, slnn_mlp_tune),
  assessment_info = map(tune_info, collect_metrics),
  best_model = map(tune_info, ~ select_best(.x, metric = "accuracy"))
)
results <- results %>%
  select(model_type, assessment_info) %>%
  unnest(assessment_info) %>%
  group_by(model_type) %>%
  summarise(accuracy = max(mean)) %>% 
  arrange(desc(accuracy))


# create run time table
run_times <- tibble(
  model_type = c("Elastic Net", "Nearest Neighbor", "Random Forest", 
                 "Boosted Tree", "Support Vector Machine (polynomial)",
                 "Support Vector Machine (radial basis function)",
                 "Multivariate Adaptive Regression Splines (MARS)",
                 "Single Layer Neural Network (multilayer perceptron)"),
  tune_info = list(en_tictoc, nn_tictoc, rf_tictoc, bt_tictoc, svm_poly_tictoc, 
                   svm_rbf_tictoc, mars_tictoc, slnn_mlp_tictoc)
) %>%
  select(model_type, tune_info) %>%
  unnest(tune_info) %>%
  unnest(tune_info) %>% 
  cSplit("tune_info", ":") %>% 
  select(model_type, tune_info_2) %>% 
  rename(tune_info = tune_info_2) 

# join tables together
model_summary <- left_join(run_times, results, by = "model_type") %>% 
  rename(run_time = tune_info) %>% 
  filter(run_time != 	"114.141 sec elapsed") %>% 
  arrange(desc(accuracy))

save(model_summary, file = "model_info/model_summary")

  
  
  


## Fit to testing data ----
# fit best model to training set
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tune, metric = "accuracy"))
en_results <- fit(en_workflow_tuned, wildfires_train)

# overall accuracy score
overall <- en_results %>% 
  predict(new_data = wildfires_test) %>% 
  bind_cols(wildfires_test %>% select(wlf)) %>% 
  accuracy(wlf, .pred_class)%>% 
  mutate(accuracy_type = "Overall") %>% 
  select(accuracy_type, .metric, .estimate)
# accuracy when there actually is a wildfire
yes_wlf <- en_results %>% 
  predict(new_data = wildfires_test) %>% 
  bind_cols(wildfires_test %>% select(wlf)) %>% 
  filter(wlf == "yes") %>% 
  accuracy(wlf, .pred_class) %>% 
  mutate(accuracy_type = "Wildfire Reached Area") %>% 
  select(accuracy_type, .metric, .estimate)
# accuracy when there actually is not a wildfire
no_wlf <- en_results %>% 
  predict(new_data = wildfires_test) %>% 
  bind_cols(wildfires_test %>% select(wlf)) %>% 
  filter(wlf == "no") %>% 
  accuracy(wlf, .pred_class) %>% 
  mutate(accuracy_type = "Wildfire Did Not Reach Area") %>% 
  select(accuracy_type, .metric, .estimate)
# create table of the 3 accuracy measurements
accuracy_table <- bind_rows(overall, yes_wlf, no_wlf)

save(accuracy_table, file = "model_info/accuracy_table.rda")
