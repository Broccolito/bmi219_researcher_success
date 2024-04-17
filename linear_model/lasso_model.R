library(dplyr)
library(data.table)
library(purrr)
library(h2o)

h2o.init()
d = fread("word_freq_hindex.csv") %>%
  as.data.frame() %>%
  as.h2o()

splits = h2o.splitFrame(d, ratios = 0.8, seed = 492357816)
train = splits[[1]]
valid = splits[[2]]

predictors = names(d)[-c(1,2)]
response = "h_index_since2018"

lasso_model = h2o.glm(x = predictors, y = response,
                      training_frame = train,
                      validation_frame = valid,
                      family = "gaussian",
                      lambda_search = TRUE,
                      alpha = 1)

# Summary of the model
summary(lasso_model)

# Make predictions
predictions = h2o.predict(lasso_model, valid)

# Print predictions
print(predictions)