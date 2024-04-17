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
test = splits[[2]]

predictors = names(d)[-c(1,2)]
response = "h_index_since2018"

rf_model = h2o.randomForest(x = predictors, y = response,
                        training_frame = train,
                        ntrees = 50,
                        max_depth = 20,
                        min_rows = 10,
                        seed = 492357816)

# Print model summary
print(rf_model)

# Make predictions on the test set
predictions = h2o.predict(rf_model, test)

# Print predictions
print(predictions)