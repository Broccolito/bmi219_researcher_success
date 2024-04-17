library(dplyr)
library(data.table)
library(purrr)
library(h2o)
library(ggplot2)
library(ggpubr)

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

plot_data = data.frame(
  prediction = as.vector(predictions),
  observation = as.vector(test[,2])
)

plt = ggplot(data = plot_data, aes(x = observation, y = prediction)) + 
  geom_point(fill = "gray", color = "black", shape = 21, size = 2) + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_smooth(formula = y~x, method = "lm", color = "black") + 
  xlab("Observed H-index since 2018") + 
  ylab("Predicted H-index since 2018") + 
  ggtitle(label = "Random Forest Model Prediction") + 
  theme_pubr() + 
  theme(element_text(size = 20))
plt

ggsave(filename = "random_forest_performance.png", plot = plt, dpi = 1200,
       width = 5, height = 5)

save(rf_model, file = "rf_model.rda")
