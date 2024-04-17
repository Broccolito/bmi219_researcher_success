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


plot_data = data.frame(
  prediction = as.vector(predictions),
  observation = as.vector(valid[,2])
)

plt = ggplot(data = plot_data, aes(x = observation, y = prediction)) + 
  geom_point(fill = "gray", color = "black", shape = 21, size = 2) + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_smooth(formula = y~x, method = "lm", color = "black") + 
  xlab("Observed H-index since 2018") + 
  ylab("Predicted H-index since 2018") + 
  ggtitle(label = "Lasso Model Prediction") + 
  theme_pubr() + 
  theme(element_text(size = 20))
plt

ggsave(filename = "lasso_performance.png", plot = plt, dpi = 1200,
       width = 5, height = 5)

save(lasso_model, file = "lasso_model.rda")

relative_importance = data.frame(
  variable = l$variable,
  importance = l$relative_importance
)

fwrite(relative_importance, file = "relative_importance.csv")
