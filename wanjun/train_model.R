library(tensorflow)
library(keras)
library(tfdatasets)
library(coro)          # For asynchronous programming
library(data.table)    # For fast data manipulation
library(dplyr)         # For data manipulation using a grammar of data manipulation
library(purrr)         # For functional programming tools
library(digest)        # For generating hash digests of data

# Set initial parameters
seed = 492357816  # Seed for reproducibility
train_ratio = 0.9  # Proportion of data to use for training
validation_ratio = 0.2  # Proportion of training data to use for validation
test_ratio = 1 - train_ratio  # Proportion of data to use for testing
batch_size = 32  # Number of samples per batch of computation
metrics_to_predict = "h_index_since2018"  # Target metric for prediction

# Load and preprocess dataset
d = fread("data/gscholar_profiles.csv")  # Read data from a CSV file

# Create a new dataframe selecting and renaming columns, and scaling the target metric
d = data.frame(predictor = d$publication_titles,
               metrics_to_predict = d[[metrics_to_predict]]) %>%
  mutate(metrics_to_predict = metrics_to_predict/100)

# Calculate dataset size, training, and test set sizes
d_count = dim(d)[1]
n_train = round(d_count * train_ratio, 0)
n_test = d_count - n_train

# Randomly split the dataset into training and test sets
set.seed(seed)
d_train = d[sample(1:d_count, size = n_train, replace = FALSE),]
d_test = d[sample(1:d_count, size = n_test, replace = FALSE),]

# Function to prepare directory structure for storing text data in training and test directories
make_temp = function(d_train, d_test){
  
  if(!dir.exists("temp")){
    dir.create("temp")
  }else{
    stop("Need to remove the temp folder...\n")
  }
  
  if(!dir.exists(file.path("temp", "train"))){
    dir.create(file.path("temp", "train"))
  }else{
    stop("Need to remove the temp folder...\n")
  }
  
  if(!dir.exists(file.path("temp", "test"))){
    dir.create(file.path("temp", "test"))
  }else{
    stop("Need to remove the temp folder...\n")
  }
  
  d_train %>% 
    split(.$metrics_to_predict) %>%
    map(function(x){
      metrics_to_predict = x$metrics_to_predict[1]
      if(!dir.exists(file.path("temp", "train", as.character(metrics_to_predict)))){
        dir.create(file.path("temp", "train", as.character(metrics_to_predict))) 
      }
      predictor = as.list(x$predictor)
      
      map(predictor, function(y){
        writeLines(y, file.path("temp", "train",
                                as.character(metrics_to_predict), 
                                paste0(digest(y, algo = "xxh3_128"), ".txt")))
      })
    })
  
  d_test %>% 
    split(.$metrics_to_predict) %>%
    map(function(x){
      metrics_to_predict = x$metrics_to_predict[1]
      if(!dir.exists(file.path("temp", "test", as.character(metrics_to_predict)))){
        dir.create(file.path("temp", "test", as.character(metrics_to_predict))) 
      }
      predictor = as.list(x$predictor)
      
      map(predictor, function(y){
        writeLines(y, file.path("temp", "test",
                                as.character(metrics_to_predict), 
                                paste0(digest(y, algo = "xxh3_128"), ".txt")))
      })
    })
  
  return(NULL)
  
}

# Prepare directories and write publication titles into separate text files, organized by the metric to predict
make_temp(d_train, d_test)

# Create TensorFlow datasets for training, validation, and testing
# These datasets are created from the directory structure prepared previously
raw_train_ds = text_dataset_from_directory(
  "temp/train",
  batch_size = batch_size,
  validation_split = validation_ratio,
  subset = 'training',
  seed = seed
)

# Batch, prefetch, and cache the datasets for efficient loading
batch = raw_train_ds %>%
  reticulate::as_iterator() %>%
  coro::collect(n = 1)

raw_val_ds = text_dataset_from_directory(
  'temp/train',
  batch_size = batch_size,
  validation_split = 0.2,
  subset = 'validation',
  seed = seed
)

raw_test_ds = text_dataset_from_directory(
  'temp/test',
  batch_size = batch_size
)

unlink("temp", recursive = TRUE)

# Text Preprocessing
re = reticulate::import("re")

punctuation = c("!", "\\", "\"", "#", "$", "%", "&", "'", "(", ")", "*",
                "+", ",", "-", ".", "/", ":", "<", "=", ">", "?", "@", "[",
                "\\", "\\", "]", "^", "_", "`", "{", "|", "}", "~")

punctuation_group = punctuation %>%
  sapply(re$escape) %>%
  paste0(collapse = "") %>%
  sprintf("[%s]", .)

# Define a custom text standardization function to preprocess text data
custom_standardization = function(input_data){
  lowercase = tf$strings$lower(input_data)
  stripped_html = tf$strings$regex_replace(lowercase, '<br />', ' ')
  tf$strings$regex_replace(
    stripped_html,
    punctuation_group,
    ""
  )
}

max_features = 10000
sequence_length = 250

# Create a TextVectorization layer to vectorize publication titles
vectorize_layer = layer_text_vectorization(
  standardize = custom_standardization,
  max_tokens = max_features,
  output_mode = "int",
  output_sequence_length = sequence_length
)

train_text = raw_train_ds %>%
  dataset_map(function(text, label){
    text
  })

# Adapt the vectorization layer to the training text data
vectorize_layer %>% adapt(train_text)

vectorize_text = function(text, label){
  text = tf$expand_dims(text, -1L)
  list(vectorize_layer(text), label)
}

batch <- reticulate::as_iterator(raw_train_ds) %>%
  reticulate::iter_next()
first_review <- as.array(batch[[1]][1])
first_label <- as.array(batch[[2]][1])
cat("Publication Titles:\n", first_review)


train_ds = raw_train_ds %>% dataset_map(vectorize_text)
val_ds = raw_val_ds %>% dataset_map(vectorize_text)
test_ds = raw_test_ds %>% dataset_map(vectorize_text)


AUTOTUNE = tf$data$AUTOTUNE

train_ds = train_ds %>%
  dataset_cache() %>%
  dataset_prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds %>%
  dataset_cache() %>%
  dataset_prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds %>%
  dataset_cache() %>%
  dataset_prefetch(buffer_size = AUTOTUNE)

# Define the model architecture using sequential API
embedding_dim = 16
model = keras_model_sequential() %>%
  layer_embedding(max_features + 1, embedding_dim) %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(64, activation = 'relu') %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(0.2) %>%
  layer_dense(1)

summary(model)

# Compile the model specifying loss function, optimizer, and metrics
model %>% compile(
  loss = "mean_absolute_error",
  optimizer = "adam",
  metrics = metric_mean_absolute_error()
)

# Train the model on the prepared datasets
epochs = 100
history = model %>%
  fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
  )

# Define and compile the export model for predictions
export_model = keras_model_sequential() %>%
  vectorize_layer() %>%
  model() %>%
  layer_activation(activation = "sigmoid")

# Evaluate the model on the test dataset
export_model %>% compile(
  loss = loss_mean_absolute_error(),
  optimizer = "adam",
  metrics = 'accuracy'
)

export_model %>% evaluate(raw_test_ds)

# Make predictions with the export model
examples = c("accuracy")
predict(export_model, examples)
