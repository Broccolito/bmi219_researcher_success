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
               metrics_to_predict = d[[metrics_to_predict]])

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