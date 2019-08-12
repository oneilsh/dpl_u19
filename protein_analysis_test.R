library(keras)
library(tensortree) # for global install: 
# devtools::install_github("oneilsh/tensortree", lib = "/usr/lib/R/site-library") # if root...
# on uniprot: search e.g. "P450 length:[450 TO 700]", then Download -> Preview first 10, grab url, change 10 to 2000...
# wget 'https://www.uniprot.org/uniprot/?query=P450%20length:[450%20TO%20700]&format=fasta&limit=2000&sort=score' -O datasets/proteins/p450s.fa

# read dataframe of metadata about the two files
seqids_df <- fastas_to_targets_df(c("p450s.fa", "fmos.fa"), directory = "datasets/proteins")
print(head(seqids_df))






# shuffle the dataset
shuffled_indices <- sample(1:nrow(seqids_df))
seqids_df <- seqids_df[shuffled_indices, ]

# a placeholder for results, 5 rows, three columns
metrics_df <- data.frame(loss = rep(NA, 5), accuracy = rep(NA, 5), fold = rep(NA, 5))

for(i in 0:4) {
  message("Training fold ", i)
  potential_indices <- 1:nrow(seqids_df)
  train_indices <- potential_indices[potential_indices %% 5 != i]
  metrics <- train_model(seqids_df, train_indices)
  metrics_df[i+1, ] <- c(metrics$loss, metrics$acc, i)
}


# split it into two dfs, one with 80% of rows, the other with the rest
#train_indices <- sample(1:nrow(seqids_df), size = nrow(seqids_df) * 0.8)

# a function which, given a dataframe with training and validation data, 
# and a vector of indices to use for training (others will be used for validation)
train_model <- function(seqids_df, train_indices) {
  seqids_df_train <- seqids_df[train_indices, ]
  seqids_df_validate <- seqids_df[-train_indices, ]
    
  # figure out the minimum length of all the sequences (both training and validation), we'll
  # trim all the sequences to this length to pack them into tensors
  min_length <- min(seqids_df$seq_len)   # minimum over all sequences; 450
  
  # we have two classes, so we'll use "binary" class_mode rather than "categorical"
  train_generator <- flow_sequences_from_fasta_df(seqids_df_train, 
                                                  targets_col = "class",
                                                  alphabet = "protein",
                                                  batch_size = 64,
                                                  class_mode = "binary",
                                                  trim_to = min_length)
  
  validate_generator <- flow_sequences_from_fasta_df(seqids_df_validate, 
                                                  targets_col = "class",
                                                  alphabet = "protein",
                                                  batch_size = 64,
                                                  class_mode = "binary",
                                                  trim_to = min_length)
  
  # let's see what we've got
  #batch <- train_generator()
  #batch[[1]] %>% tt() %>% print(bottom = "2d", end_n = 21)
  #batch[[2]] %>% tt() %>% print(bottom = "1d", end_n = 64)
  
  # we'll do a 1-d CNN, the analog for 2d CNNs for image data
  # note the activation of "sigmoid" on the final layer, which only has one node - good for predicting
  # a single number between 0 and 1 (to go along with a binary classification task)
  network <- keras_model_sequential() %>%
    layer_conv_1d(filters = 16, kernel_size = 7, activation = "relu", input_shape = c(450, 21)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 2) %>%  
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")   # 1 unit sigmoid -> good output for "binary" prediction
  
  # note the use of binary_crossentropy rather than categorical_crossentropy
  compile(network, 
          optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy"))
  
  # 3200 rows in training df, batch size 64; 3200/64 = 50
  # 800 in validation, batch size 64, 800 / 64 ~= 12 (10, 12, whateves)
  history <- fit_generator(network, 
                train_generator, 
                epochs = 5,
                steps_per_epoch = 20)
                #validation_data = validate_generator,
                #validation_steps = 10)
  
  
  # list with entries for $loss and $acc
  metrics <- evaluate_generator(network, generator = validate_generator, steps = 10)
  return(metrics)
}


# try an RNN
# note the activation of "sigmoid" on the final layer, which only has one node - good for predicting
# a single number between 0 and 1 (to go along with a binary classification task)
#network <- keras_model_sequential() %>%
#  layer_gru(input_shape = c(450, 21), units = 32, dropout = 0.3, return_sequences = TRUE) %>%
#  layer_batch_normalization() %>%
#  layer_flatten() %>%
#  layer_dropout(rate = 0.3) %>%
#  layer_batch_normalization() %>%
#  layer_dense(units = 64, activation = "relu") %>%
#  layer_dense(units = 1, activation = "sigmoid")   # 1 unit sigmoid -> good output for "binary" prediction

# note the use of binary_crossentropy rather than categorical_crossentropy
compile(network, 
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy"))

# 3200 rows in training df, batch size 64; 3200/64 = 50
# 800 in validation, batch size 64, 800 / 64 ~= 12 (10, 12, whateves)
history <- fit_generator(network, 
              train_generator, 
              epochs = 5,
              steps_per_epoch = 50,
              validation_data = validate_generator,
              validation_steps = 10)










## Same thing, except this time we want to predict the percent of the sequences that are
# non-polar letters


library(keras)
library(tensortree)

# a function that, given a single sequence, returns a percent
# e.g. get_percent_nonpolar("MEPFVVL") -> 0.85
get_percent_nonpolar <- function(seq) {
  # F, L, W, etc are "nonpolar" - we're going replace anything that's not one of these with "" (delete them)
  only_nonpolar <- gsub("[^FLWPIMVA]", "", seq)
  return(nchar(only_nonpolar)/nchar(seq))
}

# now when we read the metadata df, we can specify a named list of functions to use to produce extra columns
seqids_df <- fastas_to_targets_df(c("p450s.fa", "fmos.fa"), 
                                  directory = "datasets/proteins",
                                  function_list = list(percent_nonpolar = get_percent_nonpolar),
                                  alphabet = "protein")
# see:
print(head(seqids_df))


# split into train and validate
train_indices <- sample(1:nrow(seqids_df), size = nrow(seqids_df) * 0.8)
seqids_df_train <- seqids_df[train_indices, ]
seqids_df_validate <- seqids_df[-train_indices, ]

# figure out what we'll trim to
min_length <- min(seqids_df$seq_len)   # minimum over all sequences

# as above, except: targets_col is the new "percent_nonpolar"
# class_mode = "identity" (don't alter the targets)
train_generator <- flow_sequences_from_fasta_df(seqids_df_train, 
                                                targets_col = "percent_nonpolar",
                                                alphabet = "protein",
                                                batch_size = 64,
                                                class_mode = "identity",
                                                trim_to = min_length)

validate_generator <- flow_sequences_from_fasta_df(seqids_df_validate, 
                                                   targets_col = "percent_nonpolar",
                                                   alphabet = "protein",
                                                   batch_size = 64,
                                                   class_mode = "identity",
                                                   trim_to = min_length)


# as above, except activation is linear instead of sigmoid - so that network can output any number, positive or
# negative (though since our percents are between 0 and 1 anyway, we could have tried sigmoid)
network <- keras_model_sequential() %>%
  layer_conv_1d(filters = 16, kernel_size = 3, activation = "relu", input_shape = c(450, 21)) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%  
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")


# loss function is "mean squared error" - appropriate for regression
# metric is "mean absolute error" - appropriate for regression ("accuracy" is not appropriate for regression,
# even though keras will let us specify it and it will look like it makes sense)
compile(network, 
        optimizer = "rmsprop",
        loss = "mse",
        metrics = c("mae"))

fit_generator(network, 
              train_generator, 
              epochs = 5,
              steps_per_epoch = 50,
              validation_data = validate_generator,
              validation_steps = 10)


data_generator <- flow_sequences_from_fasta_df(seqids_df, 
                                               targets_col = "class",
                                               alphabet = "protein",
                                               batch_size = 1000,
                                               class_mode = "binary",
                                               trim_to = min_length)

batch <- data_generator()
actuals <- batch[[2]]
preds <- predict(network, batch[[1]]) %>% # returns a shape (1, 1000) tensor...
  array_reshape(dim = c(1000))            # make it a vector

head(preds)
head(actuals)



pred_classes <- preds * 0
pred_classes[preds > 0.5] <- 1

table(actuals, pred_classes)




library(PRROC)

