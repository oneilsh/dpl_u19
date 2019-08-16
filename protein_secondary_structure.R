library(keras)
library(tidytensor)

# source: https://www.kaggle.com/alfrandom/protein-secondary-structure/
structure <- read.table("/storage/dpl_u19/datasets/proteins/secondary_structure/2018-06-06-ss.cleaned.csv", 
                        header = TRUE, 
                        sep = ",", 
                        stringsAsFactors = FALSE)
print(head(structure))

subset <- structure[structure$len >= 500 & structure$has_nonstd_aa == "False", ]
print(head(subset, n = 1))
print(nrow(subset))

# note: we use 21 as the number tokens to keep track of even though there are only 20 letters, 
# because "0" is reserved. This is a known keras quirk: https://github.com/keras-team/keras/issues/8583#issuecomment-346981336
tokenizer_seq <- text_tokenizer(num_words = 21, char_level = TRUE)
tokenizer_seq <- fit_text_tokenizer(tokenizer_seq, subset$seq)

seqs_sparse <- texts_to_sequences(tokenizer_seq, subset$seq)
print(head(seqs_sparse))

seqs_padded <- pad_sequences(seqs_sparse, maxlen = 500)
seqs_padded %>% tt() %>% print(bottom = "2d", end_n = 15)

seqs_onehot <- to_categorical(seqs_padded)
seqs_onehot %>% tt() %>% print(bottom = "2d", end_n = 21)

tokenizer_sst3 <- text_tokenizer(char_level = TRUE) %>%
  fit_text_tokenizer(subset$sst3)
sst3_sparse <- texts_to_sequences(tokenizer_sst3, subset$sst3)
sst3_padded <- pad_sequences(sst3_sparse, maxlen = 500)
sst3_onehot <- to_categorical(sst3_padded)
sst3_onehot %>% tt() %>% print(bottom = "2d")


tokenizer_sst8 <- text_tokenizer(char_level = TRUE) %>%
  fit_text_tokenizer(subset$sst8)
sst8_sparse <- texts_to_sequences(tokenizer_sst8, subset$sst8)
sst8_padded <- pad_sequences(sst8_sparse, maxlen = 500)
sst8_onehot <- to_categorical(sst8_padded)

rm(list = ls(pattern = "(padded)|(sparse)"))
rm(structure)
rm(subset)
gc()

#########


train_indices <- sample(1:33887, size = 0.8 * 33887)

train_seqs <- seqs_onehot[train_indices, , ]
validate_seqs <- seqs_onehot[-train_indices, , ]

train_sst8 <- sst8_onehot[train_indices, , ]
validate_sst8 <- sst8_onehot[-train_indices, , ]

train_sst3 <- sst3_onehot[train_indices, , ]
validate_sst3 <- sst3_onehot[-train_indices, , ]

rm(seqs_onehot, sst8_onehot, sst3_onehot)
gc()

basic <- keras_model_sequential() %>%
  layer_conv_1d(input_shape = c(500, 21), filters = 32, kernel_size = 9, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  #layer_gru(units = 32, return_sequences = TRUE) %>%
  layer_flatten() %>%
  layer_dense(units = 500 * 4, activation = "linear") %>%
  layer_reshape(target_shape = c(500, 4)) %>%     # for comparison to a single sst3 one-hot seq.
  layer_activation_softmax()                      # applies softmax over the last rank

compile(basic, optimizer = "adam", loss = "categorical_crossentropy", metrics = c("accuracy"))

fit(basic,
    train_seqs,
    train_sst3,
    batch_size = 128, 
    epochs = 10, 
    validation_data = list(validate_seqs, validate_sst3))

###

sst8_basic <- keras_model_sequential() %>%
  layer_conv_1d(input_shape = c(500, 21), filters = 32, kernel_size = 9, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  #layer_gru(units = 32, return_sequences = TRUE) %>%
  layer_flatten() %>%
  layer_dense(units = 500 * 9, activation = "linear") %>%
  layer_reshape(target_shape = c(500, 9)) %>%
  layer_activation_softmax()


compile(sst8_basic, optimizer = "adam", loss = "categorical_crossentropy", metrics = c("accuracy"))

fit(sst8_basic,
    train_seqs,
    train_sst8,
    batch_size = 256, 
    epochs = 10, 
    validation_data = list(validate_seqs, validate_sst8))


# hybrid model - predict sst8 from seq and sst3

seq_input <- layer_input(shape = c(500, 21)) 
seq_conv <- seq_input %>% 
  layer_conv_1d(input_shape = c(500, 21), filters = 32, kernel_size = 9, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten()

sst3_input <- layer_input(shape = c(500, 4)) 
sst3_conv <- sst3_input %>% 
  layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten()



model_output <- layer_concatenate(list(sst3_conv, seq_conv)) %>%
  layer_dense(units = 500 * 9, activation = "linear") %>%
  layer_reshape(target_shape = c(500, 9)) %>% 
  layer_activation_softmax()

model <- keras_model(list(sst3_input, seq_input), model_output)

compile(model, optimizer = "adam", loss = "categorical_crossentropy", metrics = c("accuracy"))

fit(model,
    list(train_sst3, train_seqs),
    train_sst8,
    batch_size = 128, 
    epochs = 10, 
    validation_data = list(list(validate_sst3, validate_seqs), validate_sst8))

# hybrid - predict sst3 and sst8 from seqs

seq_input <- layer_input(shape = c(500, 21))
seq_conv <- seq_input %>% 
  layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten()

sst3_output <- seq_conv %>%
  layer_dense(units = 500 * 4, activation = "linear") %>%
  layer_reshape(target_shape = c(500, 4), name = "sst3_output") %>%
  layer_activation_softmax()

sst8_output <- seq_conv %>%
  layer_dense(units = 500 * 9, activation = "linear") %>%
  layer_reshape(target_shape = c(500, 9), name = "sst8_output") %>%
  layer_activation_softmax()

model <- keras_model(seq_input, list(sst3_output, sst8_output))

compile(model, 
        optimizer = "rmsprop", 
        loss = c("categorical_crossentropy", "categorical_crossentropy"), 
        metrics = c("accuracy", "accuracy"),
        loss_weights = c(1, 1))

fit(model,
    train_seqs, 
    list(train_sst3, train_sst8),
    batch_size = 128, 
    epochs = 10, 
    validation_data = list(validate_seqs, list(validate_sst3, validate_sst8)))


m <- keras_model_sequential() %>%
  layer_dense(input_shape = c(5, 2), units = 3, name = "i")

1:(1*5*2) %>% array_reshape(dim = c(1, 5, 2)) %>% predict(m, .) %>% tt() %>% print(bottom = "2d")

get_weights(get_layer(m, name = "i"))


# same as above but with a residual connection - doesn't seem to help much here

seq_input <- layer_input(shape = c(500, 21))
seq_conv <- seq_input %>% 
  layer_conv_1d(filters = 32, kernel_size = 9, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 500 * 21, activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.25) %>%
  layer_reshape(target_shape = c(500, 21))

seq_conv2 <- layer_add(list(seq_input, seq_conv)) %>%
  layer_flatten()

sst3_output <- seq_conv2 %>%
  layer_dense(units = 500 * 4, activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.25) %>%
  layer_reshape(target_shape = c(500, 4), name = "sst3_output") %>%
  layer_activation_softmax()

sst8_output <- seq_conv2 %>%
  layer_dense(units = 500 * 9, activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.25) %>%
  layer_reshape(target_shape = c(500, 9), name = "sst8_output") %>%
  layer_activation_softmax()

model <- keras_model(seq_input, list(sst3_output, sst8_output))

compile(model, 
        optimizer = "rmsprop", 
        loss = c("categorical_crossentropy", "categorical_crossentropy"), 
        metrics = c("accuracy", "accuracy"),
        loss_weights = c(1, 1))

fit(model,
    train_seqs, 
    list(train_sst3, train_sst8),
    batch_size = 128, 
    epochs = 10, 
    validation_data = list(validate_seqs, list(validate_sst3, validate_sst8)))






