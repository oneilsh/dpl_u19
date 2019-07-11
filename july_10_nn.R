library(keras)
library(tensortree)



# our first neural net! First let's load the data...
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y

validate_images <- mnist$test$x
validate_labels <- mnist$test$y

# let's see what we have
train_images %>% tt() %>% print(end_n = 28)
train_labels %>% tt()

validate_images %>% tt()
validate_labels %>% tt()



# do a quick vis
plot(as.raster(train_images[1, , ], max = 255))


# reformat training image data, new shape: (60000, 784)
train_images_shaped <- array_reshape(train_images, dim = c(60000, 28 * 28)) 
train_images_shaped %>% tt()

# now validation images, new shape: (10000, 784)
validate_images_shaped <- array_reshape(validate_images, dim = c(10000, 28 * 28))  

# scale to 0.0 to 1.0
train_images_shaped <- train_images_shaped / 255                                   
validate_images_shaped <- validate_images_shaped / 255



# reformat label data (one-hot encode)
train_labels %>% tt()                          # just to remember
train_labels_onehot <- to_categorical(train_labels)
train_labels_onehot %>% tt() 

validate_labels_onehot <- to_categorical(validate_labels)



# build model!
network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%  # input: rank-1 tensors of shape (784)
  layer_dense(units = 10, activation = "softmax")



# compile model!
compile(network, 
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))


# train the model!
validation_data_list <- list(validate_images_shaped, validate_labels_onehot)

history <- fit(network, 
               train_images_shaped, 
               train_labels_onehot,
               validation_data = validation_data_list,
               epochs = 5,
               batch_size = 128)



# evaluate the trained model on validation data separately
metrics <- evaluate(network, validate_images_shaped, validate_labels_onehot)
print(metrics)


# compare model predictions to actual answers
predictions <- predict(network, validate_images_shaped)
predictions %>% 
  tt() %>%
  print(end_n = 10, bottom = "2d")

# compare to:
validate_labels_onehot %>% 
  tt() %>%
  print(end_n = 10, bottom = "2d")



# using keras' predict_classes() to un-onehot the prediction vectors:
predictions <- predict_classes(network, validate_images_shaped)
predictions %>% 
  tt() %>%
  print(end_n = 10)

# compare to:
validate_labels %>%
  tt() %>%
  print(end_n = 10)







