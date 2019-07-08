library(keras)

tensor_rank1 <- c(2, 3, 1, 6, 7)
print(tensor_rank1[4])                         # 6

{
  # rank-2 tensor, shape (4, 5)
  t_rank2 <- matrix(0, nrow = 4, ncol = 5)     # 0 is recycled as needed
  print(t_rank2[4, 2])                         # row 4, col 2
}

{
  # rank-3 tensor, shape (3, 4, 5)
  t_rank3 <- array(0, dim = c(3, 4, 5))
  print(t_rank3[2, 4, 2])                      # 2 in 'z', row 4, col 2
}

{
  # rank-0 tensor, shape (1) (?)
  t_rank0 <- 7
}

{
  # rank-4 tensor, shape (2, 3, 4, 5)
  t_rank4 <- array(0, dim = c(2, 3, 4, 5))
  print(t_rank4[2, 2, 4, 2])                  # ...
}

{
  # rank-5 tensor, shape (3, 2, 3, 4, 5)
  r_rank4 <- array(0, dim = c(3, 2, 3, 4, 5))
  print(t_rank5[])
}

{
  # r vs python-style matrix filling
  nums <- 1:6                                 # same as c(1, 2, 3, 4, 5, 6)
  t_rank2 <- array(nums, dim = c(2, 3))       # same as matrix(nums, nrow = 2, ncol = 3)
  print(t_rank2)
  
  # python-style, via keras' array_reshape() function
  t_rank2 <- array_reshape(nums, dim = c(2, 3))
  print(t_rank2)
}

{
  # an 'image' of three 2x3 channels, with pixel valus 1 through 18
  image <- array_reshape(1:(3 * 2 * 4), dim = c(3, 2, 4))    # channels, rows, cols
  print(image)                                               # yuck
  
  # install fancy (probably also buggy) package
  install.packages("devtools")
  devtools::install_github("oneilsh/tensortree")
  
  # a tensortree is an array with a little extra metadata attached
  library(tensortree)
  print(as.tensortree(image))                                # nicer printout :)
  
  # shortcut for conversion to tensortree
  print(tt(image))
  
  # without explicit print()
  tt(image)
  
  # using %>%
  image %>% tt()
  
  # adjusting the printout
  print(tt(image), show_names = TRUE, max_per_level = 3)

  image %>% 
    tt() %>%
    print(show_names = TRUE, max_per_level = 3)
}

{
  # a dummy dataset, representing 1000 28x28 grayscale images
  dataset <- array_reshape(rnorm(1000 * 28 * 28), dim = c(1000, 28, 28))
  dataset %>% tt()
  
  print(dim(dataset))             # dim() gets the shape vector
  
  # subsetting to get the first 10 images:
  dataset[1:10, , ] %>% tt()
  
  # just the 10th image; R 'drop's it down to a rank-2 tensor
  dataset[10, , ] %>% tt()
  
  # use , drop = FALSE to keep it as a set of images (but with just one in there)
  dataset[10, , , drop = FALSE] %>% tt()
  
  # tensor reshaping via array_reshape()
  image <- array_reshape(1:(3 * 2 * 4), dim = c(3, 2, 4))    # channels, rows, cols
  image %>% tt() %>% print(max_per_level = 3)
  
  reshaped <- array_reshape(image, dim = c(3, 8))
  reshaped %>% tt()
}

{
  # our first neural net! First let's load the data...
  mnist <- dataset_mnist()
  train_images <- mnist$train$x
  train_labels <- mnist$train$y
  
  validate_images <- mnist$test$x
  validate_labels <- mnist$test$y
  
  train_images %>% tt()
  train_labels %>% tt()
  
  validate_images %>% tt()
  validate_labels %>% tt()
}

{
  # lets do a quick vis
  plot(as.raster(train_images[1, , ], max = 255))
}

{
  # reformat image data, new shape: (60000, 784)
  train_images_shaped <- array_reshape(train_images, dim = c(60000, 28 * 28)) 
  train_images_shaped %>% tt()
  
  # new shape: (10000, 784)
  validate_images_shaped <- array_reshape(validate_images, dim = c(10000, 28 * 28))  
  
  # scale to 0.0 to 1.0
  train_images_shaped <- train_images_shaped / 255                                   
  validate_images_shaped <- validate_images_shaped / 255
}

{
  # reformat label data (one-hot encode)
  train_labels %>% tt()                          # just to remember
  train_labels_onehot <- to_categorical(train_labels)
  train_labels_onehot %>% tt() 
  
  validate_labels_onehot <- to_categorical(validate_labels)
}

{
  # build model!
  network <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%  # input: rank-1 tensors of shape (784)
    layer_dense(units = 10, activation = "softmax")
}

{
  # compile model!
  compile(network, 
          optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))
}

{ # train the model!
  validation_data_list <- list(validate_images_shaped, validate_labels_onehot)
  
  history <- fit(network, 
                 train_images_shaped, 
                 train_labels_onehot,
                 validation_data = validation_data_list,
                 epochs = 5,
                 batch_size = 128)
}


{
  # evaluate the trained model on validation data separately
  metrics <- evaluate(network, validate_images_shaped, validate_labels_onehot)
  print(metrics)
}


{ # compare model predictions to actual answers
  predictions <- predict(network, validate_images_shaped)
  predictions %>% 
    tt() %>%
    print(end_n = 10, bottom = "2d")
  
  # compare to:
  validate_labels_onehot %>% 
    tt() %>%
    print(end_n = 10, bottom = "2d")
}

{
  # using keras' predict_classes() to un-onehot the prediction vectors:
  predictions <- predict_classes(network, validate_images_shaped)
  predictions %>% 
    tt() %>%
    print(end_n = 10)
  
  validate_labels %>%
    tt() %>%
    print(end_n = 10)
}






