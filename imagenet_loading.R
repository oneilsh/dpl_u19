library(keras)
library(tensortree)

# remove images anywhere within the folder imagenet_validation that aren't of type JPG
# find imagenet_validation -type f -exec file --mime-type {}  \; | awk '{if($NF!= "image/jpeg") print $0}' | awk -F: '{print "\""$1"\""}'

## generators for training data
train_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory("datasets/imagenet",
                                              train_datagen,
                                              target_size = c(100, 100),
                                              batch_size = 20,
                                              class_mode = "categorical",
                                              classes = c("huntsman", "beetle", "centipede", "shiitake"))

## let's see what we got
batch <- generator_next(train_generator)
str(batch)

batch[[1]] %>% tt()
batch[[2]] %>% tt()

## generators for validation data

validate_datagen <- image_data_generator(rescale = 1/255)

validate_generator <- flow_images_from_directory("datasets/imagenet_validation",
                                                 validate_datagen,
                                                 target_size = c(100, 100),
                                                 batch_size = 20,
                                                 class_mode = "categorical",
                                                 classes = c("huntsman", "beetle", "centipede", "shiitake"))


network <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu", input_shape = c(100, 100, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%   # strides defaults to pool_size
  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%  
  
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 4, activation = "softmax")

compile(network, 
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))


history <- fit_generator(network, 
                         train_generator,
                         validation_data = validate_generator,
                         steps_per_epoch = 130,
                         epochs = 10,
                         validation_steps = 50)

save_model_hdf5(network, "insects_mushroom_network_v1.h5")

