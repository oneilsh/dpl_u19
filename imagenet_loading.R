library(keras)
library(tensortree)

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


# example fitting (if network has the right input and output shape, and has been compiled...)

history <- fit_generator(network, 
                         train_generator,
                         validation_data = validate_generator,
                         steps_per_epoch = 130,
                         epochs = 10,
                         validation_steps = 50)

