library(keras)
library(tensortree)

# build a dataframe with one column listing the filenames, recursive = TRUE lists everything in all subfolders,
# include.dirs = FALSE says not to include the directories themselves in the listing
image_metadata <- data.frame(filename = list.files("datasets/PlantVillage/Tomato", 
                                                   recursive = TRUE, 
                                                   include.dirs = FALSE),
                             stringsAsFactors = FALSE)


# lets only keep 4k images to make it tougher
random_indices <- sample(1:nrow(image_metadata), size = 4000)
image_metadata <- image_metadata[random_indices,,  drop = FALSE]  

# extract a "class" column using tidyr's extract() function; class is everything up to the first / (ie the subfolder name)
library(tidyr)
image_metadata <- extract(image_metadata, "filename", "class", regex = "([^/]+)", remove = FALSE)
{}






# training and validation data frames
train_indices <- sample(1:nrow(image_metadata), size = nrow(image_metadata) * 0.8)
train_metadata <- image_metadata[train_indices, ]
validate_metadata <- image_metadata[-train_indices, ]

# data generators; one for validation...
validate_datagen <- image_data_generator(rescale = 1/255)
{}







# and one for training, with augmentation
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)



train_generator <- flow_images_from_dataframe(train_metadata,
                                              directory = "datasets/PlantVillage/Tomato",
                                              x_col = "filename",
                                              y_col = "class",
                                              generator = train_datagen,
                                              target_size = c(100, 100),
                                              batch_size = 64,
                                              class_mode = "categorical")

validate_generator <- flow_images_from_dataframe(validate_metadata,
                                                 directory = "datasets/PlantVillage/Tomato",
                                                 x_col = "filename",
                                                 y_col = "class",
                                                 generator = validate_datagen,
                                                 target_size = c(100, 100),
                                                 batch_size = 64,
                                                 class_mode = "categorical")
{}






## let's inspect what this augmentation looks like...
batch <- generator_next(train_generator)
batch[[1]] %>% tt() %>% print(end_n = 4)

first4 <- batch[[1]][1:4, , , ]

library(dplyr)
library(ggplot2)

first4 %>% tt() %>%
  set_ranknames(c("image", "row", "col", "channel")) %>%
  set_dimnames_for_rank("channel", c("R", "G", "B")) %>%
  as.data.frame() %>%
  spread(channel, value) %>%
  mutate(color = rgb(R, G, B)) %>%
  ggplot() +
    geom_tile(aes(x = col, y = -1*row, fill = color)) +
    scale_fill_identity() + 
    coord_equal() +
    facet_wrap(~ image)


# train a basic CNN on it
basicnet <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = c(100, 100, 3),
                filters = 16, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


compile(basicnet,
        loss = "categorical_crossentropy",
        optimizer = "rmsprop",
        metrics = c("accuracy"))


history <- fit_generator(basicnet,
                         train_generator,
                         steps_per_epoch = 50,   # ~ 3200 (# training examples) / 64 (batch size)
                         epochs = 10,
                         validation_data = validate_generator,
                         validation_steps = 12)    # ~ 800 / 64




# build a model based on a predefined architecture ("VGG16"), set the weights according to a predefined
# weight set ("imagenet")
# data are stored to ~/.keras/, which I don't think is configurable in the installed version
vgg16_full <- application_vgg16(weights = "imagenet")

# inspect the architecture
print(vgg16_full)

# or we can download without the flattened, dense layers on top -
# if we do so, we can alter the input shape (not sure why we can't otherwise, must be some weight interpolation
# being done for the new network architecture that would be too costly or intractable with the dense layers)
vgg16_topless <- application_vgg16(weights = "imagenet",
                           include_top = FALSE,
                           input_shape = c(100, 100, 3))



# inspect the architecture
print(vgg16_topless)



# put some new class prediction stuff on top
my_vgg16 <- keras_model_sequential() %>%
  vgg16_topless %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


# freeze the weights for the convolutional base - we don't want to mess up what's already there!
freeze_weights(vgg16_topless)


compile(my_vgg16,
        loss = "categorical_crossentropy",
        optimizer = "rmsprop",
        metrics = c("accuracy"))


history <- fit_generator(my_vgg16,
               train_generator,
               steps_per_epoch = 50,   # ~ 3200 (# training examples) / 64 (batch size)
               epochs = 10,
               validation_data = validate_generator,
               validation_steps = 12)    # ~ 800 / 64



# graph surgery: extracting a model from a set of layers in an existing model
input_vgg16 <- get_layer(vgg16_full, index = 1)$input
output_features <- get_layer(vgg16_full, index = 19)$output

vgg16_topless <- keras_model(input_vgg16, output_features)

