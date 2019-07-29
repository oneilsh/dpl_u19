library(keras)
library(tensortree)
library(tidyr)
library(dplyr)
library(ggplot2)


####### Reading model & an input image, making a prediction
vgg16 <- application_vgg16(weights = "imagenet")

apple <- image_load("datasets/apple.jpg", target_size = c(224, 224)) %>%
  image_to_array(data_format = "channels_last") %>%
  tt()

print(apple)
{}


apple_batch <- array_reshape(apple, dim = c(1, 224, 224, 3)) %>% 
  tt()
print(apple_batch)
{}


pred <- predict(vgg16, apple_batch)
print(pred)
print(which.max(pred))
print(imagenet_decode_predictions(pred, top = 5))
{}





#### Visualizing filter weights

layer <- get_layer(vgg16_full, name = "block1_conv1")
layer_weights <- get_weights(layer)
str(layer_weights)
{}



weights_matrix <- layer_weights[[1]]
bias <- layer_weights[[2]]

for(filter_num in 1:64) {
  weights_matrix[,,,filter_num] <- weights_matrix[,,,filter_num] + bias[filter_num]
}
{}


weights_matrix[,,,1:32] %>% tt() %>%
  set_ranknames(c("kernel_row", "kernel_col", "input_channel", "filter")) %>%
  set_dimnames_for_rank("input_channel", c("R", "G", "B")) %>%
  permute(c("filter", "kernel_row", "kernel_col", "input_channel")) %>%
  as.data.frame() %>% 
  ggplot() +
    geom_tile(aes(x = kernel_col, y = kernel_row, fill = value)) +
    coord_equal() +
    facet_grid(input_channel ~ filter) +
    scale_fill_gradient2(low = "purple", high = "orange")
{}




scale_to_0_1 <- function(tensor) {
  tensor <- tensor - min(tensor) # start at 0...
  tensor <- tensor / max(tensor) # scale to 1...
  return(tensor)
}
{}




weights_matrix[,,,1:64] %>% tt() %>%
  set_ranknames(c("kernel_row", "kernel_col", "input_channel", "filter")) %>%
  set_dimnames_for_rank("input_channel", c("R", "G", "B")) %>%
  permute(c("filter", "kernel_row", "kernel_col", "input_channel")) %>%
  scale_to_0_1() %>%
  as.data.frame() %>% 
  spread(input_channel, value) %>%
  ggplot() +
    geom_tile(aes(x = kernel_col, y = kernel_row, fill = rgb(R, G, B))) +
    coord_equal() +
    facet_wrap( ~ filter) +
    scale_fill_identity()
{}



########## visualizing feature maps for an input image
layer <- get_layer(vgg16, name = "block5_pool")
print(layer$output)
{}



layer_outputs <- layer$output[,,,1:32]
print(layer_outputs)
{}



compute_layer_outputs <- k_function(vgg16$input, layer_outputs)

compute_layer_outputs(apple_batch) %>% tt() %>%
  set_ranknames(c("image_num", "row", "col", "filter")) %>%
  as.data.frame(allow_huge = TRUE) %>%
  ggplot() +
    geom_tile(aes(x = col, y = -1 * row, fill = value)) +
    coord_equal() +
    facet_wrap(~ filter)
{}





