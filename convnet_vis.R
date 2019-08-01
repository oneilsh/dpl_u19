library(keras)
library(tensortree)
library(tidyr)
library(dplyr)
library(ggplot2)

##############################################
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




##############################################
#### Visualizing filter weights

layer <- get_layer(vgg16, name = "block1_conv1")
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


##############################################
########## visualizing feature maps for an input image

layer <- get_layer(vgg16, name = "block1_conv1")
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



##############################################
####### Optimizing input for filter activation


layer <- get_layer(vgg16, name = "block4_conv1")
print(layer$output)
{}



filter_output <- layer$output[,,,1]
# uncomment to instead optimize for predicting Granny Smith
#filter_output <- vgg16$output[, 949]
print(filter_output)
{}



filter_loss <- -1 * k_mean(filter_output)

filter_loss_grads <- k_gradients(filter_loss, list(vgg16$input))
filter_loss_grad_wrt_input <- filter_loss_grads[[1]]

input_image <- runif(224 * 224 * 3, min = 100, max = 110) %>%
  array_reshape(dim = c(1, 224, 224, 3))

input_image <- apple_batch

compute_filter_loss_grad_wrt_input <- k_function(vgg16$input, filter_loss_grad_wrt_input)
compute_filter_loss_grad_wrt_input(input_image) %>% tt() %>% print(end_n = 3)
{}



filter_loss_grad_wrt_input_norm <- filter_loss_grad_wrt_input / k_std(filter_loss_grad_wrt_input)
#(k_sqrt(k_mean(k_square(
#       filter_loss_grad_wrt_input))) +
#1e-5) # don't accidentally divide by 0

compute_filter_loss_grad_wrt_input_norm <- k_function(vgg16$input, filter_loss_grad_wrt_input_norm)
compute_filter_loss_grad_wrt_input_norm(input_image) %>% tt() %>% print(end_n = 3)
{}




for(i in 1:100) {
  grads_value <- compute_filter_loss_grad_wrt_input_norm(input_image)
  input_image <- input_image - grads_value
}
{}



input_image %>% tt() %>%
  set_ranknames(c("image_num", "row", "col", "channel")) %>%
  set_dimnames_for_rank("channel", c("R", "G", "B")) %>%
  scale_to_0_1() %>%
  as.data.frame() %>%
  spread(channel, value) %>%
  ggplot() +
  geom_tile(aes(x = col, y = -1 * row, fill = rgb(R, G, B))) +
  coord_equal() +
  scale_fill_identity()


predict(vgg16, input_image) %>% imagenet_decode_predictions()








###################
# putting it in a loop to try different filters automatically


optimize_filtervis <- function(model, 
                               layer_name, 
                               filter_number, 
                               input_image = runif(224 * 224 * 3, min = 100, max = 110) %>% array_reshape(dim = c(1, 224, 224, 3))
                               ) {
  
  layer <- get_layer(model, name = layer_name)
  
  filter_output <- layer$output[,,,filter_number]

  filter_loss <- -1 * k_mean(filter_output)
  
  filter_loss_grads <- k_gradients(filter_loss, list(vgg16$input))
  filter_loss_grad_wrt_input <- filter_loss_grads[[1]]
  
  compute_filter_loss_grad_wrt_input <- k_function(vgg16$input, filter_loss_grad_wrt_input)

  filter_loss_grad_wrt_input_norm <- filter_loss_grad_wrt_input / k_std(filter_loss_grad_wrt_input)
  #(k_sqrt(k_mean(k_square(
  #       filter_loss_grad_wrt_input))) +
  #1e-5) # don't accidentally divide by 0
  
  compute_filter_loss_grad_wrt_input_norm <- k_function(vgg16$input, filter_loss_grad_wrt_input_norm)

  for(i in 1:40) {
    grads_value <- compute_filter_loss_grad_wrt_input_norm(input_image)
    input_image <- input_image - grads_value
  }
  
  result <- input_image %>% tt() %>%
    set_ranknames(c("image_num", "row", "col", "channel")) %>%
    set_dimnames_for_rank("channel", c("R", "G", "B")) %>%
    scale_to_0_1() %>%
    as.data.frame() %>%
    spread(channel, value)
  
  # add columns to remember the layer name and filter number
  result$layer_name <- layer_name
  result$filter_number <- paste(filter_number, collapse = "+")
  
  return(result)
}

library(rstackdeque)

result_stack <- rstack()
for(layer_name in c("block5_conv1")) {
  num_filters <- get_layer(vgg16, name = layer_name)$filters
  for(filter_num in c(as.list(17:32), list(18, 30, c(18, 30)))) { 
    cat("Optimizing for filter ", filter_num, "\n")
    result_df <- optimize_filtervis(vgg16, layer_name, filter_num)
    result_stack <- insert_top(result_stack, result_df)
  }
}

all_results <- as.data.frame(result_stack)
all_results$R[is.nan(all_results$R)] <- 0
all_results$G[is.nan(all_results$G)] <- 0
all_results$B[is.nan(all_results$B)] <- 0

all_results %>% ggplot() +
  geom_tile(aes(x = col, y = -1 * row, fill = rgb(R, G, B))) +
  coord_equal() +
  scale_fill_identity() +
  facet_wrap(~ filter_number)





#########################
# Just playing around...

layer <- get_layer(vgg16, name = "block1_conv1")
layer_outputs <- layer$output[,,,]
print(layer_outputs)

compute_layer_outputs <- k_function(vgg16$input, layer_outputs)

apple_features <- compute_layer_outputs(apple_batch) %>% tt() %>%
  set_ranknames(c("image_num", "row", "col", "filter")) 

# optimize for an image that looks 
test_outputs <- layer_outputs * apple_features
#test_outputs <- vgg16$output[,949]
#test_outputs <- metric_mean_squared_error(predict(vgg16, apple_batch), vgg16$output)

filter_loss <- -1 * k_mean(test_outputs)

filter_loss_grads <- k_gradients(filter_loss, list(vgg16$input))
filter_loss_grad_wrt_input <- filter_loss_grads[[1]]

input_image <- runif(224 * 224 * 3, min = 200, max = 200) %>% array_reshape(dim = c(1, 224, 224, 3))
#input_image <- apple_batch

compute_filter_loss_grad_wrt_input <- k_function(vgg16$input, filter_loss_grad_wrt_input)
compute_filter_loss_grad_wrt_input(input_image) %>% tt() %>% print(end_n = 3)
{}



filter_loss_grad_wrt_input_norm <- filter_loss_grad_wrt_input / k_std(filter_loss_grad_wrt_input)
#(k_sqrt(k_mean(k_square(
#       filter_loss_grad_wrt_input))) +
#1e-5) # don't accidentally divide by 0

compute_filter_loss_grad_wrt_input_norm <- k_function(vgg16$input, filter_loss_grad_wrt_input_norm)
compute_filter_loss_grad_wrt_input_norm(input_image) %>% tt() %>% print(end_n = 3)
{}




for(i in 1:100) {
  grads_value <- compute_filter_loss_grad_wrt_input_norm(input_image)
  input_image <- input_image - grads_value
}
{}



input_image %>% tt() %>%
  set_ranknames(c("image_num", "row", "col", "channel")) %>%
  set_dimnames_for_rank("channel", c("R", "G", "B")) %>%
  scale_to_0_1() %>%
  as.data.frame() %>%
  spread(channel, value) %>%
  ggplot() +
  geom_tile(aes(x = col, y = -1 * row, fill = rgb(R, G, B))) +
  coord_equal() +
  scale_fill_identity()

predict(vgg16, input_image) %>% imagenet_decode_predictions()





###################
# Activation heatmaps - covering only briefly, but this is the alg


# 
apple_output <- vgg16$output[, 949]
layer <- get_layer(vgg16, name = "block5_conv3")

grads <- k_gradients(apple_output, layer$output)[[1]]

pooled_grads <- k_mean(grads, axis = c(1, 2, 3))

iterate <- k_function(list(vgg16$input), list(pooled_grads, layer$output[1,,,]))

c(pooled_grads_value, layer_output_value) %<-% iterate(list(apple_batch))

for(i in 1:512) {
  layer_output_value[,,i] <- layer_output_value[,,i] * pooled_grads_value[[i]]
}

heatmap <- apply(layer_output_value, c(1, 2), mean)

apple_df <- apple_batch %>% tt() %>%
  set_ranknames(c("image", "row", "col", "channel")) %>%
  set_dimnames_for_rank("channel", c("R", "G", "B")) %>%
  scale_to_0_1() %>%
  as.data.frame() %>%
  spread(channel, value)

heatmap_df <- heatmap %>% tt() %>%
  set_ranknames(c("row", "col")) %>% 
  scale_to_0_1() %>%
  as.data.frame() %>%
  mutate(row = row * 16 - 8, col = col * 16 - 8) # scale up to 224 by 224 and center tiles


ggplot() +
  geom_tile(data = apple_df, 
            aes(x = col, y = -1 * row, fill = rgb(R, G, B))) +
  geom_tile(data = heatmap_df,
            aes(x = col, y = -1 * row, fill = "blue", alpha = -1 * value)) +
  coord_equal() +
  scale_fill_identity()

  
  


