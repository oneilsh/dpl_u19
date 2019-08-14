library(keras)
library(tensortree)

mnist <- dataset_mnist()

train_x <- mnist$train$x
train_x <- train_x/ 255

validate_x <- mnist$test$x
validate_x <- validate_x / 255



# generating a model without keras_model_sequential; we start with an input layer (we can 
# attach others layers to this if we want)
input_layer <- layer_input(shape = c(28, 28))

# then we define other layers that attach to those; note that we name the layer that produces
# the 'latent vector' (the bottleneck encoding) and the one after that; the ouput produces the same shape
# and range of values as the input
# rather than reshape the data before feeding it to the network, we can build that into the network itself
output <- input_layer %>%
  layer_reshape(target_shape = c(28 * 28)) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%  
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu", name = "encoded_output") %>%
  layer_dense(units = 32, activation = "relu", name = "encoded_input") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%  
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 28 * 28, activation = "sigmoid") %>%
  layer_reshape(target_shape = c(28, 28))

# to turn the layers into a model, we specify the input and output layers
model <- keras_model(input_layer, output)

# this is a regression problem...
compile(model, optimizer = "rmsprop", loss = "mse", metrics = "mae")

# where we want to predict the input
fit(model, 
    x = train_x, 
    y = train_x, 
    batch_size = 256, 
    epochs = 10, 
    validation_data = list(validate_x, validate_x))

# let's get some predictions from originals and plot them
originals <- validate_x[1:4, , ] %>% tt()
predicted <- predict(model, originals) %>% tt()
print(originals)
print(predicted)

toplot <- bind(predicted, originals)
print(toplot)

library(ggplot2)
toplot %>% 
  set_ranknames(c("type", "image", "row", "col")) %>%
  set_dimnames_for_rank("type", c("predicted", "original")) %>%
  as.data.frame() %>% 
  ggplot() +
    geom_tile(aes(x = col, y = -1*row, fill = value)) +
    facet_grid(type ~ image) +
    coord_equal()


########

# we can create k_functions that map model inputs to the encoded tensors, 
# and encoded tensors to model outputs
encoding_out_layer <- get_layer(model, name = "encoded_output")
encoder <- k_function(model$input, encoding_out_layer$output)

decoding_in_layer <- get_layer(model, name = "encoded_input")
decoder <- k_function(decoding_in_layer$input, model$output)

seven <- validate_x[1, , , drop = FALSE] # shape (1, 28, 28)
zero <- validate_x[4, , , drop = FALSE] # shape (1, 28, 28)

# getting encoded examples
seven_latent <- encoder(seven)
zero_latent <- encoder(zero)
print(seven_latent)
print(zero_latent)

# rather than just predict those latent space vectors, let's make a brand new mix-of-two 
# and see what pops out! this works (to some degree) because the latent space is structured, 
# much like embedding spaces are structured
mean_latent <- zero_latent * 0.5 + seven_latent * 0.5
mean_latent <- runif(16) %>% array_reshape(dim = c(1, 16))
mean_decoded <- decoder(mean_latent) %>% tt()

mean_decoded %>%
  set_ranknames(c("image", "row", "col")) %>%
  as.data.frame() %>% 
  ggplot() +
  geom_tile(aes(x = col, y = -1*row, fill = value)) +
  coord_equal()


# the trouble with the above is that the latent space has *too* much structure, 
# if we were to plot the principal components of the latent vectors, they'd cluster tightly
# so a mix of two would produce a latent vector the decoder has never seen before, often
# resuling in nonsense output.

# variational autoencoders fix this by 1) sampling latent vectors during the training process,
# and 2) tweaking the loss so that the latent vectors are packed near the origin; these
# two tricks make the latent space clusters smoothly transition into each other.

# below are my initial attempts at a variational autoencoder, please ignore (and the loss
# function is incorrect)

#######################

input_layer <- layer_input(shape = c(28, 28), name = "input")

encoder_base <- input_layer %>%
  layer_reshape(target_shape = c(28 * 28)) %>%
  layer_dense(units = 128, activation = "relu") %>%  
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu")


encoder_mean <- encoder_base %>%  
  layer_dense(units = 16, activation = "linear", name = "encoded_mean")

encoder_log_var <- encoder_base %>%  
  layer_dense(units = 16, activation = "linear", name = "encoded_log_var")

# takes a list of two tensors, of shape (?, k), where ? is the batch size
sampler <- function(list_of_two_tensors) {
  encoded_mean <- list_of_two_tensors[[1]]
  encoded_log_var <- list_of_two_tensors[[2]]
  
  batch_size <- k_shape(encoded_mean)[1]
  k <- k_shape(encoded_mean)[2]
  epsilon <- k_random_normal(shape = c(batch_size, k), mean = 0, stddev = 1)
  return(encoded_mean + k_exp(encoded_log_var) * epsilon)
}

sampled <- layer_lambda(list(encoder_mean, encoder_log_var), sampler)

image_to_sampled_latent <- keras_model(input_layer, sampled)

predict(image_to_sampled_latent, originals)

decoder <- sampled %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%  
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 28 * 28, activation = "sigmoid") %>%
  layer_reshape(target_shape = c(28, 28), name = "output")


autoencoder <- keras_model(input_layer, decoder)

originals <- validate_x[1:4, , ] %>% tt()
predicted <- predict(autoencoder, originals) %>% tt()
print(originals)
print(predicted)

toplot <- bind(predicted, originals)
print(toplot)

library(ggplot2)
toplot %>% 
  set_ranknames(c("type", "image", "row", "col")) %>%
  set_dimnames_for_rank("type", c("predicted", "original")) %>%
  as.data.frame() %>% 
  ggplot() +
  geom_tile(aes(x = col, y = -1*row, fill = value)) +
  facet_grid(type ~ image) +
  coord_equal()


custom_loss <- function(original, predicted) {
  latent <- decoder(original)
  unit_sampled <- k_random_normal(shape = k_shape(latent), mean = 0, stddev = 1)
  latent_loss <- k_mean(k_square(latent - unit_sampled))
  predicted_loss <- 100*k_mean(k_square(original - predicted))
  return(predicted_loss)
}

compile(autoencoder,
        loss = custom_loss,
        optimizer = "rmsprop",
        metrics = "mae")

fit(autoencoder, 
    x = train_x, 
    y = train_x, 
    batch_size = 256, 
    epochs = 10, 
    validation_data = list(validate_x, validate_x))



originals <- validate_x[1:4, , ] %>% tt()
predicted <- predict(autoencoder, originals) %>% tt()
print(originals)
print(predicted)

toplot <- bind(predicted, originals)
print(toplot)

library(ggplot2)
toplot %>% 
  set_ranknames(c("type", "image", "row", "col")) %>%
  set_dimnames_for_rank("type", c("predicted", "original")) %>%
  as.data.frame() %>% 
  ggplot() +
  geom_tile(aes(x = col, y = -1*row, fill = value)) +
  facet_grid(type ~ image) +
  coord_equal()


decoder <- k_function(decoder$input, decoder)
