library(keras)
library(tensortree)

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
  t_rankt <- array(0, dim = c(3, 2, 3, 4, 5))
  print(t_rankt[3, 2, 2, 4, 2])
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
  reshaped %>% tt() %>% print(max_per_level = 3, end_n = 8)
}
