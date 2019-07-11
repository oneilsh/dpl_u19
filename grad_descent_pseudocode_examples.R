# these were/are largely for illustrative purposes in class

get_prediction <- function(x, p1, p2) {
  yhat <- p1 * x + p2 ^ 2
  yhat
}

get_loss <- function(x, p1, p2, y) {
  yhat <- get_prediction(x, p1, p2)
  loss <- (yhat - y) ^ 2
  loss
}

