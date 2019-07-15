# these were/are largely for illustrative purposes in class

get_prediction <- function(x, p1, p2) {
  yhat <- p1 * x + p2 ^ 2
  yhat
}


get_loss <- function(x, p1, p2, actualy) {
  yhat <- get_prediction(x, p1, p2)
  loss <- (yhat - actualy) ^ 2
  loss
}

library(Deriv)
deriv_p1 <- Deriv(get_loss, "p1")
deriv_p1(3, 1.4, 0.7, 8)



