

###
library(gutenbergr)
library(tensortree)
library(tfruns)
library(keras)

# download lines of text
alice_raw <- gutenberg_download(11)$text
warofworlds_raw <- gutenberg_download(4363)$text

gen_batches <- function(text, samples = 1000, length = 15) {
  text <- text %>% strsplit(" ") %>% unlist()
  starts <- sample(1:(length(text) - length), size = samples)
  pairs <- lapply(starts, 
         function(start) {
           xwordvec <- text[start : (start + length)]
           xwordvec <- paste(xwordvec, collapse = " ")
           ywordvec <- text[start + length + 1]
           return(list(xwordvec, ywordvec))
         })
  pairs1 <- purrr::map_chr(pairs, ~ .x[[1]])
  pairs2 <- purrr::map_chr(pairs, ~ .x[[2]])
  return(list(pairs1, pairs2))
}

alice_text <- gen_batches(alice_raw, samples = 8000)[[1]]
warofworlds_text <- gen_batches(warofworlds_raw, samples = 8000)[[1]]




# design class vectors for each line - alice is class 1, warofworlds is class 0
alice_class <- rep(1, length(alice_text))
warofworlds_class <- rep(0, length(warofworlds_text))

all_text <- c(alice_text, warofworlds_text)
all_class <- c(warofworlds_class, alice_class)
{}




tokenizer <- text_tokenizer(num_words = 10000)
tokenizer <- fit_text_tokenizer(tokenizer, all_text)


alice_sparse <- texts_to_sequences(tokenizer, alice_text)
warofworlds_sparse <- texts_to_sequences(tokenizer, warofworlds_text)
print(head(alice_sparse))
{}



alice_sparse_padded <- pad_sequences(alice_sparse, maxlen = 15)
warofworlds_sparse_padded <- pad_sequences(warofworlds_sparse, maxlen = 15)
# take a peek
alice_sparse_padded %>% tt() %>% print(end_n = 15, bottom = "2d")
{}
print(alice_text[1:2])


model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 128, input_length = 15)

embedded <- predict(model, alice_sparse_padded[1:5, ])

embedded %>% tt() %>% print(bottom = "2d")
{}






# let's try learning...
all_sparse_padded <- tokenizer %>% texts_to_sequences(all_text) %>% pad_sequences(maxlen = 15)
all_sparse_padded %>% tt() %>% print(bottom = "2d")
  
train_indices <- sample(1:length(all_text), size = 0.8 * length(all_text))

train_sparse_padded <- all_sparse_padded[train_indices, ]
validate_sparse_padded <- all_sparse_padded[-train_indices, ]
train_class <- all_class[train_indices]
validate_class <- all_class[-train_indices]




model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 128, input_length = 15, name = "input_embedding") %>%
  layer_batch_normalization() %>%
  layer_lstm(units = 32) %>%
  layer_flatten() %>%
  layer_dense(units = 1, activation = "sigmoid")

compile(model, 
        loss = "binary_crossentropy",
        optimizer = "rmsprop",
        metrics = c("accuracy"))

validate_list <- list(validate_sparse_padded, validate_class)


history <- fit(model, 
               x = train_sparse_padded,
               y = train_class,
               validation_data = validate_list,
               epochs = 4)




#tfruns::purge_runs()
#training_run("encoding_rnn_tfrun.R")
#tfruns::compare_runs()


####### Embeddings visualization
# top 1000 tokens by frequency
tokens <- tokenizer$index_word[1:1000] %>% 
  unlist()

sparse <- texts_to_sequences(tokenizer, tokens)
padded <- pad_sequences(sparse, maxlen = 1)

embedding_output <- get_layer(model, name = "input_embedding")$output

compute_embedding <- k_function(model$input, embedding_output)

embeddings <- compute_embedding(padded)
embeddings %>% tt() %>% print(bottom = "2d")

embeddings <- array_reshape(embeddings, dim = c(1000, 128))
pcres <- prcomp(embeddings)

x_df <- as.data.frame(pcres$x)
x_df$word <- iconv(tokens, "UTF-8", "UTF-8",sub='')


library(ggplot2)
library(plotly)

p <- ggplot(x_df) +
  geom_point(aes(x = PC1, y = PC2, text = word, color = 1:1000))

ggplotly(p)




library(Rtsne)
tsne_res <- Rtsne(embeddings, dims = 2)

tsne_df <- as.data.frame(tsne_res$Y)
tsne_df$word <- iconv(tokens, "UTF-8", "UTF-8",sub='')


p <- ggplot(tsne_df) +
  geom_point(aes(x = V1, y = V2, text = word, color = 1000:1))

ggplotly(p)



########### Looking for paired relationships... 

words <- x_df$word
library(rstackdeque)
library(svMisc)

pairstack <- rstack()

for(i in 1:999) {
  for(j in (i+1):1000) {
    diffvec <- embeddings[i] - embeddings[j]
    wordpair <- paste(words[i], " - ", words[j], collapse = "")
    pairstack <- insert_top(pairstack, list(diffvec = diffvec, wordpair = wordpair))
  }
}
