
### packages

library(quanteda)
library(readtext)
library(stopwords)
library(stringr)
library(readxl)

### word vectors
VECTORS_TXT = NULL
vec <- t(read.table(VECTORS_TXT, header = TRUE))
rn <- row.names(vec)

### base data
TEXTDIR_TSV = NULL
annot <- read.csv(TEXTDIR_TSV, sep = "\t")

### from python df7

TEXTDIR_NA = NULL
TEXTDIR = NULL
df7_dropna <- read.csv(TEXTDIR_NA)
df7 <- read.csv(TEXTDIR)


###

depi_data <- data.frame(id = 1:length(annot$clean_post), text = annot$clean_post)
depi_corpus <- corpus(depi_data, text_field = "text")

depi_toks <- tokens(depi_corpus, remove_numbers = FALSE, remove_punct = TRUE,
                    remove_symbols = TRUE, remove_separators = TRUE, split_hyphens = TRUE)

depi_toks_low <- lapply(depi_toks, tolower)

unique_chars <-unique(unlist(sapply(depi_toks_low, function(x){unique(unlist(strsplit(unlist(x), split = "")))})))

nonalphanum_chars <- unique_chars[!((unique_chars%in%c(letters))|(unique_chars%in%as.character(c(0,1,2,3,4,5,6,7,8,9))))]

for(i in 1:length(depi_toks_low)){
  for(j in 1:length(depi_toks_low[[i]])){
    depi_toks_low[[i]][j] <- paste(
      sapply(unlist(strsplit(depi_toks_low[[i]][j], split = "")),
             function(x){if(x %in% nonalphanum_chars){""}else{x}}),
      collapse = "")
  }
}

###

length(df7$id)
length(annot$id)

###


add_word_vectors <- function(ids, vec, rn, depi_toks_low, FUN, ...){
  mean_vectors <- matrix(NA, length(ids), ncol(vec))
  print(length(ids))
  for(i in 1:length(ids)){
    mean_vectors[i,] <- apply(vec[depi_toks_low[[which(annot$id == ids[i])]] %in% rn,], 2, FUN, ...)
    print(i)
  }
  return(mean_vectors)
}

df7_mean_vectors <- add_word_vectors(df7$id, vec, rn, depi_toks_low, mean, na.rm = TRUE)
df7_sum_vectors <- add_word_vectors(df7$id, vec, rn, depi_toks_low, sum)

df7_mean_vectors <- cbind(df7$id,df7_mean_vectors)
df7_sum_vectors <- cbind(df7$id,df7_sum_vectors)

colnames(df7_mean_vectors) <- c("id",sapply(1:100, function(x){paste(c("mean_vec_",x), collapse = "")}))
colnames(df7_sum_vectors) <- c("id",sapply(1:100, function(x){paste(c("sum_vec_",x), collapse = "")}))


###

write.csv(df7_mean_vectors, "df7_mean_vectors.csv", row.names = FALSE)
write.csv(df7_sum_vectors, "df7_sum_vectors.csv", row.names = FALSE)

