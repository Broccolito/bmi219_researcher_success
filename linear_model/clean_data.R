library(dplyr)
library(data.table)
library(purrr)
library(stringr)

min_count = 5
d = fread("../data/gscholar_profiles.csv")

all_publication_titles = paste(d[["publication_titles"]], collapse = " ")
title_words = strsplit(all_publication_titles, "\\W+")[[1]]
title_word_frequency = table(title_words)
title_word_frequency = data.frame(
  word = names(title_word_frequency),
  count = as.numeric(title_word_frequency)
) %>%
  filter(count >= min_count) %>%
  arrange(desc(count))

word_freq_by_user = d %>%
  split(.$user_id) %>%
  map(function(x){
    tmp = paste(x$publication_titles, collapse = " ")
    tmp = strsplit(tmp, "\\W+")[[1]]
    tmp = table(tmp)
    tmp = data.frame(
      word = names(tmp),
      count = as.numeric(tmp)
    ) %>%
      arrange(desc(count))
    tmp = left_join(select(title_word_frequency, -count), tmp, by = "word") %>%
      mutate(count = ifelse(is.na(count),0, count)) %>%
      select(-word)
    names(tmp) = x$user_id[1]
    return(tmp)
  }) %>%
  reduce(cbind.data.frame)

word_freq_by_user = as.data.frame(t(word_freq_by_user))
names(word_freq_by_user) = title_word_frequency[["word"]]
word_freq_by_user[["user_id"]] = rownames(word_freq_by_user)
rownames(word_freq_by_user) = NULL

word_freq_hindex = inner_join(select(d, user_id, h_index_since2018), 
                              word_freq_by_user, by = "user_id")

fwrite(word_freq_hindex, file = "word_freq_hindex.csv")
