library(dplyr)
library(data.table)
library(jsonlite)
library(purrr)

gs_profile = as.list(readLines("raw_data/gs_profiles.jsonl"))
gs_profile = map(gs_profile, parse_json)

parse_user_profile = function(user_profile){
  user_id = unlist(strsplit(user_profile$url, split = "="))[2]
  user_id = unlist(strsplit(user_id, split = "AAAAJ"))[1]
  user_id = paste0(user_id, "AAAAJ")
  
  cat(paste0("Parsing user ", user_id, "...\n"))
  
  name = user_profile$name
  title = user_profile$brief
  coauthors = paste(unlist(map(user_profile$co_authors, function(x){x[1]})), collapse = "; ")
  tags = paste(unlist(user_profile$tags), collapse = "; ")
  citation_stats = unlist(strsplit(user_profile$cited_stats, split = "\t"))
  citation_number = citation_stats[4]
  citation_number_since2018 = unlist(strsplit(citation_stats[5], split = "\n"))[1]
  h_index = citation_stats[6]
  h_index_since2018 = unlist(strsplit(citation_stats[7], split = "\n"))[1]
  i10_index = citation_stats[8]
  i10_index_since2018 =  unlist(strsplit(citation_stats[9], split = "\n"))[1]
  publication_number = length(unlist(user_profile$articles))
  publication_titles = paste(user_profile$articles, collapse = "; ")
  
  
  user_profile = data.frame(
    user_id, 
    name, title,
    coauthors,
    tags,
    citation_number,
    citation_number_since2018,
    h_index, 
    h_index_since2018,
    i10_index,
    i10_index_since2018,
    publication_number,
    publication_titles
  )
  
  return(user_profile)
}

user_profiles = map(gs_profile, parse_user_profile) %>%
  reduce(rbind.data.frame)

user_profiles = user_profiles %>%
  mutate(citation_number = as.numeric(citation_number)) %>%
  mutate(citation_number_since2018 = as.numeric(citation_number_since2018)) %>%
  mutate(h_index = as.numeric(h_index)) %>%
  mutate(h_index_since2018 = as.numeric(h_index_since2018)) %>%
  mutate(i10_index = as.numeric(i10_index)) %>%
  mutate(i10_index_since2018 = as.numeric(i10_index_since2018))

fwrite(user_profiles, file = "gscholar_profiles.csv")
