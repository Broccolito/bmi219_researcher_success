library(scholar)

# ids = list.files("raw_data/gs_htmls/", pattern = "html")
# ids = gsub(pattern = ".html", replacement = "", ids)
# ids = gsub(pattern = "profile_", replacement = "", ids)
# save(ids, file = "scholar_ids.rda")

load("scholar_ids.rda")

gscholar_list = list()
n = 1
for(id in ids){
  
  cat(paste0("Retrieving ", n, " out of ", length(ids), " scholar profiles...\r"))
  
  rm(gscholar_profiles, gscholar_citation, gscholar_publication)
  
  try({
    gscholar_profiles = lapply(id, get_profile)
    Sys.sleep(3)
    gscholar_citation = lapply(id, get_citation_history)
    Sys.sleep(3)
    gscholar_publication = lapply(id, get_publications)
    Sys.sleep(3)
  },silent = TRUE)
  
  while(any(
    c((length(gscholar_profiles[[1]])==1),
      length(gscholar_publication[[1]])==1)
  )){
    cat("Server not responding; retrying...\n")
    try({
      gscholar_profiles = lapply(id, get_profile)
      Sys.sleep(3)
      gscholar_citation = lapply(id, get_citation_history)
      Sys.sleep(3)
      gscholar_publication = lapply(id, get_publications)
      Sys.sleep(3)
    },silent = TRUE)
  }
  
  gscholar = list(
    id = gscholar_profiles[[1]]$id,
    name = gscholar_profiles[[1]]$name,
    affiliation = gscholar_profiles[[1]]$affiliation,
    total_citation = gscholar_profiles[[1]]$total_cites,
    citation_history = gscholar_citation[[1]],
    h_index = gscholar_profiles[[1]]$h_index,
    i10_index = gscholar_profiles[[1]]$i10_index,
    fields = gscholar_profiles[[1]]$fields,
    homepage = gscholar_profiles[[1]]$homepage,
    coauthors = gscholar_profiles[[1]]$coauthors,
    n_available = gscholar_profiles[[1]]$available,
    n_not_available = gscholar_profiles[[1]]$not_available,
    publications = gscholar_publication[[1]]
  )
  
  gscholar_list[[id]] = gscholar
  rm(gscholar)
  n = n + 1
  
}

save(gscholar_list, file = "gscholar_profiles.rda")

