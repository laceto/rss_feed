library(rvest)
library(xml2)
library(XML)
library(dplyr)


rss_path_cnbc <- c(
  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15837362",
  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19832390",
  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19794221"
  
)


# rss_path <- rss_path[1]
createdffromxml <- function(rss_path){
  data <- xml2::read_xml(rss_path)
  
  # Parse the food_data into R structure representing XML tree
  data_xml <- XML::xmlParse(data)
  
  df <- XML::xmlToDataFrame(nodes=XML::getNodeSet(data_xml, "//item"))

}


df_feed <- rss_path_cnbc %>% 
  purrr::map(createdffromxml) %>% 
  dplyr::bind_rows()

if (!dir.exists("output")) {
  dir.create("output")
}

file_path <- "output/feeds.txt"
df_existing <- read.table(file_path, sep = "\t", header = TRUE)

df_existing$guid <- as.character(df_existing$guid)
df_feed$guid     <- as.character(df_feed$guid)
df_existing$id <- as.character(df_existing$id)
df_feed$id     <- as.character(df_feed$id)

ncol(df_feed)

df_all <- bind_rows(df_existing, df_feed) %>%
  distinct()


# Write back to the same file
write.table(df_all, file_path, sep = "\t", row.names = FALSE)