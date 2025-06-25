library(rvest)
library(xml2)
library(XML)


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

df_feed %>% 
  write.table('feeds.txt', sep = "\t", row.names = F)
