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

file_path <- paste0("output/feeds", Sys.Date(), ".txt")


write.table(df_feed, file_path, sep = "\t", row.names = FALSE)
