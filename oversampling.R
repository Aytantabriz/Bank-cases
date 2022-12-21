library(tidyverse)
library(car)
path <- dirname(getSourceEditorContext()$path)
setwd(path)

data <- read_delim("bank-full.csv",delim = ";")

data$y %>% table() %>% prop.table() %>% round(2)

data$y <- data$y %>% recode(" 'yes'=1 ; 'no'=0 ") %>% as_factor()


# Handling Class Imbalance
library(ROSE)

df <- ovun.sample(y ~ ., data, method = "over", N = 39922*2)$data
 
df$y %>% table()

