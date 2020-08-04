# Set working directory to location of current file.
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(extrafont)
library(reshape2)
loadfonts()

data <- read.csv(file="polar_bear_updating_probdist.csv", header=TRUE, sep=",")

data_long <- melt(data)
names(data_long) <- c('animal', 'question', 'value')



ggplot(data_long, aes(x=animal, y=value)) +
  geom_bar(stat='identity') +
  facet_wrap( ~ question , ncol=1) +
  NULL
  
# TODO: rotate x axis labels
