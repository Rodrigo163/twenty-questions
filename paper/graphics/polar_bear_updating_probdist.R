# Set working directory to location of current file.
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(extrafont)
library(reshape2)
loadfonts()

data <- read.csv(file="polar_bear_updating_probdist_subset.csv", header=TRUE, sep=",")

data_long <- melt(data)
names(data_long) <- c('animal', 'question', 'value')



ggplot(data_long, aes(x=animal, y=value)) +
  geom_bar(stat='identity') +
  facet_wrap( ~ question , ncol=1) + 
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=0.3),
        axis.text.y = element_blank(),
        text = element_text(family='Times New Roman')) +
  labs(y=element_blank(), x=element_blank()) +
  NULL
  
ggsave('updating.pdf', device=cairo_pdf, width=16, height=20, units='cm')
