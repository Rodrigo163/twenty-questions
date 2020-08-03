# Set working directory to location of current file.
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(extrafont)
loadfonts()

data <- read.csv(file="polar_bear_updating_probdist.csv", header=TRUE, sep=",")

