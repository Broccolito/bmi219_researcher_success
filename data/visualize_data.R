library(dplyr)
library(data.table)
library(ggplot2)
library(ggpubr)
library(gridExtra)

# Read the data
d = fread("gscholar_profiles.csv")

# Plot 1: Citation Number vs. Citation Number Since 2018
plt1 = ggplot(data = d, aes(y = citation_number, x = citation_number_since2018)) + 
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  scale_x_continuous(labels = scales::scientific) + 
  scale_y_continuous(labels = scales::scientific) + 
  ggtitle("Citation Number vs. Citation Number Since 2018") +
  xlab("Citation Number Since 2018") +
  ylab("Citation Number")

# Plot 2: H-Index vs. H-Index Since 2018
plt2 = ggplot(data = d, aes(y = h_index, x = h_index_since2018)) + 
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("H-Index vs. H-Index Since 2018") +
  xlab("H-Index Since 2018") +
  ylab("H-Index")

# Plot 3: i10-Index vs. i10-Index Since 2018
plt3 = ggplot(data = d, aes(y = i10_index, x = i10_index_since2018)) + 
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("i10-Index vs. i10-Index Since 2018") +
  xlab("i10-Index Since 2018") +
  ylab("i10-Index")

# Plot 4: Citation Number vs. H-Index
plt4 = ggplot(data = d, aes(y = citation_number, x = h_index)) + 
  geom_point() +
  scale_x_continuous(labels = scales::scientific) + 
  scale_y_continuous(labels = scales::scientific) +
  ggtitle("Citation Number vs. H-Index") +
  xlab("H-Index") +
  ylab("Citation Number")

plt = grid.arrange(plt1, plt4, plt2, plt3)

ggsave(filename = "analysis_result.png", plot = plt, dpi = 1200, width = 10, height = 5)
