library(readxl)
library(reshape)
library(openxlsx)
library(pheatmap)
library(ggplot2)
library(tidyverse)

rm(list = ls())

long_data <- read_excel("C:\\Users\\Arbeitsaccount\\Downloads\\data_purchase_long_Iryna.xlsx")
long_data <- data.frame(long_data)
long_data <- subset(long_data, select = -c(week_group, P_baseline, delta_D, delta_F, Y))

wide_data1 <- long_data
wide_data1 <- subset(wide_data1, select = -c(panelist, week, day, store.id, brandbought, sumunits, sumvol, sumdollars, trips, weekslast, brandlag, FMYSize, Income, NoIncome, non_zero_brandlag))
wide_data1 <- reshape(wide_data1, idvar = c("id"), timevar = "brand", direction = "wide", sep = "_")

wide_data2 <- long_data
wide_data2 <- subset(wide_data2, select = -c(price, dis, feat, lagged_choice, lagged_choice_non_zero, brand_loyalty, discount))
wide_data2 <- subset(wide_data2, brand == 1)
wide_data2 <- subset(wide_data2, select = -c(brand))

wide_data <- merge(wide_data1, wide_data2, by = "id", all = TRUE)
write.xlsx(wide_data, file = "C:\\Users\\Arbeitsaccount\\Downloads\\data_purchase_wide_Dominik.xlsx", rowNames = FALSE)


mat1 <- c(68, 245, 4, 0, 0, 0,
          41, 329, 3, 0, 0, 0,
          7, 113, 3, 0, 0, 1,
          8, 82, 0, 0, 0, 0,
          2, 22, 1, 0, 0, 0,
          34, 10, 2, 0, 0, 3)

mat2 <- c(187, 113, 4, 10, 2, 1,
          123, 214, 22, 10, 3, 1,
          54, 45, 19, 4, 2, 0,
          45, 23, 7, 13, 2, 0,
          4, 12, 1, 2, 6, 0,
          6, 23, 7, 0, 0, 13)

mat3 <- c(136, 79, 67, 22, 13, 0,
          93, 173, 76, 22, 9, 0,
          33, 23, 54, 5, 9, 0,
          19, 18, 36, 12, 5, 0,
          2, 3, 5, 3, 12, 0,
          8, 24, 15, 1, 1, 0)

mat4 <- c(138, 123, 13, 41, 1, 1,
          62, 222, 23, 60, 5, 1,
          40, 39, 23, 18, 4, 0,
          23, 18, 7, 42, 0, 0,
          4, 6, 0, 8, 7, 0,
          3, 10, 16, 2, 0, 18)

mat5 <- c(152, 155, 2, 5, 1, 2,
          86, 269, 6, 5, 0, 7,
          47, 63, 8, 3, 0, 3,
          45, 35, 1, 6, 0, 3,
          3, 21, 0, 1, 0, 0,
          2, 11, 1, 1, 0, 34)

mat <- mat1 + mat2 + mat3 + mat4 + mat5
#mat <- 2 * mat

confusion_matrix <- matrix(mat, nrow = 6, byrow = TRUE)

# Convert to tibble, add row identifier, and shape "long"
dat2 <- confusion_matrix %>%
  as.data.frame() %>%
  rownames_to_column("Reference") %>%
  pivot_longer(-Reference, names_to = "Prediction", values_to = "value") %>%
  mutate(
    Reference = factor(Reference, levels = 1:6),
    Prediction = factor(gsub("V", "", Prediction), levels = 1:6)
  )

# Plotting with ggplot2
ggplot(dat2, aes(Reference, Prediction)) +
  geom_tile(aes(fill = value)) +
  geom_text(aes(label = round(value, 1))) +
  scale_fill_gradient(low = "white", high = "red") +
  labs(x = "Reference", y = "Prediction", title = "ADA Boost and Venn Predictors") +
  theme(plot.title = element_text(hjust = 0.5))


