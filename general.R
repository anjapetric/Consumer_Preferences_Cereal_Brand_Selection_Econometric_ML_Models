install.packages("dplyr")
install.packages("mlogit")
install.packages("openxlsx")
install.packages("effects")
install.packages("caret")
install.packages("e1071")  
install.packages("recipes")
install.packages("yardstick")
install.packages("ggplot2")
install.packages("reshape2")

library(mlogit)
library(caret)
library(e1071)
library(recipes)
library(dplyr)
library(tidyr)
library(car)
library(openxlsx)
library(effects)
library(yardstick)
library(ggplot2)
library(reshape2)

load("/Users/irynamatsiuk/Downloads/purchaseonly_long_data_RData 2/purchaseonly_long_datasetQ2.RData")
data_long = data.frame(temp_purchaseonly_datalong)

data_long$Y = as.numeric(data_long$brand == data_long$brandbought)
cleaned_data <- data_long %>%
  mutate(brandlag = ifelse(is.na(brandlag), 0, brandlag))
#cleaned_data$Y <- as.logical(cleaned_data$Y)  # Ensure Y is logical
cleaned_data$lagged_choice = as.numeric(cleaned_data$brand == cleaned_data$brandlag)


replace_zeros <- function(vec) {
  zero_positions <- which(vec == 0)
  for (i in zero_positions) {
    vec[i] <- vec[max(which(vec[1:i] != 0))]
  }
  return(vec)
}

cleaned_data <- cleaned_data %>%
  group_by(panelist) %>%
  arrange(id) %>%  
  mutate(
    non_zero_brandlag = lag(brandbought, 6),  # Lag by 6 positions
    non_zero_brandlag = replace_zeros(non_zero_brandlag)  # Replace zeros with last non-zero value
  ) %>%
  ungroup()  

#replace na with brandlag given 
cleaned_data <- cleaned_data %>%
  mutate(non_zero_brandlag = ifelse(is.na(non_zero_brandlag), brandlag, non_zero_brandlag))

cleaned_data$lagged_choice_non_zero = as.numeric(cleaned_data$brand == cleaned_data$non_zero_brandlag)

#Create brand loyalty

cleaned_data <- cleaned_data %>%
  group_by(panelist, brand) %>%
  mutate(group = paste(panelist, brand, sep = "_")) %>%
  arrange(id) %>%  
  mutate(
    weights = exp(-(n() - row_number()) / 50),  
    weighted_cumsum = cumsum(lag(Y, default = 0) * lag(weights, default = 0)),
    weighted_cumcount = cumsum(lag(weights, default = 0)),
    brand_loyalty = ifelse(row_number() == 1, 0, weighted_cumsum / weighted_cumcount)
  ) %>%
  ungroup()




#CREATE WEEK_GROUP
cleaned_data <- cleaned_data %>%
  mutate(week_group = cut(week, breaks = seq(min(week), max(week) + 12, by = 12), labels = FALSE))


baseline_data <- subset(cleaned_data, dis == 0 & feat == 0)
display_data <- subset(cleaned_data, dis == 1 & feat == 0)
feature_data <- subset(cleaned_data, dis == 0 & feat == 1)


# Calculate the baseline price for each brand
P_baseline_by_brand <- baseline_data %>%
  group_by(brand, week_group) %>%
  summarise(P_baseline = mean(price, na.rm = TRUE), .groups = 'drop')

P_feature_by_brand <-feature_data %>%
  group_by(brand, week_group) %>%
  summarise(P_feature = mean(price, na.rm = TRUE), .groups = 'drop')

P_display_by_brand <- display_data %>%
  group_by(brand, week_group) %>%
  summarise(P_display = mean(price, na.rm = TRUE), .groups = 'drop')


merged_data <- P_baseline_by_brand %>%
  left_join(P_display_by_brand, by = c("brand", "week_group")) %>%
  left_join(P_feature_by_brand, by = c("brand", "week_group"))
# left_join(P_feat_dis_by_brand, by = "brand")


# Calculate the deltas
merged_data <- merged_data %>%
  mutate(delta_D = (P_baseline - P_display) / P_baseline,
         delta_F = (P_baseline - P_feature) / P_baseline)
#delta_FD = (P_baseline - P_feat_display) / P_baseline)

merged_data <- merged_data %>% replace_na(list(delta_D = 0, delta_F = 0))

cleaned_data <- merged_data %>% left_join(cleaned_data, by = c("brand", "week_group") )

cleaned_data = cleaned_data[,!(names(cleaned_data) %in% c("P_display", "P_feat_display", "P_feature" ))]

cleaned_data$P_effective <- cleaned_data$price * (1 - cleaned_data$delta_D * cleaned_data$dis) * (1 - cleaned_data$delta_F * cleaned_data$feat) 

cleaned_data$discount = cleaned_data$P_baseline - cleaned_data$P_effective

cleaned_data <- cleaned_data[, -which(names(cleaned_data) %in% c("group", "P_effective", 'weighted_cumcount', 'weighted_cumsum', 'weights'))]

write.xlsx(cleaned_data, "/Users/irynamatsiuk/Desktop/data_purchase_long_Iryna.xlsx")

mlogit_data <- mlogit.data(cleaned_data, choice = "Y", shape = "long", alt.var = "brand", chid.var = "id")

# Specify the model with alternative-specific coefficients
model <- mlogit(Y ~  1 | dis + feat + price + discount + lagged_choice_non_zero + FMYSize + Income + brand_loyalty, data = mlogit_data, reflevel = "2")

#lagged_choice_non_zero 
#model <- mlogit(Y ~  0 + brand + price:brand + dis:brand + feat:brand| FMYSize + Income, data = mlogit_data)
#model <- mlogit(Y ~  1 | dis + feat + price + FMYSize + Income , data = mlogit_data, reflevel = "1")

# Print the AIC value

summary(model)
aic_value <- AIC(model)
print(aic_value)


# Step 1: Fit multinomial logit model without lagged_utility
#prelim_model <- mlogit(Y ~ price + dis + feat + discount | FMYSize + Income, data = mlogit_data)
prelim_model <- mlogit(Y ~  1 | dis + feat + discount + price + FMYSize + Income + lagged_choice_non_zero + brand_loyalty, data = mlogit_data, reflevel = "2")

predicted_utility = predict(prelim_model, newdata = mlogit_data, type = "utilities")
predicted_utility_vector <- as.vector(predicted_utility)

#predicted_utility <- predicted_utility[, 1]  # Extract the first column as a vector
#str(predicted_utility_vector)
#length(predicted_utility)
cleaned_data$predicted_utility <- predicted_utility_vector 

# Compute lagged_utility within each group
cleaned_data <- as.data.frame(cleaned_data)

cleaned_data <- cleaned_data %>%
  arrange(id) %>%
  group_by(panelist, brand) %>%
  mutate(lagged_utility = lag(predicted_utility)) %>%
  ungroup()



cleaned_data <- cleaned_data %>%
  mutate(lagged_utility = replace_na(lagged_utility, 0))


mlogit_data <- mlogit.data(cleaned_data, choice = "Y", shape = "long", alt.var = "brand", chid.var = "id")

# Refit multinomial logit model with lagged_utility
final_model <- mlogit(Y ~ 1 | price + dis + feat + FMYSize + Income + lagged_choice_non_zero + discount + brand_loyalty + lagged_utility  
  , data = mlogit_data)
final_model_2 <- mlogit(Y ~ 1 | price + dis + feat + FMYSize + Income + lagged_choice_non_zero + discount + brand_loyalty + lagged_utility, 
                      data = mlogit_data,
                      reflevel = "2")
final_model_2 <- mlogit(Y ~ 1 | price + dis + feat + FMYSize + Income  , 
                        data = mlogit_data,
                        reflevel = "2")

summary(final_model_2)
summary(final_model_2)  
aic_value <- AIC(final_model_2)
print(aic_value)



# Define a range of price values
price_values <- seq(min(cleaned_data$price), max(cleaned_data$price), by = 0.02)

delta_intercept = coef(final_model_2)["(Intercept):1"] - coef(final_model_2)["(Intercept):4"] 
delta_price = coef(final_model_2)["price:1"] - coef(final_model_2)["price:4"] 
#delta_feature = coef(final_model_2)["feat:1"] - coef(final_model_2)["feat:4"] 
delta_display = coef(final_model)["dis:1"] - coef(final_model)["dis:4"] 

# Calculate the odds ratios for the range of price values
odds_ratios <- sapply(price_values, function(price_val) {
  exp(delta_intercept + delta_price * price_val + delta_display)
})

odds_ratios2 <- sapply(price_values, function(price_val) {
  exp(delta_intercept + delta_price * price_val)
})

odds_ratios3 <- sapply(price_values, function(price_val) {
  exp(delta_intercept + delta_price * price_val + coef(final_model_2)["dis:1"] * 0 - coef(final_model_2)["dis:4"] )
})

odds_ratios4 <- sapply(price_values, function(price_val) {
  exp(delta_intercept + delta_price * price_val + coef(final_model_2)["dis:1"]  - 0 * coef(final_model_2)["dis:4"] )
})



# Plot the odds ratio against the price
plot(price_values, odds_ratios, type = "l", col = "blue", lwd = 2, lty = 1,
     xlab = "Price", ylab = "Odds Ratio",
     main = "Odds Ratio of Choosing Category 1 over Category 4 as a Function of Price")
grid()

legend("topright", legend = c("Both on feature", "none on display", 'only 4 on display', 'only 1 on display'),
       col = c("blue", "red", "green", "purple"), lty = 1, lwd = 2)


# Adding a second line to the plot
lines(price_values, odds_ratios2, type = "l", col = "red", lwd = 2, lty = 2)
lines(price_values, odds_ratios3, type = "l", col = "green", lwd = 2, lty = 3)
lines(price_values, odds_ratios4, type = "l", col = "purple", lwd = 2, lty = 4)


#Random split
# Split the data
splits <- split_data(mlogit_data)
train_data <- splits$train
test_data <- splits$test

split_data <- function(data, train_ratio = 0.8) {
  unique_ids <- unique(data$id)
  set.seed(50)  # Set seed for reproducibility
  train_ids <- sample(unique_ids, size = round(train_ratio * length(unique_ids)))
  train_data <- data[data$id %in% train_ids, ]
  test_data <- data[!data$id %in% train_ids, ]
  return(list(train = train_data, test = test_data))
}

test_data[, "lagged_utility"] <- 0

#Addinf Utility lag

prelim_model <- mlogit(Y ~ 1 | price + dis + feat + discount + lagged_choice_non_zero + FMYSize + Income + lagged_utility + brand_loyalty, data = train_data)

# Extract coefficients
model_coefficients <- coef(prelim_model)

# Define a function to calculate utilities based on the coefficients
calculate_utilities <- function(data, coefficients, alternatives, ref_category) {
  utilities <- rep(NA, nrow(data))
  
  for (alt in alternatives) {
    print(alt)
    if (alt == ref_category) {
      # Utility for reference category is zero
      utilities[data$brand == alt] <- 0
    } else {
      alt_coeff <- coefficients[grepl(paste0(":", alt), names(coefficients))]
      intercept <- coefficients[paste0("(Intercept):", alt)]
      # Initialize the utility for the current alternative with the intercept
      utility <- intercept
      # Loop through each coefficient related to this alternative
      for (coef_name in names(alt_coeff)) {
        var_name <- sub(paste0(":", alt), "", coef_name)
        
        # Check if the variable exists in the test data
        if (var_name %in% names(data)) {
          utility <- utility + alt_coeff[coef_name] * data[[var_name]]
        } else {
          warning(paste("Variable", var_name, "not found in test data"))
        }
      }
      
      # Assign the computed utility to the appropriate rows in the utility vector
      utilities[data$brand == alt] <- utility
    }
  }
  
  return(utilities)
}

alternatives <- unique(test_data$brand)

ref_category <- "1"  

# Apply the function to the test data
predicted_utility_test <- calculate_utilities(test_data, model_coefficients, alternatives, ref_category)

test_data$predicted_utility_test <- predicted_utility_test

test_data <- as.data.frame(test_data)
#class(cleaned_data)
#class(predicted_utility)

test_data <- test_data %>%
  group_by(panelist, brand) %>%
  arrange(id) %>%
  
  mutate(lagged_utility = lag(predicted_utility)) %>%
  ungroup()


test_data <- test_data %>%
  mutate(lagged_utility = replace_na(lagged_utility, 0))


test_data <- mlogit.data(test_data, choice = "Y", shape = "long", alt.var = "brand", chid.var = "id")


# Fit the model on the training data
model <- mlogit(Y ~  1 | price + dis + feat + discount + FMYSize + Income + lagged_choice + lagged_utility + brand_loyalty, data = train_data)

# Predict on the test data
pred <- predict(model, newdata = test_data, type = "probabilities")

test_data <- as.data.frame(test_data)

actual_choices <- test_data %>%
  select(id, brand, Y) %>%
  pivot_wider(names_from = brand, values_from = Y, values_fill = 0) %>%
  arrange(id) %>%  # Sort by id
  select(-id) %>%
  as.matrix()

# Check dimensions of actual_choices
dim(actual_choices)


# Select the alternative with the highest predicted probability
pred_choice <- apply(pred, 1, which.max)
actual_choice <- test_data$brand[test_data$Y == 1]
accuracy <- mean(pred_choice == actual_choice)
print(confusionMatrix(as.factor(pred_choice), as.factor(actual_choice)))
print(paste("Accuracy:", accuracy))
#KFOLD CROSS VALIDATION
###################################################################################################


# function to calculate utilities based on the coefficients
calculate_utilities <- function(data, coefficients, alternatives, ref_category) {
  utilities <- rep(NA, nrow(data))
  
  for (alt in alternatives) {
    print(alt)
    if (alt == ref_category) {
      utilities[data$brand == alt] <- 0
    } else {
      alt_coeff <- coefficients[grepl(paste0(":", alt), names(coefficients))]
      intercept <- coefficients[paste0("(Intercept):", alt)]
      utility <- intercept
      for (coef_name in names(alt_coeff)) {
        var_name <- sub(paste0(":", alt), "", coef_name)
        
=        if (var_name %in% names(data)) {
          utility <- utility + alt_coeff[coef_name] * data[[var_name]]
        } else {
          warning(paste("Variable", var_name, "not found in test data"))
        }
      }
      
      # Assign the computed utility to the appropriate rows in the utility vector
      utilities[data$brand == alt] <- utility
    }
  }
  
  return(utilities)
}



#KFOLD CROSS VALIDATION

prepare_kfold_cv <- function(data, k = 5) {
  unique_ids <- unique(data$id)
  set.seed(50) 
  folds <- groupKFold(unique_ids, k = k)
  return(folds)
}

k <- 5
folds <- prepare_kfold_cv(mlogit_data, k)
unique_ids <- unique(mlogit_data$id)


# Perform cross-validation
accuracy_list <- vector("numeric", k)
confusion_matrices <- list()
brier_scores <- vector("numeric", k)


for (i in seq_len(k)) {
  train_ids <- unique_ids[folds[[i]]]
  train_data <- mlogit_data[mlogit_data$id %in% train_ids, ]
  test_data <- mlogit_data[!mlogit_data$id %in% train_ids, ]
  
  #ADDING UTILITY LAG
  test_data[, "lagged_utility"] <- 0
  #model to predict utilities
  prelim_model <- mlogit(Y ~ 1 | price + dis + feat + discount + lagged_choice_non_zero + FMYSize + Income + lagged_utility + brand_loyalty, data = train_data)
  model_coefficients <- coef(prelim_model)
  
  alternatives <- unique(test_data$brand)
    ref_category <- "1"  
  
  # Apply the function to the test data
  predicted_utility_test <- calculate_utilities(test_data, model_coefficients, alternatives, ref_category)
  
  test_data$predicted_utility_test <- predicted_utility_test
  
  test_data <- as.data.frame(test_data)
#shifting utilities 
    test_data <- test_data %>%
    group_by(panelist, brand) %>%
    arrange(id) %>%
    mutate(lagged_utility = lag(predicted_utility)) %>%
    ungroup()
  
    test_data <- as.data.frame(test_data)
    
  test_data <- test_data %>%
    mutate(lagged_utility = replace_na(lagged_utility, 0))
  
  actual_choices <- test_data %>%
    select(id, brand, Y) %>%
    pivot_wider(names_from = brand, values_from = Y, values_fill = 0) %>%
    arrange(id) %>%  # Sort by id
    select(-id) %>%
    as.matrix()
  
  
  test_data <- mlogit.data(test_data, choice = "Y", shape = "long", alt.var = "brand", chid.var = "id")
  
  # Fit the model on the training data
  model <- mlogit(Y ~  1 |  price + dis + feat + discount + FMYSize + Income + lagged_choice + lagged_utility + brand_loyalty, data = train_data)
  
  # Predict on the test data
  pred <- predict(model, newdata = test_data, type = "probabilities")
  
  
  # Select the alternative with the highest predicted probability
  pred_choice <- apply(pred, 1, which.max)
  actual_choice <- test_data$brand[test_data$Y == 1]
  
  # Calculate accuracy
  accuracy <- mean(pred_choice == actual_choice)
  accuracy_list[i] <- accuracy
  
  # Create confusion matrix
  confusion_matrix <- confusionMatrix(as.factor(pred_choice), as.factor(actual_choice))
  confusion_matrices[[i]] <- confusion_matrix
  
  # Calculate Brier score for each class and average
  brier_score <- mean(rowSums((pred - actual_choices)^2))
  brier_scores[i] <- brier_score
  
  # Print results for this fold
  print(paste("Fold", i, "Brier Score:", brier_score))


  # Print results for this fold
  print(paste("Fold", i, "Accuracy:", accuracy))
  print(confusion_matrix)
}

# Print average accuracy across all folds
average_accuracy <- mean(accuracy_list)
average_brier_score <- mean(brier_scores)
sd_accuracy <- sd(accuracy_list)
sd_brier_score <- sd(brier_scores)
extracted_matrices <- lapply(confusion_matrices, function(cm) {
  return(cm$table)
})
#matrix = Reduce("+", extracted_matrices)
#autoplot(matrix, type = "heatmap") +
 # scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")

summed_matrix_melt <- melt(matrix)
colnames(summed_matrix_melt) <- c("Actual", "Predicted", "value")

# Create the heatmap

ggplot(summed_matrix_melt, aes(x = Actual, y = Predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = 'red') +
  labs(title = "General Logit", x = "Prediction", y = "Reference") +
  theme_minimal()

str(confusion_matrices)
print(matrix)
print(paste("Average Accuracy, SD:", average_accuracy, sd_accuracy))
print(paste("Average Brier Score, SD:", average_brier_score, sd_brier_score))

