# INITIALIZATION PROCEDURE ####################################################

# load required packages
library(caret)
library(xgboost)
library(dplyr)
library(mlogit)

# clear everything from environment
rm(list = ls())
set.seed(0)

# load wide data set into a data frame
load("C:\\Users\\Arbeitsaccount\\Downloads\\wide_datasetQ2.RData")
df_wide <- data.frame(tempdatawide)

# load purchase only data set into a data frame
load("C:\\Users\\Arbeitsaccount\\Downloads\\purchaseonly_wide_datasetQ2.RData")
df_wide_class <- data.frame(temp_purchaseonly_datawide)

# load long data set into a data frame
load("C:\\Users\\Arbeitsaccount\\Downloads\\long_datasetQ2.RData")
df_long <- data.frame(tempdatalong)


# DESCRIPTIVE STATISTICS ######################################################

# compute summary statistics for all data points
df_household = aggregate(df_wide, list(df_wide$panelist), mean)
sum_stats_house <- cbind(colnames(df_household), sapply(df_household, mean),
                         sapply(df_household, sd), sapply(df_household, min),
                         sapply(df_household, max))

# compute summary statistics for each household
sum_stats <- cbind(colnames(df_wide), sapply(df_wide, mean), 
                   sapply(df_wide, sd), sapply(df_wide, min), 
                   sapply(df_wide, max))


# BINARY LOGIT MODEL ##########################################################

# prepare the dataset
df_wide$bought <- as.numeric(df_wide$brandbought > 0)
df_wide <- na.omit(df_wide)

# add additional variable
df_wide <- df_wide %>% group_by(panelist) %>% arrange(id) %>% 
  mutate(cumulative_bought = cumsum(bought), num_observations = row_number(), proportion_bought = cumulative_bought / num_observations)

# estimate coefficients of the model
binary_logit <- glm(bought ~ price_1 + price_2 + price_3 + price_4 + price_5 + price_6 + feat_1 + feat_2 + feat_3 + feat_4 + feat_5 + feat_6 + dis_1 + dis_2 + dis_3 + dis_4 + dis_5 + dis_6 + lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + weekslast + FMYSize + Income + cumulative_bought + proportion_bought + factor(day) + factor(store.id), family = binomial(), data = df_wide)
print(summary(binary_logit))
print(r2_mcfadden(binary_logit))

# prepare k-fold cross validation
k = 5
df_wide <- df_wide[sample(nrow(df_wide)),]
folds <- cut(seq(1,nrow(df_wide)),breaks=k,labels=FALSE)
complete_mean = 0

# perform k-fold cross validation
for(i in 1:k){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- df_wide[testIndexes, ]
  trainData <- df_wide[-testIndexes, ]
  
  binary_logit <- glm(bought ~ price_1 + price_2 + price_3 + price_4 + price_5 + price_6 + feat_1 + feat_2 + feat_3 + feat_4 + feat_5 + feat_6 + dis_1 + dis_2 + dis_3 + dis_4 + dis_5 + dis_6 + lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + weekslast + FMYSize + Income + cumulative_bought + proportion_bought + factor(day) + factor(store.id), family = binomial(), data = trainData)
  #print(summary(binary_logit))
  
  probbought <- predict(binary_logit, testData, type = "response")
  predbought <- as.numeric(probbought > 0.5)
  
  print(confusionMatrix(as.factor(predbought), as.factor(testData$bought)))
  print(mean(testData$bought == predbought))
  complete_mean <- complete_mean + mean(testData$bought == predbought)
}

print(complete_mean / 5)


# XGBOOST MODEL ###############################################################

# prepare the dataset
df_xgboost <- df_wide
df_xgboost <- subset(df_xgboost, select = -c(brand_0, brand_1, brand_2, brand_3, brand_4, brand_5, brand_6, brandbought, sumdollars, sumvol, sumunits))

# prepare k-fold cross validation
k = 2
df_xgboost <- df_xgboost[sample(nrow(df_xgboost)), ]
folds <- cut(seq(1, nrow(df_xgboost)), breaks = k, labels=FALSE)

# perform k-fold cross validation
for(i in 1:k) {
  testIndexes <- which(folds == i, arr.ind = TRUE)
  testData <- df_xgboost[testIndexes, ]
  trainData <- df_xgboost[-testIndexes, ]
  
  train_x = data.matrix(trainData[, -36])
  train_y = trainData[, 36]
  
  test_x = data.matrix(testData[, -36])
  test_y = testData[, 36]
  
  xgb_train = xgb.DMatrix(data = train_x, label = train_y)
  xgb_test = xgb.DMatrix(data = test_x, label = test_y)
  
  xg_boost = xgboost(data=xgb_train, max.depth=3, nrounds=50)
  print(summary(xg_boost))
  
  probbought <- predict(xg_boost, xgb_test, type = "response", objective = "binary:logistic")
  predbought <- as.numeric(probbought > 0.5)
  
  print(confusionMatrix(as.factor(predbought), as.factor(testData$bought)))
  print(mean(testData$bought == predbought))
  
  importance_matrix <- xgb.importance(model = xg_boost)
  print(importance_matrix)
  xgb.plot.importance(importance_matrix = importance_matrix)
}


# XGBOOST CLASSIFICATION MODEL ################################################

# prepare the dataset
df_xgboost_class <- df_wide_class
df_xgboost_class$bought <- df_xgboost_class$brandbought
df_xgboost_class <- subset(df_xgboost_class, select = -c(brand_0, brand_1, brand_2, brand_3, brand_4, brand_5, brand_6, brandbought, sumdollars, sumvol, sumunits))

# prepare k-fold cross validation
k = 2
df_xgboost_class <- df_xgboost_class[sample(nrow(df_xgboost_class)), ]
folds <- cut(seq(1, nrow(df_xgboost_class)), breaks = k, labels=FALSE)

# perform k-fold cross validation
for(i in 1:k) {
  testIndexes <- which(folds == i, arr.ind = TRUE)
  testData <- df_xgboost_class[testIndexes, ]
  trainData <- df_xgboost_class[-testIndexes, ]
  
  train_x = data.matrix(trainData[, -36])
  train_y = trainData[, 36]
  
  test_x = data.matrix(testData[, -36])
  test_y = testData[, 36]
  
  xgb_train = xgb.DMatrix(data = train_x, label = train_y)
  xgb_test = xgb.DMatrix(data = test_x, label = test_y)
  
  xg_boost_class = xgboost(data=xgb_train, max.depth=3, nrounds=50)
  print(summary(xg_boost_class))
  
  probbought <- predict(xg_boost_class, xgb_test, type = "response")
  probbought[(probbought > 6)] = 6
  probbought[(probbought < 1)] = 1
  predbought = round(probbought)
  
  print(confusionMatrix(as.factor(predbought), as.factor(testData$bought)))
  print(mean(testData$bought == predbought))
  
  importance_matrix <- xgb.importance(model = xg_boost_class)
  print(importance_matrix)
  xgb.plot.importance(importance_matrix = importance_matrix)
}



