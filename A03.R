source('A01.R', echo = F)
library(vtreat)
library(leaps)
library(MASS)
library(lattice)
library(caret)
library(boot)
library(pROC)
#library(tidyverse)

#=====================================#
# Split Dataset into Train and Test  
#=====================================#

set.seed(123)
sample_size <- floor(0.75 * nrow(merged_data))
train_ind <- sample(seq_len(nrow(merged_data)), size = sample_size)
# Train
data_train <- merged_data[train_ind,]
# Test
data_test <- merged_data[-train_ind,]

#============================#
# RMSE Function  
#============================#

rmse <- function(predcol, ycol) {
  res <- predcol - ycol
  sqrt(mean(res^2))
}

#============================#
# RSquared Function  
#============================#
# function to obtain R-Squared from the data
rsq <- function(formula, data, indices) {
  d <- data[indices,] # allows boot to select sample
  fit <- glm(formula, family = 'binomial', data=d)
  return(summary(fit)$r.square)
}

#============================#
# Cross Validation Function  
#============================#

# Function to perform cross validation over a logistic regression model
# fmla - Formula of the model
# data - dataFrame of target data
# Returns a prediction vector for the model

nFold <- 10
nRows <- nrow(merged_data)
splitPlan <- kWayCrossValidation(nRows, nFold, NULL, NULL)
myCrossValidationFunction <- function (fmla, data) {
  predictionResult <- 0  # initialize the prediction vector
  for(i in 1:nFold) {
    split <- splitPlan[[i]]
    targetModel <- glm(fmla, family = "binomial", data = data[split$train, ])
    predictionResult[split$app] <- predict(targetModel, newdata = data[split$app, ], type = 'response')
  }
  return (predictionResult)
}


#==================================#
#  Logistic Regression model
#==================================#

# Find the defaulting probability of the average
averageDefaultProspect = mean(merged_data$Performance_Tag)


#====
# Backward Stepwise Model Selection
#====

model <- glm(Performance_Tag ~ . , family = binomial(link = 'logit'), data = data_train)
step_model <- step(model, trace = 0)
summary(step_model)

# Simple Train and Test Data Sampling
data_train$predictions <- predict(step_model, type = "response")
data_train$predictions <- ifelse(data_train$predictions > mean(data_train$Performance_Tag), 1, 0)
mean(data_train$Performance_Tag == data_train$predictions)
# 0.5984198

data_test$predictions <- predict(step_model, newdata = data_test, type = "response")
data_test$predictions <- ifelse(data_test$predictions > mean(data_test$Performance_Tag), 1, 0)
mean(data_test$Performance_Tag == data_test$predictions)
# 0.6082675

# 10-fold Cross Validation Sampling
# Extract step model formula
freeStepWiseFormula <- step_model$call[[2]]
merged_data$pred_freeStepwiseModel <- myCrossValidationFunction(freeStepWiseFormula, data = merged_data)

# Accuracy By Mean Default Prospect
modelPredBinarybyMean <- ifelse(merged_data$pred_freeStepwiseModel > averageDefaultProspect, 1, 0)
table(merged_data$Performance_Tag, modelPredBinarybyMean)
mean(merged_data$Performance_Tag == modelPredBinarybyMean)
# 0.5972175

ROC <- roc(merged_data$Performance_Tag, modelPredBinarybyMean)
auc(ROC)
# Area under the curve: 0.6251

# Accuracy By 0.5 Binary
modelPredMidProb <- ifelse(merged_data$pred_freeStepwiseModel > 0.5, 1, 0)
table(merged_data$Performance_Tag, modelPredMidProb)
mean(merged_data$Performance_Tag == modelPredMidProb)
# 0.9578180
ROC <- roc(merged_data$Performance_Tag, modelPredMidProb)
auc(ROC)
# Area under the curve: 0.5


#==============================================================#
# Limitations of Accuracy
# Despite the calculated high accuracy, the result is misleading 
# due to the rarity of the outcome being predicted.
# The accuracy would have been 95.7% (the default percentage) if a model had simply predicted "no default" for each record.
# The model is actually performing WORSE than if it were to predict non-default for every record.
#==============================================================#


#====
# Forward Stepwise Model Selection 
#====
null_model <- glm(Performance_Tag ~ 1, data = data_train, family = "binomial")
full_model <- glm(Performance_Tag ~ ., data = data_train, family = "binomial")
step_model_forward <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "forward", trace = 0)
summary(step_model_forward)

# 10-fold Cross Validation Sampling
# Extract step model formula
fwdSelectFormula <- step_model_forward$call[[2]]
merged_data$pred_fwdSelect <- myCrossValidationFunction(fwdSelectFormula, data = merged_data)

# Accuracy By Mean Default Prospect
modelPredBinarybyMean_fwd <- ifelse(merged_data$pred_freeStepwiseModel > averageDefaultProspect, 1, 0)
table(merged_data$Performance_Tag, modelPredBinarybyMean_fwd)
mean(merged_data$Performance_Tag == modelPredBinarybyMean_fwd)
# 0.5972175
ROC <- roc(merged_data$Performance_Tag, modelPredBinarybyMean_fwd)
auc(ROC)
# Area under the curve: 0.6251


# Accuracy By 0.5 
modelPredMidProb_fwd <- ifelse(merged_data$pred_freeStepwiseModel > 0.5, 1, 0)
table(merged_data$Performance_Tag, modelPredMidProb_fwd)
mean(merged_data$Performance_Tag == modelPredMidProb_fwd)
# 0.957818

ROC <- roc(merged_data$Performance_Tag, modelPredMidProb_fwd)
auc(ROC)
# Area under the curve: 0.5


#######################################################

# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)
# Train the model
step.model <- train(Performance_Tag ~., data = merged_data,
                    trControl = train.control,
                    method = "glm",
                    family=binomial())
step.model

####
#Generalized Linear Model 

# 69864 samples
# 32 predictor

# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 62878, 62877, 62878, 62878, 62878, 62877, ... 
# Resampling results:
  
#   RMSE       Rsquared    MAE       
# 0.1998038  0.01234167  0.07964126

#==================================#
#  PCA Model
#==================================#


