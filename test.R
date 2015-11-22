## library(doParallel)
## library(Hmisc)
source("functions.R")
set.seed(96)
data <- preProcessData()
model <- trainModel(data$training)
testModel(model, data$testing)
saveModel(model)

## pred <- predict(model, data$testing)
## testing <- read.csv("pml-testing.csv")
## testing <- cleanData(testing)
## testing <- doCenteringScaling(testing)
## predTesting <- predict(model, testing)

## predTesting <- as.character(predTesting)
## createFiles(predTesting)

