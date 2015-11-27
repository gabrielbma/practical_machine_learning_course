library(caret)

## Fold01.Rep3: mtry= 2 22:00

## @knitr acquireDataFunction
acquireData <- function(){
    read.csv("pml-training.csv")
}

## @knitr cleanDataFunction
cleanData <- function(data) {
    testing <- read.csv("pml-testing.csv")
    removedTestingColumns <- apply(testing, 2, function(x) any(is.na(x)))
    removedTrainingColumns <- apply(data, 2, function(x) any(is.na(x)))
    removedColumns <- removedTestingColumns | removedTrainingColumns
    kurtosisIndex <- grep("kurtosis.*", names(data))
    skewnessIndex <- grep("skewness.*", names(data))
    minmaxyawIndex <- grep("(max_yaw.*|min_yaw.*)", names(data))
    amplitudeIndex <- grep("amplitude.*", names(data))
    rawtimestampIndex <- grep("raw_timestamp.*", names(data))

    removedColumns[c("X", "user_name", "new_window", "num_window",
                     "cvtd_timestamp", "raw_timestamp_part_1",
                     "raw_timestamp_part_2",
                     kurtosisIndex,
                     skewnessIndex, minmaxyawIndex,
                     amplitudeIndex, rawtimestampIndex)] <- TRUE

    data[, !removedColumns]
}

## @knitr doCenteringScalingFunction
doCenteringScaling <- function(data){
    pp <- preProcess(x=data,
                     method = c("center", "scale"))
    predict(pp, data)
}

## @knitr createPartitionsFunction
createPartitions <- function(data) {
    inBuild <- createDataPartition(data$classe, p=0.7, list = FALSE)
    validation <- data[-inBuild,]
    buildData <- data[inBuild,]
    inTrain <- createDataPartition(buildData$classe, p=0.7, list=FALSE)
    list("training"=buildData[inTrain,], 
         "testing"=buildData[-inTrain,],
         "validation="=validation)
}

## @knitr preProcessDataFunction
preProcessData <- function() {
  data <- acquireData()
  data <- cleanData(data)
  data <- doCenteringScaling(data)
  createPartitions(data)
}

## @knitr trainModelRFFunction
trainModelRF <- function(training) {
    mtryGrid <- expand.grid(mtry = c(5, 10, 13))
    trControl <- trainControl(method = "repeatedcv", number=10, 
                              repeats = 3, 
                              verboseIter = TRUE, 
                              allowParallel = TRUE,
                              returnResamp = "all",
                              classProbs = TRUE,
                              sampling="down") 
    train(classe~., data=training, method="rf", prox=TRUE, ntree=1000,
          nodesize=30, metric = "ROC", trControl = trControl, 
          varImp = TRUE, importance = TRUE, tuneGrid = mtryGrid)
}

trainModelKNN <- function(training, testing){
    knn(training, testing, testing$classe, prob=TRUE)
}

trainAdaBoost <- function(training){
    ## grid <- expand.grid(.iter = c(50, 100),
    ##                     .maxdepth = c(4, 8),
    ##                     .nu = c(0.1, 1))
    grid <- expand.grid(.mfinal=10,
                        .maxdepth = c(4, 8))

    trControl <- trainControl(method = "repeatedcv", number=10, 
                              repeats = 3, 
                              verboseIter = TRUE, 
                              allowParallel = TRUE,
                              returnResamp = "all",
                              classProbs = TRUE,
                              sampling="down") 
    train(classe~.,
          data = training,
          method = "AdaBag",
          trControl=trControl)
          ## tuneGrid=grid)

}
trainModelGBM <- function(training) {
    trControl <- trainControl(method = "repeatedcv", number = 10, 
                           returnResamp = "all",
                           verboseIter = TRUE,
                           classProbs = TRUE, sampling="down")
    train(classe~., data=training, method="gbm", metric="ROC", 
          verbose=TRUE,trControl=trControl)

}
trainModelC50 <- function(training) {
    trControl <- trainControl(method = "repeatedcv", number = 10, 
                           returnResamp = "all",
                           verboseIter = TRUE,
                           classProbs = TRUE, sampling="down")
    grid <- expand.grid(.model="")
    train(classe~., 
          data=training,
          method = "C5.0", 
          trControl = trControl,
          tuneGrid = data.frame(trials = 10, model = "tree", winnow = FALSE),
          metric = "ROC")
}

## @knitr trainModelGLMFunction
trainModelGLM <- function(training) {
    trControl <- trainControl(method = "repeatedcv", number=10, 
                              repeats = 3, 
                              allowParallel = TRUE,
                              returnResamp = "all") 
    train(classe~., data=training, method="glm", trainControl = trControl, preProcess=c("range", "pca"))
}

## @knitr trainModelSVMFunction
trainModelSVM <- function(training){
    train(classe~., data=training, method="lssvmPoly", preProcess=c("range","pca"))
}

## @knitr testModelFunction
testModel <- function(model, testing) {
  predictions <- predict(model, testing)
  confusionMatrix(predictions, testing$classe)
}

## @knitr buildModelFunction
buildModel <- function() {
  set.seed(96)
  data <- preProcessData()
  model <- trainModel(data$training)
  testModel(model, data$testing)
  saveModel("", model)
  model
}

## @knitr saveModelFunction
saveModel <- function(name, model) {
  saveRDS(model, file=paste(name, 
                            format(Sys.time(),"%d-%m-%Y-T%H-%M-%S")))
}

## @knitr loadModelFunction
loadModel <- function(file) {
  readRDS(file)
}

## @knitr createFilesFunction
createFiles <- function(predictions){
  n = length(predictions)
  for(i in 1:n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(predictions[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

