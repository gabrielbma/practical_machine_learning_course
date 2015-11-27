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
    sampleClass <- createDataPartition(data$classe, p=0.6, list = FALSE)
    list("training"=data[sampleClass,], "testing"=data[-sampleClass,])
}

## @knitr preProcessDataFunction
preProcessData <- function() {
  data <- acquireData()
  data <- cleanData(data)
  data <- doCenteringScaling(data)
  createPartitions(data)
}

## @knitr trainModelFunction
trainModel <- function(training) {
    mtryGrid <- expand.grid(mtry = c(15, 20, 25, 30, 35, 40))
    trControl <- trainControl(method = "repeatedcv", number=10, 
                              repeats = 3, 
                              verboseIter = TRUE, 
                              allowParallel = TRUE,
                              returnResamp = "all",
                              classProbs = TRUE) 
    train(classe~., data=training, method="rf", prox=TRUE, ntree=500,
          nodesize=30, metric = "ROC", trControl = trControl, 
          varImp = TRUE, importance = TRUE, tuneGrid = mtryGrid)
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
  saveModel(model)
  model
}

## @knitr saveModelFunction
saveModel <- function(model) {
  saveRDS(model, file=paste("model-RandomForest-", 
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

