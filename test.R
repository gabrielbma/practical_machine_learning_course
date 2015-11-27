## library(doParallel)
## library(Hmisc)
source("functions2.R")
set.seed(96)
data <- preProcessData()

modelRF <- trainModelRF(data$training)
testModel(modelRF, data$testing)
saveModel("ModelRF-",modelRF)

modelC50 <- trainModelC50(data$training)
testModel(modelC50, data$testing)
saveModel("ModelC50-",modelC50)

modelAdaBoost <- trainAdaBoost(data$training)
testModel(modelAdaBoost, data$testing)
saveModel("ModelAdaBoost-",modelAdaBoost)

modelGBM <- trainModelGBM(data$training)
testModel(modelGBM, data$testing)
saveModel("ModelGBM", modelGBM)
predict(modelGBM, testing)

modelKNN <- trainModelKNN(data$training, data$testing)


#model <- loadModel("model-RandomForest- 22-11-2015-T20-24-53")
confusionMatrix(predict(modelRF,data$testing), data$testing$classe)
### Combining predictors
predDF <- data.frame(model1=predict(modelGBM, data$testing), 
                     model2=predict(modelRF, data$testing), 
                     classe=data$testing$classe)
trControl <- trainControl(method = "repeatedcv", number = 10,
                          repeats = 3, 
                          returnResamp = "all",
                          verboseIter = TRUE,
                          classProbs = TRUE, sampling="down")
combModel <- train(classe~., method="glmnet", data=predDF, trControl=trControl)

predValid <- data.frame(model1=predict(modelGBM, data$validation), 
                        model2=predict(modelRF, data$validation))
combPredValid <- predict(combModel, predValid)
confusionMatrix(combPredValid, data$validation$classe)
predict(combModel, testing)


testingAssignment <- data.frame(model1=predict(modelGBM, testing),
                                model2=predict(modelRF, testing))
predict(combModel, testingAssignment)


pred <- predict(modelRF, data$testing)
testing <- read.csv("pml-testing.csv")
testing <- cleanData(testing)
testing <- doCenteringScaling(testing)
predTesting <- predict(modelRF, testing)


E A A E A E D D A E B C D A E E E B E B # Comb GLMNET: ModelGBM e RF
E B A E C B D D A E B C B A E E E B E B
E A A E A E D D A E B C D A E E E B E B
E A A E A E D D A E B A D A E D E B E B
A A A A A A A A A A A A A A A A A A A A
X A X A A X X X A X B C X A E E X B X B

## predTesting <- as.character(predTesting)
createFiles(predTesting)

## train(trainX, trainY, 
##       method = "C5.0", 
##       trControl = cctrl3,
##       tuneGrid = data.frame(trials = 10, model = "tree", winnow = FALSE),
##       metric = "ROC", 
##       preProc = c("center", "scale"))

## grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )

## mdl<- train(x=x,y=y,tuneGrid=grid,trControl=fitControl,method="C5.0",verbose=FALSE)
