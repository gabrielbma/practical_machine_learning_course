library(caret)
set.seed(0)
pmlTraining <- read.csv("pml-training.csv")
pmlTesting <- read.csv("pml-testing.csv")
head(pmlTraining$classe)
#trControl = trainControl(method = "cv")
trControl = trainControl(method = "cv", preProcOptions = list(pcaComp=15))
rfModel <- train(classe~., data=pmlTraining, method="rf", preProcess=c("nzv", "center", "scale", "pca"), prox=TRUE, trControl = trControl)
rfModel
nearZeroVar(pmlTraining, saveMetrics=T)

pred <- predict(rfModel, pmlTesting[,-nzv(pmlTraining)])

trControl2 = trainControl(method = "cv")
rfModel2 <- train(classe~., data=pmlTraining, method="rf", prox=TRUE, trControl = trControl2)

#pmlTesting <- read.csv("pml-testing.csv")



pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers = rep("A", 20)
pml_write_files(answers)
