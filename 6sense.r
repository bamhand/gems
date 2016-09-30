rm(list=ls())

library(dplyr)
library(magrittr)
library(tidyr)
library(doParallel)

train<-read.csv(file="c:/Users/f367573/Desktop/R/6sense/training.tsv", sep="")
train$id<-train[,1]
train$date<-train[,2]
train$action<-train[,3]
train<-train[,4:6]
train<-group_by(train, id, action) %>% summarize(count = n()) %>% spread(action, count)
train[is.na(train)]<-0
train$Purchase1<-as.integer(pmin(train$Purchase, 1))
train$Purchase1[train$Purchase1 == 1]<-"Purchase"
train$Purchase1[train$Purchase1 == 0]<-"NoPurchase"
train$Purchase1<-as.factor(train$Purchase1)

test<-read.csv(file="c:/Users/f367573/Desktop/R/6sense/test.tsv", sep="")
test$id<-test[,1]
test$date<-test[,2]
test$action<-test[,3]
test<-test[,4:6]
test<-group_by(test, id, action) %>% summarize(count = n()) %>% spread(action, count)
test[is.na(test)]<-0

library(caret)
library(glmnet)

gbm_ctrl <-trainControl(method="cv"
                        ,number = 4
                        ,repeats = 1
                        ,classProbs= T
                        ,allowParallel=T
                        ,verboseIter = F
                        ,summaryFunction = twoClassSummary)

gbm_tunegrid <- expand.grid(interaction.depth = (3:7)*3
                            ,n.trees = (10:15)*100
                            ,shrinkage = 0.01
                            ,n.minobsinnode = 10)

vars<-c("EmailClickthrough", "EmailOpen", "FormSubmit", "PageView", "WebVisit")

registerDoParallel(3)
# Start the clock!
ptm <- proc.time()
gbm_mdl <- train(as.formula(paste("~", paste(vars, collapse="+")))
                 ,y = train$Purchase1
                 ,data = train
                 ,method = 'gbm'
                 ,trControl = gbm_ctrl
                 ,tuneGrid = gbm_tunegrid
                 ,verbose = T
                 ,metric = "ROC")

proc.time() - ptm

gbm_VI <- varImp(gbm_mdl,scale=F)$importance
gbm_VI$var <- row.names(gbm_VI)
gbm_VI <- gbm_VI[order(gbm_VI$Overall,decreasing=T),]  

plot(gbm_mdl)

confusionMatrix(gbm_mdl)
test_predict <- predict(gbm_mdl, test, type = "prob")

library(nnet)
library(e1071)
library(Hmisc)

#5 fold cross validation for training
fitControl <- trainControl(
  method = 'repeatedcv',
  number = 4,
  repeats = 0)

#testing 10-20 notes with decaye of 0.001, single hidden layer
nngrid <- expand.grid(size=c(2:10), decay = 0.001)

trainnet = train(
  x=train[,c(vars)],
  y=train$Purchase1,
  method = 'nnet',
  verbose = F,
  tuneGrid = nngrid,
  trControl = fitControl)

plot(trainnet)

#confusion matrix for training and test data
table(predict(trainnet, train, type='raw'), train$Purchase1)

nnet_predict<-predict(trainnet, test, type='prob')

diff<-nnet_predict$Purchase-test_predict$Purchase
avg<-(nnet_predict$Purchase+test_predict$Purchase)/2

test$rank<-avg

sorted<-test[order(-test$rank),]

write.csv(sorted[1:1000,],file="6sense.csv")

confusionMatrix(trainnet)

train$clickopen<-train$EmailOpen*train$EmailClickthrough
train$clicksubmit<-train$EmailClickthrough*train$FormSubmit
train$clickview<-train$EmailClickthrough*train$PageView
train$clickvisit<-train$EmailClickthrough*train$WebVisit
train$opensubmit<-train$EmailOpen*train$FormSubmit
train$openview<-train$EmailOpen*train$PageView
train$openvisit<-train$EmailOpen*train$WebVisit
train$submitview<-train$FormSubmit*train$PageView
train$viewvisit<-train$PageView*train$WebVisit

lvars<-names(train[,!names(train) %in% c("id","CustomerSupport","Purchase","Purchase1")])

options(na.action="na.fail")
x = model.matrix(as.formula(paste("~", paste(lvars, collapse="+"))), train)

set.seed(12345)
registerDoParallel(3)
# Start the clock!
ptm <- proc.time()
lasso_lambda<-cv.glmnet(x=x,y=train$Purchase1,family="binomial", nfolds = 4, parallel = T)

proc.time() - ptm

plot(lasso_lambda)
lasso_lambda$lambda.min
lasso_lambda$lambda.1se

coef(lasso_lambda, s=lasso_lambda$lambda.min)
predict(lasso_lambda, x, type = "class", s=lasso_lambda$lambda.1se)
sum(predict(lasso_lambda, x, type = "class", s=lasso_lambda$lambda.1se)==train$Purchase1)
sum(predict(lasso_lambda, x, type = "class", s=lasso_lambda$lambda.1se)!=train$Purchase1)