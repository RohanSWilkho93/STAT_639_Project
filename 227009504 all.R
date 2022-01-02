######################## Classification ############################
'Classification Problem'

######################## SVM ############################
print()
'SVM'
library(dplyr)
library(e1071)
library(ISLR)
library(caret)
library(mlr)
library(mlbench)
library(randomForest)

load(file ="class_data(1).RData") #variables x, y, x_new

set.seed(2021)
test_x<-xnew
train_x<-x
train_y<-y

names(x)
summary(x)
summary(y)

#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

cut<- 0.75
sig<-0.05

corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut) # setting cutoff
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

# ANOVA Selection
i = 0
vec <- vector() # vector for storing indices of statistically significant variables
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]
project.data.train = data.frame(anova.train.x, y=(train.y))

#RFE
rfe.x <- anova.train.x

control <- rfeControl(functions = rfFuncs, method = "repeatedcv", 
                      repeats = 5, number =10)
result_rfe1 <- rfe(x= rfe.x, y= as.factor(as.matrix(train.y)), 
                   sizes = c(1:38), rfeControl= control)


result_rfe1
rfe.predictors <- predictors(result_rfe1)

ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

rfe.train.x <- rfe.x[rfe.predictors]
rfe.train.x

project.data.train = data.frame(rfe.train.x, y=(train.y))

# Create a Learning Task
classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

#hypertunning
rdesc = makeResampleDesc("CV", iters = 10)#, reps = 5)
num_ps = makeParamSet(
  makeNumericParam("C", lower = -10, upper = 10, trafo = function(x) 10^x),
  makeNumericParam("sigma", lower = -10, upper = 10, trafo = function(x) 10^x)
)
lrn = makeLearner("classif.ksvm")
ctrl = makeTuneControlGrid()
tunedRF = tuneParams(lrn, task = classif.task, resampling = rdesc,
                     par.set = num_ps, control = ctrl, show.info = FALSE, 
                     measures = list(acc, setAggregation(acc, test.sd)))
tunedRF
tunedRF$x
tunedRF$y

rfTuningData <- generateHyperParsEffectData(tunedRF)


plotHyperParsEffect(rfTuningData, x = "C", y = "acc.test.mean",
                    plot.type = "line") + theme_bw()

plotHyperParsEffect(rfTuningData, x = "sigma", y = "acc.test.mean",
                    plot.type = "line") + theme_bw()

#training with hyperpars

tuned_RF = setHyperPars(makeLearner("classif.ksvm"), C = tunedRF$x$C, sigma = tunedRF$x$sigma)
mod_rf = train(tuned_RF, classif.task)
mod_rf

# train_accuracy
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 100,
                          stratify = TRUE)
kFoldCV <- resample(learner = tuned_RF, task = classif.task,
                    resampling = kFold, measures = list(mmce, acc))

predictions = predict(mod_rf, task = classif.task)
kFoldCV$aggr

predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

# test_accuracy
test.x = data.frame(select(test,-y))
test.x = test.x[rfe.predictors] #if ANOVA +RFE
test.y = select(test, y)
test.data = data.frame(test.x, y = (test.y))
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")
predictions = predict(mod_rf, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'SVM Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()

######################## Random Forest ############################
print()
'Random Forest'
library(dplyr)
library(e1071)
library(ISLR)
library(caret)
library(mlr)
library(mlbench)
library(randomForest)

load(file ="class_data(1).RData") #variables x, y, x_new

set.seed(2021)
test_x<-xnew
train_x<-x
train_y<-y

names(x)
summary(x)
summary(y)

#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

cut<- 0.9
sig<-0.05

corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut) # setting cutoff
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

# ANOVA Selection
i = 0
vec <- vector() # vector for storing indices of statistically significant variables
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]
project.data.train = data.frame(anova.train.x, y=(train.y))

rfe.x <- anova.train.x

control <- rfeControl(functions = rfFuncs, method = "repeatedcv", 
                      repeats = 5, number =10)
result_rfe1 <- rfe(x= rfe.x, y= as.factor(as.matrix(train.y)), 
                   sizes = c(1:35), rfeControl= control)

result_rfe1
rfe.predictors <- predictors(result_rfe1)

ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

rfe.train.x <- rfe.x[rfe.predictors]
rfe.train.x

project.data.train = data.frame(rfe.train.x, y=(train.y))

# Create a Learning Task
classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

#hypertunning
rdesc = makeResampleDesc("CV", iters = 10)#, reps = 5)
num_ps = makeParamSet(
  makeIntegerParam("mtry", lower = 1, upper = 9),
  makeDiscreteParam("ntree", c(250, 500, 1000, 1250, 1500, 1750, 2000))
)
lrn = makeLearner("classif.randomForest")
ctrl = makeTuneControlGrid()
tunedRF = tuneParams(lrn, task = classif.task, resampling = rdesc,
                     par.set = num_ps, control = ctrl, show.info = FALSE, 
                     measures = list(acc, setAggregation(acc, test.sd)))
tunedRF
tunedRF$x
tunedRF$y

rfTuningData <- generateHyperParsEffectData(tunedRF)


plotHyperParsEffect(rfTuningData, x = "ntree", y = "acc.test.mean",
                    plot.type = "line") +  theme_bw()

plotHyperParsEffect(rfTuningData, x = "mtry", y = "acc.test.mean",
                    plot.type = "line") +  theme_bw()

tuned_RF = setHyperPars(makeLearner("classif.randomForest"), mtry = tunedRF$x$mtry, ntree = tunedRF$x$ntree)
mod_rf = train(tuned_RF, classif.task)
mod_rf

# train_accuracy
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 100,
                          stratify = TRUE)
kFoldCV <- resample(learner = tuned_RF, task = classif.task,
                    resampling = kFold, measures = list(mmce, acc))

predictions = predict(mod_rf, task = classif.task)
kFoldCV$aggr

predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

# test_accuracy
test.x = data.frame(select(test,-y))
test.x = test.x[rfe.predictors] #if ANOVA +RFE
test.y = select(test, y)
test.data = data.frame(test.x, y = (test.y))
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")
predictions = predict(mod_rf, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'Random Forest Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()


######################## KNN ############################
print()
'K-NN'
library(ISLR)
library(caret)
library(mlr)
library(fda)
library(FNN)
library(kknn)
library(dplyr)

load(file ="class_data(1).RData") #variables x, y, x_new

set.seed(2021)
test_x<-xnew
train_x<-x
train_y<-y

names(x)
summary(x)
summary(y)

#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

cut<- 0.90 #or .75 could be used
sig<-0.05 #or .01 could be used

corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut) # setting cutoff
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

# ANOVA Selection
i = 0
vec <- vector() # vector for storing indices of statistically significant variables
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]

# RFE 
rfe.x <- anova.train.x
control <- rfeControl(functions = rfFuncs, method = "repeatedcv", 
                      repeats = 5, number =10)
result_rfe1 <- rfe(x= rfe.x, y= as.factor(as.matrix(train.y)), 
                   sizes = c(1:25), rfeControl= control)

print(result_rfe1)
ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()
project.data.train = data.frame(anova.train.x, y=(train.y))

rfe.predictors <- predictors(result_rfe1)
rfe.train.x <- rfe.x[rfe.predictors]
rfe.train.x

project.data.train = data.frame(rfe.train.x, y=(train.y))

classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 1:20))
gridSearch <- makeTuneControlGrid()
cvForTuning <- makeResampleDesc("RepCV", folds = 10, reps = 50)
tunedK <- tuneParams("classif.knn", task = classif.task,
                     resampling = cvForTuning,
                     par.set = knnParamSpace,
                     control = gridSearch)
tunedK
tunedK$x
knnTuningData <- generateHyperParsEffectData(tunedK)

plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean",
                    plot.type = "line") +
  theme_bw()
tunedK$x

#confirming with chart, set k=5
tunedKnn <- setHyperPars(makeLearner("classif.knn"), par.vals = list("k" = 5))
mod_knn = train(tunedKnn, classif.task)
mod_knn

predictions = predict(mod_knn, task = classif.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

test.x = data.frame(select(test,-y))
#removing colinear
test.x = test.x[,-c(highCorrelated)]
#ANOVA +RFE
test.x = test.x[rfe.predictors]
test.y = select(test, y)
test.data = data.frame(test.x, y = (test.y))
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")

#predictions
predictions = predict(mod_knn, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'KNN Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()


######################## Logistic Regression ############################
print()
'Logistic Regression'
library(mlbench)
library(ISLR)
library(caret)
library(mlr) #machine learning
library(glmnet) #cv testing
library(mboost) #boosting for mlr
library(faux) #needed for RFE
library(DataExplorer)
library(randomForest)
library(GGally)
library(dplyr)

load(file ="class_data(1).RData") #variables x, y, x_new

set.seed(2021)
test_x<-xnew
train_x<-x
train_y<-y

names(x)
summary(x)
summary(y)

#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

cut<-0.90 #or .75 could be used
corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut)
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

sig<-0.01 #or .01 could be used
i = 0
vec <- vector() 
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]
project.data.train = data.frame(anova.train.x, y=(train.y))
vec

#Correlation + ANOVA
#Or stepwise regression could have been used
rfe.x <- anova.train.x

control <- rfeControl(functions = rfFuncs, method = "repeatedcv", 
                      repeats = 5, number =10)
result_rfe1 <- rfe(x= rfe.x, y= as.factor(as.matrix(train.y)), 
                   sizes = c(1:4), rfeControl= control)

result_rfe1
rfe.predictors <- predictors(result_rfe1)

ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

rfe.train.x <- rfe.x[rfe.predictors]
rfe.train.x

project.data.train = data.frame(rfe.train.x, y=(train.y))
#Consider the possible interactions of variables

cor(project.data.train)
interaction.x <- as.data.frame(model.matrix(~ (. + .)^2 - 1, project.data.train[, 1:4]))
project.data.train = data.frame(interaction.x, y=(train.y))
summary(project.data.train)
#Create the learner models

# Create a Learning Task
classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

# Construct a Learner
lrn1 = makeLearner("classif.cvglmnet", link= "logit", predict.type = "prob", fix.factors.prediction=TRUE)
mod_bin = train(lrn1, classif.task)
mod_bin

#Predictions on training data

predictions = predict(mod_bin, task = classif.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

#Applying the model on testing data

test.y = select(test, y)
test.x = data.frame(select(test,-y))

#removing co-linear variables
test.x = test.x[,-c(highCorrelated)]

#ANOVA+RFE
test.x = test.x[rfe.predictors]
#Interactions
test.x <- data.frame(model.matrix(~ (. + .)^2 - 1, test.x[, 1:4]))
test.data = data.frame(test.x, y = (test.y))
#Create learner
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")

#Predict the results
predictions = predict(mod_bin, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'Logit Model Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()

######################## LDA ############################
print()
'LDA'
library(mlbench)
library(ISLR)
library(caret)
library(mlr) #machine learning
library(glmnet) #cv testing
library(mboost) #boosting for mlr
library(faux) #needed for RFE
library(DataExplorer)
library(randomForest)
library(GGally)
library(dplyr)
library(plyr)
library(sparseLDA)

load(file ="class_data(1).RData") #variables x, y, x_new

set.seed(2021)
test_x<-xnew
train_x<-x
train_y<-y

#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

#Correlation Coefficient Threshold

cut<-0.75
sig<-0.01

corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut) # setting cutoff
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

#ANOVA Selection
i = 0
vec <- vector() # vector for storing indices of statistically significant variables
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]
project.data.train = data.frame(anova.train.x, y=(train.y))

#StepWise Regression


base.mod <- glm(y ~ 1, data= project.data.train, family= binomial)
all.mod <- glm(y ~ .,  data = project.data.train, family=binomial)
for.sel=step(object=base.mod,scope=list(upper=all.mod),direction="forward",k=2,trace=TRUE)
shortlistedVars <- names(unlist(for.sel[[1]]))
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"]
print(shortlistedVars)
step.train.x <- project.data.train[shortlistedVars]
step.train.x

project.data.train = data.frame(step.train.x, y=(train.y))

# Create a Learning Task
classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

#hypertunning
rdesc = makeResampleDesc("CV", iters = 10)#, reps = 5)
num_ps = makeParamSet(
  makeNumericParam("tol", lower=0, upper=0.1)
)
lrn = makeLearner("classif.lda")
ctrl = makeTuneControlGrid()
tunedLDA = tuneParams(lrn, task = classif.task, resampling = rdesc,
                      par.set = num_ps, control = ctrl, show.info = FALSE, 
                      measures = list(acc, setAggregation(acc, test.sd)))
tunedLDA
tunedLDA$x
tunedLDA$y

ldaTuningData <- generateHyperParsEffectData(tunedLDA)

#training with hyperpars
tuned_LDA = setHyperPars(makeLearner("classif.lda"), tol = tunedLDA$x$tol)
mod_lda = train(tuned_LDA, classif.task)
mod_lda

# train_accuracy
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 100,
                          stratify = TRUE)
kFoldCV <- resample(learner = tuned_LDA, task = classif.task,
                    resampling = kFold, measures = list(mmce, acc))

predictions = predict(mod_lda, task = classif.task)
kFoldCV$aggr

predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

# test_accuracy
test.x = data.frame(select(test,-y))
test.x = test.x[,-c(highCorrelated)] # removing co-linear variables
test.x = test.x[shortlistedVars] #if stepwise regression
test.y = select(test, y)
test.data = data.frame(test.x, y = (test.y))
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")

predictions = predict(mod_lda, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
mean(predicted.truth!=predicted.response)
table(predicted.truth==predicted.response)
predictions = predict(mod_lda, task = classif.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'LDA Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()

######################## Naive Bayes ############################
print()
'Naive Bayes'
library(mlbench)
library(ISLR)
library(caret)
library(mlr) #machine learning
library(glmnet) #cv testing
library(mboost) #boosting for mlr
library(faux) #needed for RFE
library(DataExplorer)
library(randomForest)
library(GGally)
library(dplyr)
library(plyr)
library(sparseLDA)

load(file ="class_data(1).RData") #variables x, y, x_new

test_x<-xnew
train_x<-x
train_y<-y
#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

#Correlation Coefficient Threshold

cut<-0.75
sig<-0.01

corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut) # setting cutoff
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

#ANOVA Selection
i = 0
vec <- vector() # vector for storing indices of statistically significant variables
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]
project.data.train = data.frame(anova.train.x, y=(train.y))

#StepWise Regression

base.mod <- glm(y ~ 1, data= project.data.train, family= binomial)
all.mod <- glm(y ~ .,  data = project.data.train, family=binomial)
for.sel=step(object=base.mod,scope=list(upper=all.mod),direction="forward",k=2,trace=TRUE)
shortlistedVars <- names(unlist(for.sel[[1]]))
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"]
print(shortlistedVars)
step.train.x <- project.data.train[shortlistedVars]
step.train.x

project.data.train = data.frame(step.train.x, y=(train.y))

# Create a Learning Task
classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

#hypertunning
rdesc = makeResampleDesc("CV", iters = 10)#, reps = 5)
num_ps = makeParamSet(
  makeNumericParam("laplace", lower=0, upper=10)
)
lrn = makeLearner("classif.naiveBayes")
ctrl = makeTuneControlGrid()
tunedNB = tuneParams(lrn, task = classif.task, resampling = rdesc,
                     par.set = num_ps, control = ctrl, show.info = FALSE, 
                     measures = list(acc, setAggregation(acc, test.sd)))
tunedNB
tunedNB$x
tunedNB$y

nbTuningData <- generateHyperParsEffectData(tunedNB)
#training with hyperpars
tuned_NB = setHyperPars(makeLearner("classif.naiveBayes"), laplace = tunedNB$x$laplace)
mod_nb = train(tuned_NB, classif.task)
mod_nb

# train_accuracy
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 100,
                          stratify = TRUE)
kFoldCV <- resample(learner = tuned_NB, task = classif.task,
                    resampling = kFold, measures = list(mmce, acc))

predictions = predict(mod_nb, task = classif.task)
kFoldCV$aggr

predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

# test_accuracy
test.x = data.frame(select(test,-y))
test.x = test.x[,-c(highCorrelated)] # removing co-linear variables
test.x = test.x[shortlistedVars] #if stepwise regression
test.y = select(test, y)
test.data = data.frame(test.x, y = (test.y))
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")

predictions = predict(mod_nb, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'Naive Bayes Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()

######################## QDA ############################
print()
'QDA'
library(mlbench)
library(ISLR)
library(caret)
library(mlr) #machine learning
library(glmnet) #cv testing
library(mboost) #boosting for mlr
library(faux) #needed for RFE
library(DataExplorer)
library(randomForest)
library(GGally)
library(dplyr)
library(plyr)
library(sparseLDA)

load(file ="class_data(1).RData") #variables x, y, x_new

set.seed(2021)
test_x<-xnew
train_x<-x
train_y<-y

#Scaling the data
x.scaled = scale(x, center = TRUE, scale = TRUE)
project.data = data.frame(x=x.scaled, y=(y))

#splitting training and testing data
dt = sort(sample(nrow(project.data), nrow(project.data)*.80))
train<-project.data[dt,]
test<-project.data[-dt,]

train.x = select(train, -y) # only the co-variates
train.y = select(train, y) # only the y

#Correlation Coefficient Threshold
cut<-0.9
sig<-0.01

corr.train.x = cor(train.x)
highCorrelated = findCorrelation(corr.train.x, cutoff = cut) # setting cutoff
train.x.non.cor = train.x[,-c(highCorrelated)]

train.x.non.cor = data.frame(train.x.non.cor)
non.corr.data = data.frame(x=train.x.non.cor, y=(train.y))

#ANOVA Selection
i = 0
vec <- vector() # vector for storing indices of statistically significant variables
for (col in train.x.non.cor){
  i = i + 1
  res.aov <- aov(y ~ col, data = non.corr.data)
  value = summary(res.aov)[[1]][["Pr(>F)"]]
  if(value[1] < sig){
    vec <- c(vec, i)
  }
}

anova.train.x = train.x.non.cor[vec]
project.data.train = data.frame(anova.train.x, y=(train.y))

#RFE Method
rfe.x <- anova.train.x

control <- rfeControl(functions = rfFuncs, method = "repeatedcv", 
                      repeats = 5, number =10)
result_rfe1 <- rfe(x= rfe.x, y= as.factor(as.matrix(train.y)), 
                   sizes = c(1:35), rfeControl= control)

result_rfe1
rfe.predictors <- predictors(result_rfe1)

ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

rfe.train.x <- rfe.x[rfe.predictors]
rfe.train.x

project.data.train = data.frame(rfe.train.x, y=(train.y))

# Create a Learning Task
classif.task = makeClassifTask(id = "639_Project", data = project.data.train, target = "y", positive = "1")
classif.task

#hypertunning
rdesc = makeResampleDesc("CV", iters = 10)#, reps = 5)
num_ps = makeParamSet(
  makeDiscreteParam("method", c('moment','mle','mve','t'))
)
lrn = makeLearner("classif.qda")
ctrl = makeTuneControlGrid()
tunedQDA = tuneParams(lrn, task = classif.task, resampling = rdesc,
                      par.set = num_ps, control = ctrl, show.info = FALSE, 
                      measures = list(acc, setAggregation(acc, test.sd)))
tunedQDA
tunedQDA$x
tunedQDA$y

qdaTuningData <- generateHyperParsEffectData(tunedQDA)
#training with hyperpars
tuned_QDA = setHyperPars(makeLearner("classif.qda"), method = tunedQDA$x$method)
mod_qda = train(tuned_QDA, classif.task)
mod_qda

# train_accuracy
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 100,
                          stratify = TRUE)
kFoldCV <- resample(learner = tuned_QDA, task = classif.task,
                    resampling = kFold, measures = list(mmce, acc))

predictions = predict(mod_qda, task = classif.task)
kFoldCV$aggr

predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
mean(predicted.truth==predicted.response)
table(predicted.truth==predicted.response)

# test_accuracy
test.x = data.frame(select(test,-y))
test.x = test.x[,-c(highCorrelated)] # removing co-linear variables
test.x = test.x[rfe.predictors] #if ANOVA +RFE
test.y = select(test, y)
test.data = data.frame(test.x, y = (test.y))
test.task = makeClassifTask(id = "639_Project", data = test.data, target = "y", positive = "1")

predictions = predict(mod_qda, task = test.task)
predicted.response = predictions$data$response
predicted.truth = predictions$data$truth
'QDA Test Results'
'Accuracy'
mean(predicted.truth==predicted.response)
'Misclassification error'
mean(predicted.truth!=predicted.response)
'Table'
table(predicted.truth==predicted.response)
print()

######################## Clustering ############################
'Clustering Problem'

######################## K-Means ############################
print()
'K-Means'
library(ISLR)
library(caret)
library(tidyverse)
library(mlr)
library(fda)
library(FNN)
library(kknn)
library(GGally)
library(factoextra)
library(NbClust)
library(dplyr)
library(cluster)

load(file ="cluster_data(1).RData") # Loads y

y.scaled = scale(y, center = TRUE, scale = TRUE)

# Elbow method
#total within cluster sum of squares (WSS) as a function of the number of clusters
fviz_nbclust(y.scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2) +
  labs(subtitle = "Elbow method") # add subtitle

# Silhouette method
#quality of a clustering and determines how well each point lies within its cluster
fviz_nbclust(y.scaled, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")


km_res <- kmeans(y.scaled, centers = 2, nstart = 20)

sil <- silhouette(km_res$cluster, dist(y.scaled))
fviz_silhouette(sil)

nbclust_out <- NbClust(
  data = y.scaled,
  distance = "euclidean",
  min.nc = 2, # minimum number of clusters
  max.nc = 10, # maximum number of clusters
  method = "ward.D2" 
)
# create a dataframe of the optimal number of clusters
nbclust_plot <- data.frame(clusters = nbclust_out$Best.nc[1, ])
# select only indices which select between 2 and 5 clusters
nbclust_plot <- subset(nbclust_plot, clusters >= 2 & clusters <= 20)

# create plot
ggplot(nbclust_plot) +
  aes(x = clusters) +
  geom_histogram(bins = 30L, fill = "#0c4c8a") +
  labs(x = "Number of clusters", y = "Frequency among all indices", title = "Optimal number of clusters") +  theme_minimal()

gradient.color <- list(low = "steelblue",  high = "white")

y.scaled %>%    # Remove column 5 (Species)
  get_clust_tendency(n = 784, gradient = gradient.color)

k2 <- kmeans(y.scaled, centers=2, nstart=25)
fviz_cluster(k2, data=y.scaled)

k2 <- kmeans(y.scaled, centers=3, nstart=25)
fviz_cluster(k2, data=y.scaled)

k2 <- kmeans(y.scaled, centers=4, nstart=25)
fviz_cluster(k2, data=y.scaled)

k2 <- kmeans(y.scaled, centers=10, nstart=25)
fviz_cluster(k2, data=y.scaled)

k2 <- kmeans(y.scaled, centers=19, nstart=25)
fviz_cluster(k2, data=y.scaled)
print()

######################## KMedoids ############################

print()
'K Medoids'
library(factoextra)
library(cluster)

load("cluster_data.RData") # loads y
df <- y

#scale each variable to have a mean of 0 and sd of 1
df <- scale(df)

fviz_nbclust(df, pam, method = "silhouette")+  theme_classic()

fviz_nbclust(df, pam, method = "wss",  k.max = 20)

#calculate gap statistic based on number of clusters
gap_stat <- clusGap(df,
                    FUN = pam,
                    K.max = 20, #max clusters to consider
                    B = 10) #total bootstrapped iterations

#plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

set.seed(2021)
kmed <- pam(df, k = 2)
fviz_cluster(kmed, data = df)

set.seed(2021)
kmed <- pam(df, k = 4)
fviz_cluster(kmed, data = df)

set.seed(2021)
kmed <- pam(df, k = 19)
fviz_cluster(kmed, data = df)
print()

######################## DBSCAN ############################

print()
'DBSCAN'

library(factoextra)
library(dbscan)
library("fpc")
library(cluster)

load(file ="cluster_data(1).RData") # Loads y

df = as.data.frame(y)
df <- scale(df)

dbscan::kNNdistplot(df, k =  1)
abline()
db <- fpc::dbscan(df, eps = 40, MinPts = 5)
plot(df, col=db$cluster)
print(db$cluster)

dbscan::kNNdistplot(df, k =  3)
abline()
db <- fpc::dbscan(df, eps = 40, MinPts = 5)
plot(df, col=db$cluster)
print(db$cluster)

dbscan::kNNdistplot(df, k =  5)
abline()
db <- fpc::dbscan(df, eps = 40, MinPts = 5)
plot(df, col=db$cluster)
print(db$cluster)

dbscan::kNNdistplot(df, k =  7)
abline()
db <- fpc::dbscan(df, eps = 40, MinPts = 5)
plot(df, col=db$cluster)
print(db$cluster)

print()

######################## Hierarchical Clustering ############################

print()
'Hierarchical Clustering'

load(file ="cluster_data(1).RData") # Loads y
df = as.data.frame(y)
df <- na.omit(df)
df <- scale(df)
df
d <- dist(df, method = "euclidean")


hc.complete = hclust(dist(df), method = 'complete')
plot(hc.complete)


hc1 <- hclust(d, method = "complete" )
plot(hc1, cex = 0.2, hang = 0)

# Compute with agnes
hc2 <- agnes(df, method = "complete")
plot(hc2)
# Agglomerative coefficient
hc2$ac
print()