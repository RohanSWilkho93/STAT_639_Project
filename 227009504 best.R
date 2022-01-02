## Best Classification

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

estimater.error = mean(predicted.truth!=predicted.response)
x.new.scaled = scale(xnew, center = TRUE, scale = TRUE)
x.new.scaled = data.frame(x = x.new.scaled)
x.new.scaled = x.new.scaled[rfe.predictors]
y = rep(1, times = 1000)
new.data = data.frame(x.new.scaled, y = as.factor(y)) # this y is dummy; just used just to create a dataframe and task. Only the predicted repsonses are finally selected
new.test.task = makeClassifTask(id = "639_Project", data = new.data, target = "y", positive = "1")
predictions = predict(mod_rf, task = new.test.task) 
ynew = predictions$data$response
save(ynew,estimater.error,file="227009504.RData")
print()