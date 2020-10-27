# Name: Mick Shaw
# Last updated: 7-3-2020


###############
# Project notes
###############

#Analysis of iPhone and Samsung Galaxy Overall Sentiment


###############
# Housekeeping
###############

# Clear all variables from R
rm(list = ls())

# Set working directory
getwd()
setwd('F:/UT Data Analytics/Course 4 - Data Science and Big Data/Task 3')
dir()


################################
## Install and load packages
################################

install.packages("dplyr")
install.packages("tidyr")
install.packages("lubridate")
install.packages("forecast")
install.packages("TTR")
install.packages("RMySQL")
install.packages("plotly")
install.packages("ggfortify")
install.packages("ggplot2")
install.packages("reshape")
install.packages("naniar")
install.packages("plot3D")
install.packages("shiny")
install.packages("shinydashboard")
install.packages("devtools")
install.packages("GGally")
install.packages("pacman")
install.packages("kknn")

library(pacman)
library(kknn)

p_load(shiny, shinydashboard, dplyr, ggplot2, plotly, lubridate, naniar, devtools,
       corrplot, GGally, caret, reshape, doParallel, readr, mlbench, tidyverse, e1071, 
       kernlab, randomForest, gridExtra, caTools)

options(max.print = 1000000)

###############################
# --- Parallel processing --- #
###############################

detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores; 2 in this example
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

######################## 
# --- Load dataset --- #  
########################

# --- Load Train/Existing data (Dataset 1) --- #
iPhoneData <- read.csv("iphone_smallmatrix_labeled_8d.csv", stringsAsFactors = FALSE)
class(iPhoneData)  # "data.frame"
str(iPhoneData)


# --- Load Train/Existing data (Dataset 2) --- #
galaxyData <- read.csv("galaxy_smallmatrix_labeled_9d.csv", stringsAsFactors = FALSE)
class(galaxyData)  # "data.frame"
str(galaxyData)
galaxyData 


# ---Data Exploration--- #

## Inspect the Data Types
y <- list(
  title = "iPhone Sentiment",
  size = 14,
  color = 'black'
)
x <- list(
  title = "Sentiment Count",
  size = 14,
  color = 'black'
)

plot_ly(iPhoneData, y= ~iPhoneData$iphonesentiment, type='histogram', orientation = 'h')%>%
 layout(xaxis = x, yaxis = y)

yy <- list(
  title = "Galaxy Sentiment",
  size = 14,
  color = 'black'
)
plot_ly(galaxyData, y= ~galaxyData$galaxysentiment, type='histogram', orientation = 'h')%>%
  layout(xaxis = x, yaxis = yy)



# --- Preprocessing Data --- #

### --- Correlation --- ###

## -- iPhone -- ##

iphoneCOR <- cor(iPhoneData)
iphoneCOR
class(iphoneCOR)


iphoneCORdf <- as.data.frame(iphoneCOR)
write.csv(iphoneCORdf, file = "iPhoneCORdf.csv")


iphoneCORplot <- corrplot(iphoneCOR)
iphoneCORplot

iphoneHIcor <- findCorrelation(iphoneCOR, cutoff = 0.9)
iphoneHIcor = sort(iphoneHIcor)
iphoneHIcor
red_iphoneData <- iPhoneData[,-c(iphoneHIcor)]
red_iphoneData
red_iphoneDataCor <- cor(red_iphoneData)
red_iphoneDataCor
red_iphoneDataCorPlot <- corrplot(red_iphoneDataCor)
red_iphoneDataCorPlot

red_iphoneCORdf <- as.data.frame(red_iphoneDataCor)
write.csv(red_iphoneCORdf, "ReducediPhoneCORdf.csv")

#iPhone Correlation Data Set#
iPhoneCORR <- subset(iPhoneData, select = -c(5,6,16,21,24,29,31,34,46,51,55,56,57))
head(iPhoneCORR)
class(iPhoneCORR)

## -- Galaxy -- ##

galaxyCOR <- cor(galaxyData)
galaxyCOR
class(galaxyCOR)

galaxyCORdf <- as.data.frame(galaxyCOR)
write.csv(galaxyCORdf, file = "galaxyCORdf.csv")


galaxyCORplot <- corrplot(galaxyCOR)
galaxyCORplot

galaxyHIcor <- findCorrelation(galaxyCOR, cutoff = 0.9)
galaxyHIcor = sort(galaxyHIcor)
galaxyHIcor
red_galaxyData <- galaxyData[,-c(galaxyHIcor)]
red_galaxyData
red_galaxyDataCor <- cor(red_galaxyData)
red_galaxyDataCor
red_galaxyDataCorPlot <- corrplot(red_iphoneDataCor)
red_galaxyDataCorPlot

red_galaxyCORdf <- as.data.frame(red_galaxyDataCor)
write.csv(red_galaxyCORdf, "ReducedgalaxyCORdf.csv")


#Galaxy Correlation Data Set#

galaxyCORR <- subset(galaxyData, select = -c(5,6,16,21,24,29,31,34,46,51,55,56,57))
head(galaxyCORR)



### --- Near Zero Variance --- ###

#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 

#iPhone#
nzvMetricsiPhone <- nearZeroVar(iPhoneData, saveMetrics = TRUE)
nzvMetricsiPhone

#Galaxy#
nzvMetricsGalaxy <- nearZeroVar(galaxyData, saveMetrics = TRUE)
nzvMetricsGalaxy




# nearZeroVar() with saveMetrics = FALSE returns an vector 
#iPhone#
nzviPhone <- nearZeroVar(iPhoneData, saveMetrics = FALSE) 
nzviPhone

#Galaxy#
nzvGalaxy <- nearZeroVar(galaxyData, saveMetrics = FALSE)
nzvGalaxy


### --- Create new NZV data set --- ###

#iPhone#
iphoneNZV <- iPhoneData[,-nzviPhone]
iphoneNZV
str(iphoneNZV)
iPhoneNZV <- iphoneNZV

#Galaxy#
galaxyNZV <- galaxyData[,-nzvGalaxy]
galaxyNZV
str(galaxyNZV)




### --- Recursive Feature Elimination --- ###

## -- iPhone -- ##

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- iPhoneData[sample(1:nrow(iPhoneData), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeiPhone <- rfeResults

# Plot results
plot(rfeiPhone, type=c("g", "o"))



# The top 5 variables (out of 18):
#   iphone, googleandroid, iphonedispos, iphonedisneg, samsunggalaxy


# create new data set with rfe recommended features
iPhoneRFE <- iPhoneData[,predictors(rfeiPhone)]

# add the dependent variable to iphoneRFE
iPhoneRFE$iphonesentiment <- iPhoneData$iphonesentiment

# review outcome
str(iPhoneRFE)
is.na(iPhoneRFE)


## -- Galaxy -- ##

# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxyData[sample(1:nrow(galaxyData), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfegalaxy <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfegalaxy

# The top 5 variables (out of 46):
#   iphone, googleandroid, samsunggalaxy, iphoneperunc, iphoneperpos

# Plot results
plot(rfegalaxy, type=c("g", "o"))


# create new data set with rfe recommended features
galaxyRFE <- galaxyData[,predictors(rfegalaxy)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxyData$galaxysentiment

# review outcome
str(galaxyRFE)
is.na(galaxyRFE)


### --- Current Data Sets --- ###
# iPhoneData - Out of the Box
# iPhoneCORR - Removed high correllated variables
# iPhoneNZV - Near Zero Variance
# iPhoneRFE - Recursive Feature Elimination | Returns list of recommended features.

# galaxyData - Out of the Box
# galaxyCORR - Removed high correllated variables
# galaxyNZV - Near Zero Variance
# galaxyRFE - Recursive Feature Elimination | Returns list of recommended features.

write.csv(iPhoneCORR, "iPhoneCORR.csv")
write.csv(iPhoneNZV, "iPhoneNZV.csv")
write.csv(iPhoneRFE, "iPhoneRFE.csv")

write.csv(galaxyCORR, "galaxyCORR.csv")
write.csv(galaxyNZV, "galaxyNZV.csv")
write.csv(galaxyRFE, "galaxyRFE.csv")

## -- Change sentiment variables to factors -- ##
iPhoneData$iphonesentiment <- as.factor(iPhoneData$iphonesentiment)
iPhoneCORR$iphonesentiment <- as.factor(iPhoneCORR$iphonesentiment)
iPhoneNZV$iphonesentiment <- as.factor(iPhoneNZV$iphonesentiment)
iPhoneRFE$iphonesentiment <- as.factor(iPhoneRFE$iphonesentiment)

galaxyData$galaxysentiment <- as.factor(galaxyData$galaxysentiment)
galaxyCORR$galaxysentiment <- as.factor(galaxyCORR$galaxysentiment)
galaxyNZV$galaxysentiment <- as.factor(galaxyNZV$galaxysentiment)
galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)


### --- Train/Test Sets --- ###


seed <- 123



## iPhone Out of the Box ##
set.seed(seed)
inTrainingOOBiPhone <- createDataPartition(iPhoneData$iphonesentiment, p=0.70, list=FALSE)
trainSetOOBiPhone <- iPhoneData[inTrainingOOBiPhone,]   
testSetOOBiPhone <- iPhoneData[-inTrainingOOBiPhone,]  
trainSetOOBiPhone



## Galaxy Out of the Box ##
set.seed(seed)
inTrainingOOBgalaxy <- createDataPartition(galaxyData$galaxysentiment, p=0.70, list=FALSE)
trainSetOOBgalaxy <- galaxyData[inTrainingOOBgalaxy,]   
testSetOOBgalaxy <- galaxyData[-inTrainingOOBgalaxy,]  
trainSetOOBgalaxy



## iPhone - iPhoneCorr ##
set.seed(seed)
inTrainingCORiPhone <- createDataPartition(iPhoneCORR$iphonesentiment, p=0.70, list=FALSE)
trainSetCORiPhone <- iPhoneCORR[inTrainingCORiPhone,]   
testSetCORiPhone <- iPhoneCORR[-inTrainingCORiPhone,]  
trainSetCORiPhone


## Galaxy - galaxyCorr ##
set.seed(seed)
inTrainingCORgalaxy <- createDataPartition(galaxyCORR$galaxysentiment, p=0.70, list=FALSE)
trainSetCORgalaxy <- galaxyCORR[inTrainingOOBgalaxy,]   
testSetCORgalaxy <- galaxyCORR[-inTrainingOOBgalaxy,]  
trainSetCORgalaxy

## iPhone - iPhoneNZV ##
set.seed(seed)
inTrainingNZViPhone <- createDataPartition(iPhoneNZV$iphonesentiment, p=0.70, list=FALSE)
trainSetNZViPhone <- iPhoneNZV[inTrainingNZViPhone,]   
testSetNZViPhone <- iPhoneNZV[-inTrainingNZViPhone,]  
trainSetNZViPhone


## Galaxy - galaxyNZV ##
set.seed(seed)
inTrainingNZVgalaxy <- createDataPartition(galaxyNZV$galaxysentiment, p=0.70, list=FALSE)
trainSetNZVgalaxy <- galaxyNZV[inTrainingNZVgalaxy,]   
testSetNZVgalaxy <- galaxyNZV[-inTrainingNZVgalaxy,]  
trainSetNZVgalaxy


## iPhone - iPhoneRFE ##
set.seed(seed)
inTrainingRFEiPhone <- createDataPartition(iPhoneRFE$iphonesentiment, p=0.70, list=FALSE)
trainSetRFEiPhone <- iPhoneRFE[inTrainingRFEiPhone,]   
testSetRFEiPhone <- iPhoneRFE[-inTrainingRFEiPhone,]  
trainSetRFEiPhone


## Galaxy - galaxyRFE ##
set.seed(seed)
inTrainingRFEgalaxy <- createDataPartition(galaxyRFE$galaxysentiment, p=0.70, list=FALSE)
trainSetRFEgalaxy <- galaxyRFE[inTrainingRFEgalaxy,]   
testSetRFEgalaxy <- galaxyRFE[-inTrainingRFEgalaxy,]  
trainSetRFEgalaxy



### --- KKNN Models --- ###

## Train control

# set 10 fold cross validation
kknnControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)



## iPhone Out of the Box Model
set.seed(seed)
KKNNfit_iPhoneOOB <- train(iphonesentiment~., data=trainSetOOBiPhone, method="kknn", trControl=kknnControl)
KKNNfit_iPhoneOOB

# kmax  Accuracy   Kappa    
# 5     0.3104693  0.1560163
# 7     0.3230223  0.1594246
# 9     0.3283062  0.1609948
# 
# Tuning parameter 'distance' was held constant at a value of 2
# Tuning
# parameter 'kernel' was held constant at a value of optimal
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.


saveRDS(KKNNfit_iPhoneOOB, file = "KKNNfit_iPhoneOOB.rds")
KKNNfit_iPhoneOOB <- readRDS("KKNNfit_iPhoneOOB.rds")

kknnPred_iPhoneOOB <- predict(KKNNfit_iPhoneOOB, testSetOOBiPhone)
kknnPred_iPhoneOOB

postResample(kknnPred_iPhoneOOB, testSetOOBiPhone$iphonesentiment)
# Accuracy     Kappa 
# 0.3383033 0.1713617 



## Galaxy Out of the Box Model
set.seed(seed)
KKNNfit_galaxyOOB <- train(galaxysentiment~., data=trainSetOOBgalaxy, method="kknn", trControl=kknnControl)
KKNNfit_galaxyOOB

# kmax  Accuracy   Kappa    
# 5     0.6591826  0.4097381
# 7     0.7336326  0.4887705
# 9     0.7209199  0.4785159
# 
# Tuning parameter 'distance' was held constant at a value of 2
# Tuning
# parameter 'kernel' was held constant at a value of optimal
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were kmax = 7, distance = 2 and kernel = optimal.

saveRDS(KKNNfit_galaxyOOB, file = "KKNNfit_galaxyOOB.rds")
KKNNfit_galaxyOOB <- readRDS("KKNNfit_galaxyOOB.rds")

kknnPred_galaxyOOB <- predict(KKNNfit_galaxyOOB, testSetOOBgalaxy)
kknnPred_galaxyOOB

postResample(kknnPred_galaxyOOB, testSetOOBgalaxy$galaxysentiment)

# Accuracy     Kappa 
# 0.7318522 0.4877367



### --- RF Models --- ###

## Train control

# set 10 fold cross validation
rfControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

## iPhone Out of the Box Model
set.seed(seed)
RFfit_iPhoneOOB <- train(iphonesentiment~., data=trainSetOOBiPhone, method="rf", trControl=rfControl)
RFfit_iPhoneOOB

# mtry  Accuracy   Kappa    
# 2    0.6995490  0.3693246
# 30    0.7738634  0.5645740
# 58    0.7643964  0.5508612
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 30.

saveRDS(RFfit_iPhoneOOB, file = "RFfit_iPhoneOOB.rds")
RFfit_iPhoneOOB <- readRDS("RFfit_iPhoneOOB.rds")

rfPred_iPhoneOOB <- predict(RFfit_iPhoneOOB, testSetOOBiPhone)
rfPred_iPhoneOOB

postResample(rfPred_iPhoneOOB, testSetOOBiPhone$iphonesentiment)

# Accuracy     Kappa 
# 0.7737789 0.5611529 


# cmRF_iPhoneOOB <- confusionMatrix(data = rfPred_iPhoneOOB, reference = testSetOOBiPhone$iphonesentiment)
# cmRF_iPhoneOOB

rfVarfit_iPhoneOOB <- varImp(RFfit_iPhoneOOB)
rfVarfit_iPhoneOOB

# Overall
# iphone        100.000
# iphonedisunc   34.557
# samsunggalaxy  33.678
# htcphone       30.966
# iphonedisneg   29.474
# googleandroid  27.395
# iphoneperpos   23.535
# iphonedispos   17.977
# iphonecamunc   14.765
# iphonecampos   13.098
# iphoneperneg   12.859
# iphoneperunc   11.389
# iphonecamneg   10.071
# htccampos       4.412
# sonyxperia      3.953
# ios             2.895
# htcdispos       2.569
# samsungperpos   2.073
# htccamneg       1.657
# htcperpos       1.434


## Galaxy Out of the Box Model
set.seed(seed)
RFfit_galaxyOOB <- train(galaxysentiment~., data=trainSetOOBgalaxy, method="rf", trControl=rfControl)
RFfit_galaxyOOB

# mtry  Accuracy   Kappa    
# 2    0.7057536  0.3588622
# 30    0.7632767  0.5295357
# 58    0.7556447  0.5186300
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 30.

saveRDS(RFfit_galaxyOOB, file = "RFfit_galaxyOOB.rds")
RFfit_galaxyOOB <- readRDS("RFfit_galaxyOOB.rds")

rfPred_galaxyOOB <- predict(RFfit_galaxyOOB, testSetOOBgalaxy)
rfPred_galaxyOOB

postResample(rfPred_galaxyOOB, testSetOOBgalaxy$galaxysentiment)

# Accuracy     Kappa 
# 0.7680186 0.5367246

rfVarfit_galaxyOOB <- varImp(RFfit_galaxyOOB)
rfVarfit_galaxyOOB

# Overall
# iphone        100.000
# iphonedisunc   34.752
# samsunggalaxy  34.379
# htcphone       31.968
# iphonedisneg   28.326
# iphoneperpos   25.705
# googleandroid  23.842
# iphonedispos   18.735
# iphonecamunc   16.201
# iphonecampos   14.125
# iphoneperneg   13.976
# iphoneperunc   11.689
# iphonecamneg   10.587
# htccampos       6.986
# ios             5.196
# sonyxperia      4.448
# htcdispos       3.684
# htcperpos       2.403
# samsungperpos   1.987
# samsungdispos   1.896



### --- C5.0 Models --- ###

## Train control        

c5.0Control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)


## iPhone Out of the Box Model ##
set.seed(seed)
c5.0fit_iPhoneOOB <- train(iphonesentiment~., data=trainSetOOBiPhone, method="C5.0", trControl=c5.0Control,importancetrControl=c5.0Control)
c5.0fit_iPhoneOOB 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8174, 8173, 8175, 8175, 8175, 8174, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7721012  0.5572920
# rules  FALSE   10      0.7603238  0.5399142
# rules  FALSE   20      0.7603238  0.5399142
# rules   TRUE    1      0.7734225  0.5601200
# rules   TRUE   10      0.7604328  0.5393405
# rules   TRUE   20      0.7604328  0.5393405
# tree   FALSE    1      0.7729815  0.5592708
# tree   FALSE   10      0.7630753  0.5460562
# tree   FALSE   20      0.7630753  0.5460562
# tree    TRUE    1      0.7728707  0.5593231
# tree    TRUE   10      0.7603195  0.5406196
# tree    TRUE   20      0.7603195  0.5406196
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

saveRDS(c5.0fit_iPhoneOOB , file = "c5.0fit_iPhoneOOB .rds")
c5.0fit_iPhoneOOB  <- readRDS("c5.0fit_iPhoneOOB .rds")

c5.0Pred_iPhoneOOB <- predict(c5.0fit_iPhoneOOB , testSetOOBiPhone)
c5.0Pred_iPhoneOOB



postResample(c5.0Pred_iPhoneOOB, testSetOOBiPhone$iphonesentiment)

# Accuracy     Kappa 
# 0.7724936 0.5558736

#confusionMatrix(data = c5.0PredB0LOCr3, reference =  testSetB0LOC$LOCATION)


## Galaxy Out of the Box Model ##

set.seed(seed)
c5.0fit_galaxyOOB <- train(galaxysentiment~., data=trainSetOOBgalaxy, method="C5.0", trControl=c5.0Control,importancetrControl=c5.0Control)
c5.0fit_galaxyOOB 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8136, 8136, 8136, 8135, 8137, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7653784  0.5294103
# rules  FALSE   10      0.7537646  0.5086583
# rules  FALSE   20      0.7537646  0.5086583
# rules   TRUE    1      0.7658209  0.5298597
# rules   TRUE   10      0.7508891  0.4990666
# rules   TRUE   20      0.7508891  0.4990666
# tree   FALSE    1      0.7628352  0.5249169
# tree   FALSE   10      0.7567509  0.5144032
# tree   FALSE   20      0.7567509  0.5144032
# tree    TRUE    1      0.7636093  0.5258885
# tree    TRUE   10      0.7574152  0.5167121
# tree    TRUE   20      0.7574152  0.5167121
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.


saveRDS(c5.0fit_galaxyOOB , file = "c5.0fit_galaxyOOB .rds")
c5.0fit_galaxyOOB  <- readRDS("c5.0fit_galaxyOOB .rds")

c5.0Pred_galaxyOOB <- predict(c5.0fit_galaxyOOB , testSetOOBgalaxy)
c5.0Pred_galaxyOOB



postResample(c5.0Pred_galaxyOOB, testSetOOBgalaxy$galaxysentiment)

# Accuracy     Kappa 
# 0.7680186 0.5323955

### --- SVM Models --- ###

## Train control        

## Building 0 Location Model C5.0
svmControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
      
set.seed(seed)
svmfit_iPhoneOOB <- train(iphonesentiment~., data=trainSetOOBiPhone, method="svmLinear", trControl=c5.0Control,importancetrControl=svmControl)
svmfit_iPhoneOOB 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8174, 8173, 8175, 8175, 8175, 8174, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7083582  0.4116978
# 
# Tuning parameter 'C' was held constant at a value of 1


saveRDS(svmfit_iPhoneOOB , file = "svmfit_iPhoneOOB .rds")
svmfit_iPhoneOOB  <- readRDS("svmfit_iPhoneOOB .rds")

svmPred_iPhoneOOB <- predict(svmfit_iPhoneOOB , testSetOOBiPhone)
svmPred_iPhoneOOB

postResample(svmPred_iPhoneOOB, testSetOOBiPhone$iphonesentiment)

# Accuracy     Kappa 
# 0.7113111 0.4190589 

## Galaxy Out of the Box Model ##

set.seed(seed)
svmfit_galaxyOOB <- train(galaxysentiment~., data=trainSetOOBgalaxy, method="svmLinear", trControl=svmControl,importancetrControl=svmControl)
svmfit_galaxyOOB 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8136, 8136, 8136, 8135, 8137, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7035399  0.3798445
# 
# Tuning parameter 'C' was held constant at a value of 1

saveRDS(svmfit_galaxyOOB , file = "svmfit_galaxyOOB .rds")
svmfit_galaxyOOB  <- readRDS("svmfit_galaxyOOB .rds")

svmPred_galaxyOOB <- predict(svmfit_galaxyOOB , testSetOOBgalaxy)
svmPred_galaxyOOB



postResample(svmPred_galaxyOOB, testSetOOBgalaxy$galaxysentiment)

# Accuracy     Kappa 
# 0.6982692 0.3680156 

### --- iPhone Model Data - Out of the Box--- ###
ModelData_iPhoneOOB <- resamples(list(KKNNfit_iPhoneOOB=KKNNfit_iPhoneOOB, RFfit_iPhoneOOB=RFfit_iPhoneOOB, c5.0fit_iPhoneOOB=c5.0fit_iPhoneOOB, svmfit_iPhoneOOB=svmfit_iPhoneOOB))
ModelData_iPhoneOOB
class(ModelData_iPhoneOOB)
summary(ModelData_iPhoneOOB)

# Models: KKNNfit_iPhoneOOB, RFfit_iPhoneOOB, c5.0fit_iPhoneOOB, svmfit_iPhoneOOB 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# KKNNfit_iPhoneOOB 0.3083700 0.3196828 0.3276442 0.3283062 0.3342511 0.3513216    0
# RFfit_iPhoneOOB   0.7599119 0.7657571 0.7728300 0.7738634 0.7820352 0.7931793    0
# c5.0fit_iPhoneOOB 0.7533040 0.7634912 0.7724030 0.7734225 0.7828621 0.7920792    0
# svmfit_iPhoneOOB  0.6971366 0.7048458 0.7058820 0.7083582 0.7118090 0.7246696    0
# 
# Kappa 
#                        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# KKNNfit_iPhoneOOB 0.1417769 0.1501527 0.1562114 0.1609948 0.1712841 0.1871676    0
# RFfit_iPhoneOOB   0.5332005 0.5456836 0.5628832 0.5645740 0.5831302 0.6082148    0
# c5.0fit_iPhoneOOB 0.5168090 0.5382150 0.5592303 0.5601200 0.5804472 0.6025526    0
# svmfit_iPhoneOOB  0.3822059 0.4029439 0.4100419 0.4116978 0.4173607 0.4467894    0

### --- Galaxy Model Data - Out of the Box--- ###
ModelData_galaxyOOB <- resamples(list(KKNNfit_galaxyOOB=KKNNfit_galaxyOOB, RFfit_galaxyOOB=RFfit_galaxyOOB, c5.0fit_galaxyOOB=c5.0fit_galaxyOOB, svmfit_galaxyOOB=svmfit_galaxyOOB))
ModelData_galaxyOOB
class(ModelData_galaxyOOB)
summary(ModelData_galaxyOOB)
ModelData_galaxyOOBdf <- as.data.frame(ModelData_galaxyOOB)
write.csv(ModelData_galaxyOOBdf, "ModelData_galaxyOOBdf.csv")

# Models: KKNNfit_galaxyOOB, RFfit_galaxyOOB, c5.0fit_galaxyOOB, svmfit_galaxyOOB 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# KKNNfit_galaxyOOB 0.6983425 0.7269973 0.7362905 0.7336326 0.7419801 0.7511062    0
# RFfit_galaxyOOB   0.7535912 0.7578094 0.7664649 0.7632767 0.7676348 0.7688053    0
# c5.0fit_galaxyOOB 0.7580110 0.7602210 0.7652280 0.7658209 0.7706774 0.7765487    0
# svmfit_galaxyOOB  0.6950276 0.6980597 0.7033769 0.7035399 0.7071013 0.7134956    0
# 
# Kappa 
#                        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# KKNNfit_galaxyOOB 0.4369133 0.4766108 0.4900178 0.4887705 0.5094091 0.5172740    0
# RFfit_galaxyOOB   0.5057503 0.5183267 0.5293526 0.5295357 0.5409783 0.5494057    0
# c5.0fit_galaxyOOB 0.5118590 0.5161435 0.5262821 0.5298597 0.5443783 0.5514706    0
# svmfit_galaxyOOB  0.3617064 0.3650152 0.3823304 0.3798445 0.3917353 0.4024262    0


### --- Top Performing Model - iPhone --- ###

#Accuracy
#RFfit_iPhoneOOB   0.7599119 0.7657571 0.7728300 0.7738634 0.7820352 0.7931793    0

#Kappa
#RFfit_iPhoneOOB   0.5332005 0.5456836 0.5628832 0.5645740 0.5831302 0.6082148    0


### --- Top Performing Model - Galaxy --- ###

#Accuracy
# c5.0fit_galaxyOOB 0.7580110 0.7602210 0.7652280 0.7658209 0.7706774 0.7765487    0

#Kappa
# c5.0fit_galaxyOOB 0.5118590 0.5161435 0.5262821 0.5298597 0.5443783 0.5514706    0


### --- Current Data Sets --- ###
# iPhoneCORR - Removed high correllated variables
# iPhoneNZV - Near Zero Variance
# iPhoneRFE - Recursive Feature Elimination | Returns list of recommended features.

# galaxyCORR - Removed high correllated variables
# galaxyNZV - Near Zero Variance
# galaxyRFE - Recursive Feature Elimination | Returns list of recommended features.


############################################################
### Test best performing models with alternate data sets ###
############################################################


## -- iPhone| Correlation Data Set -- ##

set.seed(seed)
RFfit_iPhoneCOR <- train(iphonesentiment~., data=trainSetCORiPhone, method="rf", trControl=rfControl)
RFfit_iPhoneCOR

# mtry  Accuracy   Kappa    
# 2    0.6912903  0.3450048
# 23    0.7719925  0.5611163
# 45    0.7634046  0.5492467
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 23.


saveRDS(RFfit_iPhoneCOR, file = "RFfit_iPhoneCOR.rds")
RFfit_iPhoneCOR <- readRDS("RFfit_iPhoneCOR.rds")

rfPred_iPhoneCOR <- predict(RFfit_iPhoneCOR, testSetCORiPhone)
rfPred_iPhoneCOR

postResample(rfPred_iPhoneCOR, testSetCORiPhone$iphonesentiment)
# Accuracy     Kappa 
# 0.7724936 0.5586641

cmRF_iPhoneCOR <- confusionMatrix(data = rfPred_iPhoneCOR, reference = testSetCORiPhone$iphonesentiment)
cmRF_iPhoneCOR

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0  375    0    1    0    4    9
# 1    1    0    0    0    0    1
# 2    1    1   17    0    0    2
# 3    2    0    1  236    4    5
# 4    4    0    1    4  142   10
# 5  205  116  116  116  281 2235
# 
# Overall Statistics
# 
# Accuracy : 0.7725         
# 95% CI : (0.759, 0.7856)
# No Information Rate : 0.5815         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.5587         
# 
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: 0  Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity            0.6378 0.0000000 0.125000  0.66292  0.32947   0.9881
# Specificity            0.9958 0.9994699 0.998934  0.99660  0.99451   0.4877
# Pos Pred Value         0.9640 0.0000000 0.809524  0.95161  0.88199   0.7283
# Neg Pred Value         0.9392 0.9699074 0.969243  0.96705  0.92250   0.9671
# Prevalence             0.1512 0.0300771 0.034961  0.09152  0.11080   0.5815
# Detection Rate         0.0964 0.0000000 0.004370  0.06067  0.03650   0.5746
# Detection Prevalence   0.1000 0.0005141 0.005398  0.06375  0.04139   0.7889
# Balanced Accuracy      0.8168 0.4997350 0.561967  0.82976  0.66199   0.7379

rfVarfit_iPhoneCOR <- varImp(RFfit_iPhoneCOR)
rfVarfit_iPhoneCOR
# Overall
# iphone        100.000
# samsunggalaxy  34.998
# iphonedisunc   31.315
# iphonedisneg   28.691
# googleandroid  26.003
# iphoneperpos   23.169
# iphonedispos   17.374
# iphonecamunc   14.038
# iphonecampos   12.824
# sonyxperia     12.795
# iphoneperneg   11.988
# iphoneperunc   10.807
# iphonecamneg    9.224
# htccampos       7.299
# htcdispos       4.996
# htcperpos       2.553
# samsungperpos   2.179
# htccamneg       1.976
# samsungcamunc   1.531
# htcdisneg       1.465



## -- iPhone| Near Zero Variance Data Set -- ##

set.seed(seed)
RFfit_iPhoneNZV <- train(iphonesentiment~., data=trainSetNZViPhone, method="rf", trControl=rfControl)
RFfit_iPhoneNZV

# mtry  Accuracy   Kappa    
# 2    0.7452398  0.4968294
# 6    0.7379743  0.4895368
# 10    0.7310373  0.4805050

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 2.

saveRDS(RFfit_iPhoneNZV, file = "RFfit_iPhoneNZV.rds")
RFfit_iPhoneNZV <- readRDS("RFfit_iPhoneNZV.rds")

rfPred_iPhoneNZV <- predict(RFfit_iPhoneNZV, testSetNZViPhone)
rfPred_iPhoneNZV

postResample(rfPred_iPhoneNZV, testSetNZViPhone$iphonesentiment)
# Accuracy     Kappa 
# 0.7408740 0.4855716


cmRF_iPhoneNZV <- confusionMatrix(data = rfPred_iPhoneNZV, reference = testSetNZViPhone$iphonesentiment)
cmRF_iPhoneNZV

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0  363    0   17   56    9    6
# 1    0    0    0    0    0    0
# 2    0    0    0    0    0    0
# 3    0    0    0  139    2    2
# 4    3    0    0    4  131    5
# 5  222  117  119  157  289 2249
# 
# Overall Statistics
# 
# Accuracy : 0.7409          
# 95% CI : (0.7268, 0.7546)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4856          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.61735  0.00000  0.00000  0.39045  0.30394   0.9943
# Specificity           0.97335  1.00000  1.00000  0.99887  0.99653   0.4447
# Pos Pred Value        0.80488      NaN      NaN  0.97203  0.91608   0.7133
# Neg Pred Value        0.93457  0.96992  0.96504  0.94209  0.91994   0.9824
# Prevalence            0.15116  0.03008  0.03496  0.09152  0.11080   0.5815
# Detection Rate        0.09332  0.00000  0.00000  0.03573  0.03368   0.5781
# Detection Prevalence  0.11594  0.00000  0.00000  0.03676  0.03676   0.8105
# Balanced Accuracy     0.79535  0.50000  0.50000  0.69466  0.65024   0.7195

rfVarfit_iPhoneNZV <- varImp(RFfit_iPhoneNZV)
rfVarfit_iPhoneNZV
# Overall
# iphone        100.000
# samsunggalaxy  32.296
# iphonedisunc   20.010
# iphonedisneg   16.201
# iphonedispos   10.283
# iphonecamunc    7.967
# iphoneperpos    6.645
# iphonecampos    4.611
# iphoneperneg    2.117
# iphoneperunc    0.000

## -- iPhone| Recursive Feature Elimination Data Set -- ##

set.seed(seed)
RFfit_iPhoneRFE <- train(iphonesentiment~., data=trainSetRFEiPhone, method="rf", trControl=rfControl)
RFfit_iPhoneRFE

# mtry  Accuracy   Kappa    
# 2    0.7413835  0.4792763
# 10    0.7713321  0.5607395
# 18    0.7640653  0.5507399
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 10.


saveRDS(RFfit_iPhoneRFE, file = "RFfit_iPhoneRFE.rds")
RFfit_iPhoneRFE <- readRDS("RFfit_iPhoneRFE.rds")

rfPred_iPhoneRFE <- predict(RFfit_iPhoneRFE, testSetRFEiPhone)
rfPred_iPhoneRFE

postResample(rfPred_iPhoneRFE, testSetRFEiPhone$iphonesentiment)
# Accuracy     Kappa 
# 0.7735219 0.5613390 

cmRF_iPhoneRFE <- confusionMatrix(data = rfPred_iPhoneRFE, reference = testSetRFEiPhone$iphonesentiment)
cmRF_iPhoneRFE

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0  379    1    1    0    5    9
# 1    1    0    0    0    0    1
# 2    1    1   17    0    0    3
# 3    2    0    1  236    3    5
# 4    3    1    1    4  143   10
# 5  202  114  116  116  280 2234
# 
# Overall Statistics
# 
# Accuracy : 0.7735        
# 95% CI : (0.76, 0.7866)
# No Information Rate : 0.5815        
# P-Value [Acc > NIR] : < 2.2e-16     
# 
# Kappa : 0.5613        
# 
# Mcnemar's Test P-Value : NA            
# 
# Statistics by Class:
# 
#                      Class: 0  Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.64456 0.0000000 0.125000  0.66292  0.33179   0.9876
# Specificity           0.99515 0.9994699 0.998668  0.99689  0.99451   0.4914
# Pos Pred Value        0.95949 0.0000000 0.772727  0.95547  0.88272   0.7296
# Neg Pred Value        0.94020 0.9699074 0.969235  0.96706  0.92275   0.9662
# Prevalence            0.15116 0.0300771 0.034961  0.09152  0.11080   0.5815
# Detection Rate        0.09743 0.0000000 0.004370  0.06067  0.03676   0.5743
# Detection Prevalence  0.10154 0.0005141 0.005656  0.06350  0.04165   0.7871
# Balanced Accuracy     0.81986 0.4997350 0.561834  0.82990  0.66315   0.7395

rfVarfit_iPhoneRFE <- varImp(RFfit_iPhoneRFE)
rfVarfit_iPhoneRFE

# Overall
# iphone        100.0000
# iphonedisunc   32.6975
# samsunggalaxy  32.1121
# iphonedisneg   29.9040
# htcphone       28.8644
# googleandroid  25.5088
# iphoneperpos   24.3953
# iphonedispos   17.8512
# iphonecampos   12.9590
# iphonecamunc   12.8333
# iphoneperneg   11.4233
# iphoneperunc    9.8775
# iphonecamneg    8.9669
# htccampos       3.6611
# sonyxperia      2.5022
# ios             1.2795
# htcperpos       0.8859
# htcdisunc       0.0000


### --- iPhone Model Data --- ###
ModelData_iPhone <- resamples(list(RFfit_iPhoneOOB=RFfit_iPhoneOOB, RFfit_iPhoneCOR=RFfit_iPhoneCOR, RFfit_iPhoneNZV=RFfit_iPhoneNZV, RFfit_iPhoneRFE=RFfit_iPhoneRFE))
ModelData_iPhone

class(ModelData_iPhone)
summary(ModelData_iPhone)

# Accuracy 
#                     Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# RFfit_iPhoneOOB 0.7599119 0.7657571 0.7728300 0.7738634 0.7820352 0.7931793    0
# RFfit_iPhoneCOR 0.7555066 0.7638954 0.7705029 0.7719925 0.7794604 0.7920792    0
# RFfit_iPhoneNZV 0.7359736 0.7385928 0.7455947 0.7452398 0.7497942 0.7601760    0
# RFfit_iPhoneRFE 0.7588106 0.7627300 0.7689790 0.7713321 0.7801051 0.7865787    0
# 
# Kappa 
#                      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# RFfit_iPhoneOOB 0.5332005 0.5456836 0.5628832 0.5645740 0.5831302 0.6082148    0
# RFfit_iPhoneCOR 0.5251136 0.5416703 0.5585688 0.5611163 0.5773964 0.6056825    0
# RFfit_iPhoneNZV 0.4775717 0.4796900 0.4956698 0.4968294 0.5084065 0.5312249    0
# RFfit_iPhoneRFE 0.5318177 0.5418360 0.5559600 0.5607395 0.5806099 0.5976940    0


## -- Best Performing Model and Data Set for iPhone -- ##
# Accuracy
# RFfit_iPhoneOOB 0.7599119 0.7657571 0.7728300 0.7738634 0.7820352 0.7931793    0
# Kappa
# RFfit_iPhoneOOB 0.5332005 0.5456836 0.5628832 0.5645740 0.5831302 0.6082148    0





## -- Galaxy| Correlation Data Set -- ##

set.seed(seed)
c5.0fit_galaxyCOR <- train(galaxysentiment~., data=trainSetCORgalaxy, method="C5.0", trControl=c5.0Control)
c5.0fit_galaxyCOR

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8136, 8136, 8136, 8135, 8137, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7653778  0.5291839
# rules  FALSE   10      0.7554228  0.5097624
# rules  FALSE   20      0.7554228  0.5097624
# rules   TRUE    1      0.7660414  0.5308805
# rules   TRUE   10      0.7524366  0.5016800
# rules   TRUE   20      0.7524366  0.5016800
# tree   FALSE    1      0.7631668  0.5250334
# tree   FALSE   10      0.7575234  0.5166521
# tree   FALSE   20      0.7575234  0.5166521
# tree    TRUE    1      0.7640513  0.5267765
# tree    TRUE   10      0.7549798  0.5090347
# tree    TRUE   20      0.7549798  0.5090347
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.


saveRDS(c5.0fit_galaxyCOR, file = "c5.0fit_galaxyCOR.rds")
c5.0fit_galaxyCOR <- readRDS("c5.0fit_galaxyCOR.rds")

c5.0Pred_galaxyCOR <- predict(c5.0fit_galaxyCOR, testSetCORgalaxy)
c5.0Pred_galaxyCOR

postResample(c5.0Pred_galaxyCOR, testSetCORgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7672436 0.5332168

cmc5.0_galaxyCOR <- confusionMatrix(data = c5.0Pred_galaxyCOR, reference = testSetCORgalaxy$galaxysentiment)
cmc5.0_galaxyCOR

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0  351    0    3    4   10   27
# 1    0    0    0    0    0    0
# 2    1    0   17    0    0    0
# 3    3    3    0  216    5   25
# 4    5    0    1    3  117   16
# 5  148  111  114  129  293 2269
# 
# Overall Statistics
# 
# Accuracy : 0.7672          
# 95% CI : (0.7536, 0.7805)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5332          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.69094  0.00000 0.125926  0.61364  0.27529   0.9709
# Specificity           0.98692  1.00000 0.999732  0.98977  0.99275   0.4817
# Pos Pred Value        0.88861      NaN 0.944444  0.85714  0.82394   0.7405
# Neg Pred Value        0.95483  0.97055 0.969375  0.96242  0.91740   0.9157
# Prevalence            0.13123  0.02945 0.034875  0.09093  0.10979   0.6037
# Detection Rate        0.09067  0.00000 0.004392  0.05580  0.03022   0.5862
# Detection Prevalence  0.10204  0.00000 0.004650  0.06510  0.03668   0.7915
# Balanced Accuracy     0.83893  0.50000 0.562829  0.80170  0.63402   0.7263


c5.0Varfit_galaxyCOR <- varImp(c5.0fit_galaxyCOR)
c5.0Varfit_galaxyCOR

# Overall
# sonyxperia    100.00000
# samsungcamunc  99.43107
# iphone         17.67754
# iphonedisneg   10.48461
# googleandroid   9.54993
# iphoneperpos    7.73138
# iphonedispos    7.33516
# htccampos       7.28436
# samsungperneg   7.20309
# iphonecamneg    4.65305
# iphonedisunc    4.47018
# iphonecampos    3.22056
# samsunggalaxy   2.14366
# iphoneperneg    2.02174
# samsungperpos   1.97094
# iphoneperunc    1.66616
# iosperpos       1.19882
# htcdisneg       0.06096
# sonycamunc      0.00000
# htcdisunc       0.00000




## -- Galaxy| Near Zero Variance Data Set -- ##

set.seed(seed)
c5.0fit_galaxyNZV <- train(galaxysentiment~., data=trainSetNZVgalaxy, method="C5.0", trControl=c5.0Control)
c5.0fit_galaxyNZV

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8136, 8136, 8136, 8135, 8137, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7521027  0.4967127
# rules  FALSE   10      0.7344069  0.4552173
# rules  FALSE   20      0.7344069  0.4552173
# rules   TRUE    1      0.7522137  0.4973220
# rules   TRUE   10      0.7338543  0.4523551
# rules   TRUE   20      0.7338543  0.4523551
# tree   FALSE    1      0.7512178  0.4953727
# tree   FALSE   10      0.7348490  0.4585663
# tree   FALSE   20      0.7348490  0.4585663
# tree    TRUE    1      0.7511078  0.4954581
# tree    TRUE   10      0.7303161  0.4493145
# tree    TRUE   20      0.7303161  0.4493145
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.


saveRDS(c5.0fit_galaxyNZV, file = "c5.0fit_galaxyNZV.rds")
c5.0fit_galaxyNZV <- readRDS("c5.0fit_galaxyNZV.rds")

c5.0Pred_galaxyNZV <- predict(c5.0fit_galaxyNZV, testSetNZVgalaxy)
c5.0Pred_galaxyNZV

postResample(c5.0Pred_galaxyNZV, testSetNZVgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7514854 0.4936633


cmc5.0_galaxyNZV <- confusionMatrix(data = c5.0Pred_galaxyNZV, reference = testSetNZVgalaxy$galaxysentiment)
cmc5.0_galaxyNZV

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0  352    0   20    3   11   24
# 1    0    0    0    0    0    0
# 2    0    0    0    0    0    0
# 3    4    3    0  170    3   20
# 4    5    1    2    3  112   18
# 5  147  110  113  176  299 2275
# 
# Overall Statistics
# 
# Accuracy : 0.7515         
# 95% CI : (0.7376, 0.765)
# No Information Rate : 0.6037         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.4937         
# 
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.69291  0.00000  0.00000  0.48295  0.26353   0.9735
# Specificity           0.98275  1.00000  1.00000  0.99147  0.99158   0.4492
# Pos Pred Value        0.85854      NaN      NaN  0.85000  0.79433   0.7292
# Neg Pred Value        0.95493  0.97055  0.96513  0.95042  0.91609   0.9174
# Prevalence            0.13123  0.02945  0.03487  0.09093  0.10979   0.6037
# Detection Rate        0.09093  0.00000  0.00000  0.04392  0.02893   0.5877
# Detection Prevalence  0.10592  0.00000  0.00000  0.05167  0.03642   0.8060
# Balanced Accuracy     0.83783  0.50000  0.50000  0.73721  0.62756   0.7113

c5.0Varfit_galaxyNZV <- varImp(c5.0fit_galaxyNZV)
c5.0Varfit_galaxyNZV


## -- Galaxy| Recursive Feature Elimination Data Set -- ##

set.seed(seed)
c5.0fit_galaxyRFE <- train(galaxysentiment~., data=trainSetRFEgalaxy, method="C5.0", trControl=c5.0Control)
c5.0fit_galaxyRFE

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8136, 8136, 8136, 8135, 8137, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7652672  0.5291796
# rules  FALSE   10      0.7538730  0.5077408
# rules  FALSE   20      0.7538730  0.5077408
# rules   TRUE    1      0.7659311  0.5301317
# rules   TRUE   10      0.7556450  0.5093166
# rules   TRUE   20      0.7556450  0.5093166
# tree   FALSE    1      0.7628352  0.5248674
# tree   FALSE   10      0.7567509  0.5144032
# tree   FALSE   20      0.7567509  0.5144032
# tree    TRUE    1      0.7639410  0.5265565
# tree    TRUE   10      0.7542051  0.5087959
# tree    TRUE   20      0.7542051  0.5087959
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.


saveRDS(c5.0fit_galaxyRFE, file = "c5.0fit_galaxyRFE.rds")
c5.0fit_galaxyRFE <- readRDS("c5.0fit_galaxyRFE.rds")

c5.0Pred_galaxyRFE <- predict(c5.0fit_galaxyRFE, testSetRFEgalaxy)
c5.0Pred_galaxyRFE


postResample(c5.0Pred_galaxyRFE, testSetRFEgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7682769 0.5339278  


# cmc5.0_galaxyRFE <- confusionMatrix(data = c5.0Pred_galaxyRFE, reference = testSetRFEgalaxy$galaxysentiment)
cmc5.0_galaxyRFE

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1    2    3    4    5
# 0  354    0    2    4    8   26
# 1    0    0    0    0    0    0
# 2    0    0   17    0    0    0
# 3    3    3    1  212    5   20
# 4    5    0    0    2  117   17
# 5  146  111  115  134  295 2274
# 
# Overall Statistics
# 
# Accuracy : 0.7683          
# 95% CI : (0.7547, 0.7815)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5339          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
# Sensitivity           0.69685  0.00000 0.125926  0.60227  0.27529   0.9730
# Specificity           0.98811  1.00000 1.000000  0.99091  0.99304   0.4778
# Pos Pred Value        0.89848      NaN 1.000000  0.86885  0.82979   0.7395
# Neg Pred Value        0.95571  0.97055 0.969382  0.96140  0.91743   0.9209
# Prevalence            0.13123  0.02945 0.034875  0.09093  0.10979   0.6037
# Detection Rate        0.09145  0.00000 0.004392  0.05477  0.03022   0.5874
# Detection Prevalence  0.10178  0.00000 0.004392  0.06303  0.03642   0.7944
# Balanced Accuracy     0.84248  0.50000 0.562963  0.79659  0.63416   0.7254


c5.0Varfit_galaxyRFE <- varImp(c5.0fit_galaxyRFE)
c5.0Varfit_galaxyRFE

# Overall
# iphone        100.0000
# googleandroid   8.9863
# iphoneperpos    8.7661
# samsungperneg   7.0950
# iphonedispos    6.9849
# samsungdisneg   6.9048
# iphonecamunc    6.7547
# iphonedisneg    6.5646
# sonyxperia      6.0743
# iphonedisunc    5.9342
# htccampos       5.7640
# iphonecampos    4.8734
# iphoneperunc    3.3523
# samsunggalaxy   2.8220
# ios             2.0614
# iphoneperneg    1.9013
# iphonecamneg    1.7312
# samsungcamneg   0.2101
# htcdisneg       0.0000
# samsungdisunc   0.0000

### --- Galaxy Model Data --- ###
ModelData_galaxy <- resamples(list(c5.0fit_galaxyOOB=c5.0fit_galaxyOOB, c5.0fit_galaxyCOR=c5.0fit_galaxyCOR, c5.0fit_galaxyNZV=c5.0fit_galaxyNZV, c5.0fit_galaxyRFE=c5.0fit_galaxyRFE))
ModelData_galaxy
class(ModelData_galaxy)
summary(ModelData_galaxy)

# Accuracy 
#                        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c5.0fit_galaxyOOB 0.7580110 0.7602210 0.7652280 0.7658209 0.7706774 0.7765487    0
# c5.0fit_galaxyCOR 0.7522124 0.7627877 0.7678275 0.7660414 0.7712942 0.7765487    0
# c5.0fit_galaxyNZV 0.7444690 0.7471679 0.7498628 0.7522137 0.7562920 0.7654867    0
# c5.0fit_galaxyRFE 0.7580110 0.7602210 0.7646724 0.7659311 0.7706774 0.7787611    0
# 
# Kappa 
#                        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c5.0fit_galaxyOOB 0.5118590 0.5161435 0.5262821 0.5298597 0.5443783 0.5514706    0
# c5.0fit_galaxyCOR 0.5068242 0.5200452 0.5288856 0.5308805 0.5460383 0.5519834    0
# c5.0fit_galaxyNZV 0.4804913 0.4828482 0.4912245 0.4973220 0.5078842 0.5315341    0
# c5.0fit_galaxyRFE 0.5119025 0.5161435 0.5251555 0.5301317 0.5443783 0.5563604    0

## -- Best Performing Model and Data Set for iPhone -- ##
# Accuracy
# c5.0fit_galaxyCOR 0.7522124 0.7627877 0.7678275 0.7660414 0.7712942 0.7765487    0
# Kappa
# c5.0fit_galaxyCOR 0.5068242 0.5200452 0.5288856 0.5308805 0.5460383 0.5519834    0


### --- Predict iPhone Sentiment in Large Matrix --- ###

iPhoneLargeMatrix <- read.csv("iPhoneLargeMatrix.csv", stringsAsFactors = FALSE)
na.omit(iPhoneLargeMatrix$iphonesentiment)

iPhoneLargeMatrixPred <- predict(RFfit_iPhoneOOB, iPhoneLargeMatrix)
iPhoneLargeMatrixPred
summary(iPhoneLargeMatrixPred)
#    0     1     2     3     4     5 
# 9989     2   997  1487   798 11863 



### --- Predict Galaxy Sentiment in Large Matrix --- ###

galaxyLargeMatrix <- read.csv("galaxyLargeMatrix.csv", stringsAsFactors = FALSE)
sum(is.na(galaxyLargeMatrix$galaxysentiment))

galaxyLargeMatrixPred <- predict(c5.0fit_galaxyCOR, galaxyLargeMatrix)
galaxyLargeMatrixPred
summary(galaxyLargeMatrixPred)
#    0     1     2     3     4     5 
# 9983     0   894  1524   649 12086 



####################################



pieData_iPhone <- data.frame(COM = c("Very Negative", "Negative", "Somewhat Negative", "Somewhat Positive", "Positive", "Very Positive"),
                             values = c(9989, 2, 997, 1487, 798, 11863))
pieData_galaxy <- data.frame(COM = c("Very Negative", "Negative", "Somewhat Negative", "Somewhat Positive", "Positive", "Very Positive"),
                             values = c(9983, 0, 894, 1524, 649, 12086))
# create pie chart iphone
iPhoneSpie <- plot_ly(pieData_iPhone, labels = ~COM, values = ~ values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
iPhoneSpie

# create pie chart galaxy
galaxySpie <- plot_ly(pieData_galaxy, labels = ~COM, values = ~ values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'Galaxy Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
galaxySpie
################################################################

pieDF <- data.frame(COM = c("Very Negative", "Negative", "Somewhat Negative", "Somewhat Positive", "Positive", "Very Positive", "Very Negative", "Negative", "Somewhat Negative", "Somewhat Positive", "Positive", "Very Positive"), phone = c("iPhone","iPhone","iPhone","iPhone","iPhone","iPhone", "Galaxy","Galaxy","Galaxy","Galaxy","Galaxy","Galaxy"), value = c(9989, 2, 997, 1487, 798, 11863,9983, 0, 894, 1524, 649, 12086))
pieDF

iPhonePIE <- filter(pieDF, phone == "iPhone")
iPhonePIE
galaxyPIE <- filter(pieDF, phone == "Galaxy")
galaxyPIE

################################################################

iPhoneSbar <- plot_ly(pieData_iPhone, type = "bar") %>%
                     
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
iPhoneSbar

