# Practical Machine Learning
Andrea Bruna  
January 18th, 2017  



# Introduction 

The following project aims to predict how a population of individual performed weight lifting exercises (correctly or incorrectly) according to training data collected from retail wearable fitness trackers.

The training and data sets are available from the following sites:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

More information are provided at the following link: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

From the page linked above: "_Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)._"

Essentially, all but class "A" records, are related to different mistakes in the execution form of the exercise.

## Initialization and Data Cleaning 

First of all, the required libraries are loaded, the train and test data are downloaded directly from internet and stored as data frame. The random seed set to today's date.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(gbm)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
set.seed(18012017)
```

A brief data analysis showed that the first seven columns contain factors which are not useful for predictions so they were removed. Furthermore I decided to exclude columns where there are even a single NA value in either the train or test data sets and values from the training sets with a variance near to zero (they would not improve the model).


```r
names(train)[1:7]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
train2 <- train[, colSums(is.na(train)) == 0 | colSums(is.na(test)) == 0 ] 
train2 <- train2[, -c(1:7)]

nzvar <- (nearZeroVar(train2, saveMetrics = TRUE))
train2 <- train2[, (nzvar$nzv==FALSE)]


test2 <- test[, colSums(is.na(train)) == 0 | colSums(is.na(test)) == 0]
test2 <- test2[, -c(1:7)]
test2 <- test2[ , (nzvar$nzv == FALSE)]
```

## Splitting training data

The "clean" training set is split: 60% of the rows will be used to train the model, the remaining 40% to validate and estimate the error rate of the prediction model before applying it to the test data. 


```r
partition <- createDataPartition(train2$classe, p = 0.60, list = FALSE)
training <- train2[partition, ]
cv <- train2[-partition, ]
```

## Training Model(s)

Afterwards two prediction models are generated based on the "cleaned"" training data frame using respectively Stochastic Gradient Boosting and Random Forest.


```r
modfitgbm <- train(classe ~ ., data = training, method="gbm", 
                   trControl = trainControl(method="repeatedcv", repeats=3, number=4), 
                   verbose=FALSE)
```

```
## Loading required package: plyr
```

```r
modfitrf <- train(classe ~ ., data = training, method="rf",
                  trControl=trainControl(method="repeatedcv", repeats=3, number=4))
```

## Extimation of o.o.s. error rate

Let's quantify the accuracy of the prediction using the cv set respectively for GBM and RF: 

```r
confusionMatrix(predict(modfitgbm, cv), cv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2187   51    0    0    2
##          B   22 1413   45    5   19
##          C   10   46 1305   41   12
##          D    5    1   15 1227   18
##          E    8    7    3   13 1391
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9588          
##                  95% CI : (0.9542, 0.9631)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9479          
##  Mcnemar's Test P-Value : 1.575e-08       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9798   0.9308   0.9539   0.9541   0.9646
## Specificity            0.9906   0.9856   0.9832   0.9941   0.9952
## Pos Pred Value         0.9763   0.9395   0.9229   0.9692   0.9782
## Neg Pred Value         0.9920   0.9834   0.9902   0.9910   0.9921
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2787   0.1801   0.1663   0.1564   0.1773
## Detection Prevalence   0.2855   0.1917   0.1802   0.1614   0.1812
## Balanced Accuracy      0.9852   0.9582   0.9686   0.9741   0.9799
```

```r
confusionMatrix(predict(modfitrf, cv), cv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232   10    2    0    0
##          B    0 1497   10    0    0
##          C    0   11 1353   29    0
##          D    0    0    3 1254    1
##          E    0    0    0    3 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9912          
##                  95% CI : (0.9889, 0.9932)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9889          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9862   0.9890   0.9751   0.9993
## Specificity            0.9979   0.9984   0.9938   0.9994   0.9995
## Pos Pred Value         0.9947   0.9934   0.9713   0.9968   0.9979
## Neg Pred Value         1.0000   0.9967   0.9977   0.9951   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1908   0.1724   0.1598   0.1837
## Detection Prevalence   0.2860   0.1921   0.1775   0.1603   0.1840
## Balanced Accuracy      0.9989   0.9923   0.9914   0.9873   0.9994
```

The accuracy of the two models is quite close however Random Forest provides even more accurate predictions. The accuracy rate is close to 99.12% therefore I expect the out of sample error rate to be around 0.9% and the predictions to be fairly accurate even using the test data.

## Predicted values

Applying the Random Forest model to the test data, we obtain the following predicted values for "classe":

```r
predict(modfitrf, test2)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
table(predict(modfitrf, test2))
```

```
## 
## A B C D E 
## 7 8 1 1 3
```
