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

A brief data analysis showed that the first seven columns contain factors which are not useful for predictions so they were removed. Furthermore I decided to exclude columns where there are even a single NA value in either the train or test data sets and values from the training sets with a variance near to zero (they would largely increase the calculation time without improving the predictive model so much).


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

## Training data split

The "clean" training set is split: 60% of the rows will be used to train the model, the remaining 40% to validate the prediction model before applying it to the test data. (Note of the Author: I am aware the training set would benefit from a larger data set, unfortunately I had to decrease the training populationdue to performance issues with my current hardware configuration...)


```r
training <- train2[createDataPartition(train2$classe, p = 0.60, list = FALSE), ]
cv <- train2[-createDataPartition(train2$classe, p = 0.40, list = FALSE), ]
```

## Training Model(s)

Afterwards two prediction models are generated based on the "cleaned"" training data frame using respectively Stochastic Gradient Boosting and Random Forest with 4-folds without repetition (Note: again I had to partially sacrifice accuracy to obtain "acceptable" elaboration times...) 


```r
modfitgbm <- train(classe ~ ., data = training, method="gbm", 
                   trControl = trainControl(method="cv", number=4), 
                   verbose=FALSE)
```

```
## Loading required package: plyr
```

```r
modfitrf <- train(classe ~ ., data = training, method="rf",
                  trControl=trainControl(method="cv", number=4))
```

Below the confusion matrixes, respectively, for the training and cross-validation sets: 

```r
confusionMatrix(predict(modfitgbm, training), training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3318   54    0    1    5
##          B   19 2189   48    4    6
##          C    6   34 1988   49   20
##          D    5    1   17 1870   16
##          E    0    1    1    6 2118
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9751          
##                  95% CI : (0.9721, 0.9779)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9685          
##  Mcnemar's Test P-Value : 3.871e-12       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9910   0.9605   0.9679   0.9689   0.9783
## Specificity            0.9929   0.9919   0.9888   0.9960   0.9992
## Pos Pred Value         0.9822   0.9660   0.9480   0.9796   0.9962
## Neg Pred Value         0.9964   0.9905   0.9932   0.9939   0.9951
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2818   0.1859   0.1688   0.1588   0.1799
## Detection Prevalence   0.2869   0.1924   0.1781   0.1621   0.1805
## Balanced Accuracy      0.9920   0.9762   0.9783   0.9825   0.9887
```

```r
confusionMatrix(predict(modfitgbm, cv), cv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3310   61    0    2    5
##          B   21 2162   55    4   17
##          C    9   51 1974   54   21
##          D    6    1   19 1863   23
##          E    2    3    5    6 2098
## 
## Overall Statistics
##                                           
##                Accuracy : 0.969           
##                  95% CI : (0.9657, 0.9721)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9608          
##  Mcnemar's Test P-Value : 4.71e-13        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9886   0.9491   0.9615   0.9658   0.9695
## Specificity            0.9919   0.9898   0.9861   0.9950   0.9983
## Pos Pred Value         0.9799   0.9571   0.9360   0.9744   0.9924
## Neg Pred Value         0.9955   0.9878   0.9918   0.9933   0.9932
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2812   0.1837   0.1677   0.1583   0.1782
## Detection Prevalence   0.2870   0.1919   0.1792   0.1624   0.1796
## Balanced Accuracy      0.9903   0.9694   0.9738   0.9804   0.9839
```

```r
confusionMatrix(predict(modfitrf, training), training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(predict(modfitrf, cv), cv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3346    9    0    0    0
##          B    1 2261    5    0    0
##          C    1    8 2045    9    1
##          D    0    0    3 1918    2
##          E    0    0    0    2 2161
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9965          
##                  95% CI : (0.9953, 0.9975)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9956          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9925   0.9961   0.9943   0.9986
## Specificity            0.9989   0.9994   0.9980   0.9995   0.9998
## Pos Pred Value         0.9973   0.9974   0.9908   0.9974   0.9991
## Neg Pred Value         0.9998   0.9982   0.9992   0.9989   0.9997
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1921   0.1737   0.1629   0.1836
## Detection Prevalence   0.2850   0.1926   0.1753   0.1634   0.1837
## Balanced Accuracy      0.9992   0.9960   0.9971   0.9969   0.9992
```

With the latest versions of the above mentioned R libraries, Random Forest provides more accurate results.

## Prediction

Applying the Random Forest model to the test data:

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
