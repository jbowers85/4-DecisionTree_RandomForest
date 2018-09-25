#__________________________________________________________________________________
# DECISION TREE & RANDOM FOREST - SURVIVAL CLASSIFICATION 
# TITANIC DATA
#
# By: James Bowers
#
# collapse sections:  Alt + O
# expand sections:    Shift + Alt + O
#
#___________________________________________________________________________________

## 1.) DECISION TREE - CLASSIFICATION - BINARY DEPENDENT VARIABLE ####

### Get Data ####
titanic.train <- read.csv("Titanic_train.csv", header = TRUE)
head(titanic.train)

titanic.test <- read.csv("Titanic_test.csv", header = TRUE)
head(titanic.test)


### Profile ####
survived <- titanic.train$Survived # retain classification for training

titanic.combined <- rbind(titanic.train[, -2], titanic.test) # create a combined data frame to for profiling/cleansing

sapply(titanic.combined, function(y) sum(length(which(is.na(y))))) # NA count
dim(titanic.combined)
str(titanic.combined)
summary(titanic.combined)


### Cleanse #### 
#### convert data types
titanic.combined$Name <- as.character(titanic.combined$Name)
titanic.combined$Ticket <- as.character(titanic.combined$Ticket)
titanic.combined$Cabin <- as.character(titanic.combined$Cabin)

titanic.combined$Pclass <- as.factor(titanic.combined$Pclass)
titanic.combined$Sex <- as.factor(titanic.combined$Sex)

str(titanic.combined)

### Handle Nulls ####

#### Variable: Fare
titanic.combined$Fare[is.na(titanic.combined$Fare)] <- median(titanic.combined$Fare, na.rm = TRUE)

#### Variable: Age
titanic.combined$Age[is.na(titanic.combined$Age)] <- median(titanic.combined$Age, na.rm = TRUE)


### Build Model ####

#### Split data back into testing and training sets
titanic.train <- titanic.combined[1:891, c(2,4,5,6,7,9,11)]
titanic.train$Survived <- as.factor(survived)

titanic.test <- titanic.combined[892:1309, c(1,2,4,5,6,7,9,11)]

str(titanic.test)
str(titanic.train)


library(caret)
####  set up training controls
caret.control <- trainControl(method = "cv", number = 10)

####  train model using RandomForest
dt.cv <- train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
               data = titanic.train,
               method = "rpart",
               trControl = caret.control,
               # na.action = na.pass,
               minsplit = 5, cp = 0.01, xval = 10, maxdepth = 10)


#### retrieve trained model using the best parameters from all the data
dt.best <- dt.cv$finalModel

#### plot the decision tree
plot(dt.best)
text(dt.best)


library(rattle)
fancyRpartPlot(dt.best, main = "Decision Tree", sub = "", caption = "TESTER", type=5)


### Predict ####
preds <- predict(dt.cv, titanic.test, type = "raw") # or use class="prob" to get percentages so we can adjust threshold manually. 


### Save Results ####
submission <- data.frame(PassengerId = titanic.test$PassengerId, Survived = preds)

#### Write out a .CSV suitable for Kaggle submission
write.csv(submission, file = "Kaggle_Titanic.csv", row.names = FALSE)



## 2.) RANDOM FOREST CLASSIFICATION - ORDINAL DEPENDENT VARIABLE ####

### Get Data ####
titanic.train <- read.csv("Titanic_train.csv", header = TRUE)
head(titanic.train)

titanic.test <- read.csv("Titanic_test.csv", header = TRUE)
head(titanic.test)


### Profile ####
survived <- titanic.train$Survived # retain survival classification for training

titanic.combined <- rbind(titanic.train[, -2], titanic.test) # create a combined data frame to for profiling/cleansing

sapply(titanic.combined, function(y) sum(length(which(is.na(y))))) # NA count
dim(titanic.combined)
str(titanic.combined)
summary(titanic.combined)

### Cleanse #### 
#### convert data types
titanic.combined$Name <- as.character(titanic.combined$Name)
titanic.combined$Ticket <- as.character(titanic.combined$Ticket)
titanic.combined$Cabin <- as.character(titanic.combined$Cabin)

titanic.combined$Pclass <- as.factor(titanic.combined$Pclass)
titanic.combined$Sex <- as.factor(titanic.combined$Sex)

str(titanic.combined)


### Feature Engineer ####  
titanic.combined$CabinPrefix <- as.factor(substr(titanic.combined$Cabin,1,1))

titanic.combined$FamUnit <- titanic.combined$SibSp + titanic.combined$Parch


titanic.combined$Type <- NA # initialize type feature
#### Miss
missIndex <- grep("Miss", titanic.combined$Name, ignore.case = FALSE)
titanic.combined$Type[missIndex] <- "Miss"
#### Mrs
mrsIndex <- grep("Mrs\\.", titanic.combined$Name, ignore.case = FALSE)
msIndex <- grep("Ms\\.", titanic.combined$Name, ignore.case = FALSE)
mmeIndex <- grep("Mme", titanic.combined$Name, ignore.case = FALSE)
mlleIndex <- grep("Mlle", titanic.combined$Name, ignore.case = FALSE)
titanic.combined$Type[mrsIndex] <- "Mrs"
titanic.combined$Type[msIndex] <- "Mrs"
titanic.combined$Type[mmeIndex] <- "Mrs"
titanic.combined$Type[mlleIndex] <- "Mrs"
#### Mr
mrIndex <- grep("Mr\\.", titanic.combined$Name, ignore.case = FALSE)
titanic.combined$Type[mrIndex] <- "Mr"
sirIndex <- grep("Sir\\.", titanic.combined$Name, ignore.case = FALSE)
titanic.combined$Type[sirIndex] <- "Mr"
titanic.combined$Type[823] <- "Mr" 
#### Master
masterIndex <- grep("Master", titanic.combined$Name, ignore.case = FALSE)
titanic.combined$Type[masterIndex] <- "Master"
#### Other Official Title
drIndex <- grep("Dr\\.", titanic.combined$Name, ignore.case = FALSE)
majorIndex <- grep("Major", titanic.combined$Name, ignore.case = FALSE)
colIndex <- grep("Col\\.", titanic.combined$Name, ignore.case = FALSE)
captIndex <- grep("Capt", titanic.combined$Name, ignore.case = FALSE)
revIndex <- grep("Rev\\.", titanic.combined$Name, ignore.case = FALSE)
donIndex <- grep("Don\\.", titanic.combined$Name, ignore.case = FALSE)
donaIndex <- grep("Dona\\.", titanic.combined$Name, ignore.case = FALSE)
countessIndex <- grep("Countess", titanic.combined$Name, ignore.case = FALSE)
ladyIndex <- grep("Lady", titanic.combined$Name, ignore.case = FALSE)
titanic.combined$Type[drIndex] <- "Other_Official"
titanic.combined$Type[majorIndex] <- "Other_Official"
titanic.combined$Type[colIndex] <- "Other_Official"
titanic.combined$Type[captIndex] <- "Other_Official"
titanic.combined$Type[revIndex] <- "Other_Official"
titanic.combined$Type[donIndex] <- "Other_Official"
titanic.combined$Type[donaIndex] <- "Other_Official"
titanic.combined$Type[countessIndex] <- "Other_Official"
titanic.combined$Type[ladyIndex] <- "Other_Official"

#### convert to factor
titanic.combined$Type <- as.factor(titanic.combined$Type)
table(titanic.combined$Type)
head(titanic.combined)


### Handle Nulls ####

#### Variable: Fare
titanic.combined$Fare[is.na(titanic.combined$Fare)] <- median(titanic.combined$Fare, na.rm = TRUE)

#### Variable: Age

##### Method 1: Multiple Linear Regression
titanic.combined.ageNA <- titanic.combined[is.na(titanic.combined[,5]),]
titanic.combined.ageNotNA <- titanic.combined[!is.na(titanic.combined[,5]),]

pairs(titanic.combined.ageNotNA[,c(5,2,11,14)])

library(car)

m <- lm(Age ~ Pclass+Embarked+Type, data = titanic.combined.ageNotNA)
vifdf <- data.frame(vif(m))
vifdf  # VIF > 10 is large; VIF >5 & <10 is medium 

library(MASS)
step <- stepAIC(m, direction="both", trace=FALSE)
summary(step)$coeff 
summary(step)$adj.r.squared # this model explains 42% of the variation in mpg

ages <- predict(step, titanic.combined.ageNA)

##### Method 2: Imputation using MICE
titanic.imputed <- titanic.combined[ ,c(2, 5, 14)]## keep only necessary columns for imputation
str(titanic.imputed)

library(mice)
tempData <- mice(titanic.imputed,m=5,maxit=50,meth='pmm',seed=500)
summary(tempData)
titanic.imputed <- complete(tempData,1)

###### save to CSV then Read the CSV to get rid of imputed data frame
write.csv(titanic.imputed, file = "titanic.imputed.csv", row.names = FALSE)
titanic.imputed <- read.csv("titanic.imputed.csv")
str(titanic.imputed)

##### Compare Method 1 & Method 2
imputedAges <- titanic.imputed$Age
imputedAges <- imputedAges[titanic.combined.ageNA[,1]]

regressAges <- as.numeric(ages)

data.frame(imputedAges, regressAges)

##### apply values from Method 1 or Method 2
titanic.combined[is.na(titanic.combined[,5]),]$Age <- regressAges
titanic.combined[is.na(titanic.combined[,5]),]$Age <- imputedAges

head(titanic.combined,25)

### Build Model ####
#### Split data back into testing and training sets
titanic.train <- titanic.combined[1:891, c(2,4,5,6,7,9,11,12,14)]
titanic.train$Survived <- as.factor(survived)

titanic.test <- titanic.combined[892:1309, c(2,4,5,6,7,9,11,12,14)]

str(titanic.test)
str(titanic.train)

####  set seed to ensure reproducibility between runs
set.seed(217)

####  set up caret to perform 10-fold cross validation repeated 3 times
library(caret)
caret.control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

####  train model using RandomForest
rf.cv <- train(Survived ~ .
               , 
               data = titanic.train,
               method = "rf",
               trControl = caret.control,
               tuneLength = 7,
               importance = TRUE,
               ntree = 100,
               nodesize = 5,
               maxnodes = 100)

#### Display the results of the cross validation
plot(rf.cv)
rf.cv

#### standard deviation of CV
cat(paste("\nCross validation standard deviation:",  
          sd(rf.cv$resample$Accuracy), "\n", sep = " "))

#### retrieve trained model using the best parameters from all the data
rf.best <- rf.cv$finalModel

#### which variable are important?
varImpPlot(rf.best)

### Predict ####
preds <- predict(rf.cv, titanic.test, type = "raw")

### Save Results ####
submission <- data.frame(PassengerId = titanic.test$PassengerID, Survived = preds)

#### Write out a .CSV suitable for Kaggle submission
write.csv(submission, file = "Kaggle_Titanic.csv", row.names = FALSE)
