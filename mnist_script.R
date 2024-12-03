## download data directly:
# http://yann.lecun.com/exdb/mnist/

# or Cybox:
# https://iastate.box.com/s/huwcjw4yp2djhpfky7u079pp1iboz75s

## reading in data:
#load_mnist()
mnist = load_mnist()

ls()
names(train)
names(test)

head(train$y)

dim(train$x)
# each observations represents a 28 x 28 pixel image (we treat it as 784 dimensional observation)
show_digit(train$x[5,])
train$y[5]

for(i in 1:5) {
  show_digit(train$x[i,])
}

## see corresponding X values for 5th observation in my training set?
train$x[5,200:220]

labels <- paste(train$y[1:25],collapse = ", ")
par(mfrow=c(5,5), mar=c(0.1,0.1,0.1,0.1))
for(i in 1:25) show_digit(train$x[i,], axes=F)

table(train$y)

## apply KNN classifier on MNIST dataset

library(class)
# do not run unless you've saved your work (very slow)
#k5 = knn(train$x, test$x,train$y, k=5)

index.train = sample(1:dim(train$x)[1], 2000, replace=FALSE)
index.test = sample(1:dim(test$x)[1], 100, replace=FALSE)

#k5 = knn(train$x[index.train,], test$x[index.test,],train$y[index.train], k=5)

## how to choose k? cross-validation

k_folds <- 10
K <- c(1,5,7,9)
library(caret)
flds <- createFolds(train$y[index.train], k = k_folds, list = TRUE)
names(flds)
cv_error = matrix(NA,5,4)

for(j in 1:length(K)) {
  k <- K[j]
  for(i in 1:5) {
    test_index <- flds[[i]]
    trainX <- train$x[-test_index,]
    testX <- train$x[test_index,]
    trainY <- train$y[-test_index]
    testY <- train$y[test_index]
    knn.pred = knn(trainX, testX, trainY,k=k)
    cv_error[i,j] = mean(testY!=knn.pred)
  }
  print(K[j]) #This just lets me know its running
}
cv_error
MSE <- apply(cv_error,2,mean)
best_model <- K[which.min(MSE)]
best_model

k_best = knn(train$x[index.train,], test$x[index.test,],train$y[index.train], k=best_model)

conf_matrix <- confusionMatrix(as.factor(k_best), as.factor(test$y[index.test]))
conf_matrix
mcr <- 1-conf_matrix$overall["Accuracy"]
mcr




library(MASS)
lda_model <- lda(train$x, grouping = train$y)
lda_pred <- predict(lda_model, newdata = test$x)$class
