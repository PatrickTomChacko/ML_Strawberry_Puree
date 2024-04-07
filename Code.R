
x_w <- matrix( c(-0.70,0.18,0.65,1.12,0.89,0.52,-0.31,0.29,-0.60,2.13),nrow = 5,byrow = T)
x_int <- c(1,1,1,1,1)                #adding coefficient for intercept term
x <- cbind(x_int,x_w)
x
y <- c(0,1,1,0,1)
w <- matrix(c(0.5,1.1,-0.3))#removing the intercept

pr <- exp(x%*%w)/( 1+exp(x%*%w) )            

sum = 0
for (i in 1:5) {
sum = sum + (y[i]*log(pr[i])+(1-y[i])*log(1-pr[i]))
}  
N = 5      # we have 5 observations here
cr_ent_loss <- -sum/N
cr_ent_loss

y_hat <- NULL
y_bar <- NULL
for (i in 1:5) {
y_hat[i] <- exp(x[i,]%*%w)/( 1+exp(x[i,]%*%w))
if(y_hat[i]>0.5){y_bar[i] <- 1;
  print(y_bar[i])}
 else y_bar[i] <- 0
}  
Accuracy_calc <- cbind(y_hat,y_bar,y)
colnames(Accuracy_calc) <- c('Pr(y)','Pred_Y','Actual_Y')
Accuracy_df <- as.data.frame(Accuracy_calc)
Accuracy_df

correct <- 0
for (i in 1:5) {
if(Accuracy_df$Pred_Y[i] == Accuracy_df$Actual_Y[i]) correct<- correct +1  
}
correct/N    #N <-  Total number of observations

load("data_hw1_strawberry.RData")   #loaded to global environment
X <- train_data[,-c(1)]    # removing the target variable
dim(X)

class <- as.factor(train_data$target)
table(class)
col <- c("darkorange2", "deepskyblue3") # set colors according to classes
cols <- col[class]

col_no <- as.numeric( gsub("x", "", colnames(X)) )
levels(class)
matplot(t(X), x = col_no, type = "l", lty = 1, col = adjustcolor(cols, 0.5), main = "Training Dataset")
legend("topright", fill = col, legend = levels(class), bty = "n") # add legend

range_norm <- function(x, a = 0, b = 1) {
( (x - min(x)) / (max(x) - min(x)) )*(b - a) + a
}
X_norm <- apply(X, 2, range_norm)
matplot(t(X_norm), x = col_no, type = "l", lty = 1, col = adjustcolor(cols, 0.5),  main = "Normalised training dataset")
legend("topleft", fill = col, legend = levels(class), bty = "n") # add legend


qvec <- 2:10   #range of Q given in question
B <- 100     #number of replicates
err_val <- matrix(NA, B, length(qvec)+1)    #We need 10 columns but here number starts from 1:9 we want from 2:10
best_q <- rep(NA,10)        #To store overall misclassifiction rate

N1 <-nrow(X_norm)        #Total training data set 
N_t <- round(N1*(2/3))  #training set 2/3 
N_c <- N1 - N_t         #1/3 of train_data for cross validation 
set <- sample(N1,N_t)    #out of N1 take a sample of N_t 

for ( b in 1:B ){
# sample randomly training, validation and test data
set <- sample(N1,N_t)    #out of N1 take a sample of N_t
X_val= X_norm[-set,]
Y_val = train_data$target[-set]
X_train <- X_norm[set,]
Y_train <- class[set]
dat_train <- data.frame(Y_train,X_train)

for ( Q in qvec ) {
  
  pca <- prcomp(X_train) # pca
  
  xz_train <- pca$x[,1:Q]    # extract first Q principal component
  dat_train <- data.frame(Y_train, xz_train) # new representation of the training data
  
  # Fitting our model with lower dimensional input variable
  #1000 iterations to ensure convergence
  C1 <- glm(Y_train ~ ., data = dat_train, family = binomial, #fit model
  control = list(maxit = 100))
  
  # Model is ready to predict let's use our validation set, but first need to map
  xz_val <- predict(pca, X_val)[,1:Q]
  
  # Now let us predict 
  preds <- predict.glm(C1, newdata = data.frame(xz_val), type = "response")
  
  y_test_hat <- ifelse(preds > 0.5, "Strawberry","Adulterated")  
  
  # Now calculate the misclassification rate
  calc <- table(Y_val, y_test_hat)

  
  miscalculation = (calc[2]+calc[3])/sum(calc)
  
  err_val[b,Q] = miscalculation    #not working need another work around

 }
}

#Now we will find the mean misclassification error for individual Q, and see which one gives minimum
best_q <- apply(err_val,2,mean)
best_q

#### C2 classifier (Regularized logistic regression model with L1 penalty function)
library('glmnet')
library(e1071) # for 'classAgreement'
# another way of calculating accuracy

class_acc <- function(y, yhat) {
tab <- table(y, yhat)
classAgreement(tab)$diag
}
loss <- function(y, prob) {
-sum( y*log(prob) + (1-y)*log(1-prob) )/length(y)
}
tau <- 0.5        #threshold
S <- 100           #Sample size
lambda <- exp(seq(-3.5, -5.5, length = S-1)) # set sequence for lambda with log pace

lambda <- c(lambda, 0) # this corresponds to plain logistic regression
acc_train <- acc_val <- loss_train <- loss_val <- matrix(NA, B, S)   #placeholders
lambda_best <- rep(NA, B)
x <- X_norm         #our normalised training dataset
y <- as.factor(train_data$target)      #adulterated or strawberry
y <-ifelse(y == "adulterated",0,1)     #mapping my factors to numbers due to requirement of algorithm

for ( b in 1:B ) {
# sample train and validation data
train <- sample(N1,N_t)    #out of N1 take a sample of N_t
val <- setdiff(1:N1,train) #remaining give to val
#sort(union(train,val))  helps to verify uniqueness


# train the model
C2 <- glmnet(x[train,], y[train], family = "binomial", alpha = 1, lambda = lambda)

# obtain predicted classes for training and validation data
p_train <- predict(C2, newx = x[train,], type = "response")
y_train <- apply(p_train, 2, function(v) ifelse(v > tau, 1, 0)) #if prob is greater than Tau assign 1 or else 0,


# prediction for validation set
p_val <- predict(C2, newx = x[val,], type = "response")
y_val <- apply(p_val, 2, function(v) ifelse(v > tau, 1, 0))


# estimate classification accuracy
acc_train[b,] <- sapply( 1:S, function(s) class_acc(y[train], y_train[,s]) ) #this function gives accuracy

#similarly find accuracy of validation set
acc_val[b,] <- sapply( 1:S, function(s) class_acc(y[val], y_val[,s]) )

# compute loss
loss_train[b,] <- sapply( 1:S, function(s) loss(y[train], p_train[,s]) )
loss_val[b,] <- sapply( 1:S, function(s) loss(y[val], p_val[,s]) )


# select lambda which maximizes classification accuracy on validation data
best <- which.max(acc_val[b,])
lambda_best[b] <- lambda[best]
}
lambda_star <- lambda[ which.max( colMeans(acc_val) ) ]
round(lambda_star,2)
matplot(x = lambda, t(acc_train), type = "l", lty = 1, ylab = "Accuracy", xlab = "Lambda",
col = adjustcolor("black", 0.05), log = "y") # accuracy on log scale
matplot(x = lambda, t(acc_val), type = "l",ylim = c(0,1), lty = 1,
col = adjustcolor("deepskyblue2", 0.05), add = TRUE, log = "y")
lines(lambda, colMeans(acc_train), col = "black", lwd = 2)
lines(lambda, colMeans(acc_val), col = "deepskyblue3", lwd = 2)
legend("topright", legend = c("Training accuracy", "Validation accuracy"),
fill = c("black", "deepskyblue2"), bty = "n")
matplot(x = lambda, t(loss_train), type = "l", lty = 1, ylab = "Loss", xlab = "Lambda",
col = adjustcolor("black", 0.05))
matplot(x = lambda, t(loss_val), type = "l", lty = 1,
col = adjustcolor("deepskyblue2", 0.05), add = TRUE, log = "y")
lines(lambda, colMeans(loss_train), col = "black", lwd = 2)
lines(lambda, colMeans(loss_val), col = "deepskyblue3", lwd = 2)
legend("bottomright", legend = c("Training loss", "Validation loss"),
fill = c("black", "deepskyblue2"), bty = "n")
# plot optimal lambdas
abline(v = lambda_star, col = "magenta")
abline(v = lambda[ which.min( colMeans(loss_val) ) ], col = "red")

C2 <- glmnet(X_norm, y, family = "binomial", lambda = lambda_star)
X_test <- test_data[,-c(1)]
X_test <-  apply(X_test, 2, range_norm)
xz_test <- predict(pca, X_test)[,1:10]   #mapping to lower space using optimal Q
# Now let us predict 
  preds <- predict.glm(C1, newdata = data.frame(xz_test), type = "response")
  
  y_test_hat <- ifelse(preds > 0.5, "Strawberry","Adulterated")  
  
  # Now calculate the misclassification rate
  calc <- table(test_data$target, y_test_hat)
  miscalculation = (calc[2]+calc[3])/sum(calc)
  miscalculation
calc

##### C2 classifier
# prediction for validation set
p_test <- predict(C2, newx = X_test, type = "response")
y_test_pred <- apply(p_test, 2, function(v) ifelse(v > tau, 1, 0))


# estimate classification accuracy
diff <- class_acc(y_test_pred,test_data$target) #this function gives accuracy
diff
table(y_test_pred,test_data$target)
129/132
