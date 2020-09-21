library("rBayesianOptimization")
library("readr")
library("dplyr")
library("mxnet")
library("abind")
library("numbers")
Test_Fun <- function(x) {
mydata <- read.csv("natal.csv")
# generating datasets
len<-4400
df <- data.frame(dim1= numeric(len),dim2= numeric(len),dim3= numeric(len),dim4= numeric(len),dim5= numeric(len),dim6= numeric(len),dim7= numeric(len))
df$dim1 <- sin(pi/12 * (1:(len)))
df$dim2 <- sin(5 + pi/6 * (1:(len)))
df$dim3 <- sin(15 + pi/8 * (1:(len)))
df$dim4 <- sin(55 + pi/10 * (1:(len)))
df$dim5 <- sin(2 + pi/16 * (1:(len)))
df$dim6 <- sin(1+ pi/7 * (1:(len)))
df$dim7 <- 7*df$dim1+ 5*df$dim2+ 17*df$dim3+ 25*df$dim4 +   19*df$dim5+ 21*df$dim6
df <- mydata
# Normalizing features
df <- matrix(as.matrix(df),
             ncol = ncol(df),
             dimnames = NULL)
rangenorm <- function(x) {
    (x - min(x))/(max(x) - min(x))
}
df <- apply(df, 2, rangenorm)
df <- t(df)
n_dim <- 6
seq_len <- 10
num_samples <- 20
# extract only required data from dataset
trX <- df[1:n_dim, 1:200]
# the label data(next output value) should be one time 
# step ahead of the current output value
trY <- df[6, 1:200]
trainX <- trX
dim(trainX) <- c(n_dim, seq_len, num_samples)
trainY <- trY
dim(trainY) <- c(seq_len, num_samples)
batch.size <- 32
train_ids <- 1:6
eval_ids <- 7:10
## create data iterators
train.data <- mx.io.arrayiter(data = trainX[,,train_ids, drop = F],          label = trainY[, train_ids], batch.size = batch.size,             shuffle = TRUE)

eval.data <- mx.io.arrayiter(data = trainX[,,eval_ids, drop = F], label = trainY[, eval_ids], batch.size = batch.size, shuffle = FALSE)
## Create the symbol for RNN
symbol <- rnn.graph(num_rnn_layer = 2,
                    num_hidden = 50,
                    input_size = NULL,
                    num_embed = NULL,
                    num_decode = 1,
                    masking = F, 
                    loss_output = "linear",
                    dropout =x, 
                    ignore_label = -1, 
                    cell_type = "lstm", 
                    output_last_state = T,
                    config = "one-to-one")
mx.metric.mse.seq <- mx.metric.custom("MSE", function(label, pred) {
    label = mx.nd.reshape(label, shape = -1)
    pred = mx.nd.reshape(pred, shape = -1)
    res <- mx.nd.mean(mx.nd.square(label - pred))
    return(as.array(res))
})
ctx <- mx.cpu()
initializer <- mx.init.Xavier(rnd_type = "gaussian",
                              factor_type = "avg", 
                              magnitude = 1)
optimizer <- mx.opt.create("adadelta",
                           rho = 0.9, 
                           eps = 1e-06, 
                           wd = 1e-06, 
                           clip_gradient = 1, 
                           rescale.grad = 1/batch.size)
logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 10, 
                                                   logger = logger)
## train the network
system.time(model <- mx.model.buckets(symbol = symbol, 
                                      train.data = train.data, 
                                      eval.data = eval.data,
                                      num.round = 200, 
                                      ctx = ctx, 
                                      verbose = TRUE, 
                                      metric = mx.metric.mse.seq, 
                                      initializer = initializer,
                                      optimizer = optimizer, 
                                      batch.end.callback = NULL, 
                                      epoch.end.callback=   epoch.end.callback))
## Extracting the state symbols for RNN

internals <- model$symbol$get.internals()
sym_state <- internals$get.output(which(internals$outputs %in% "rnn.state"))
sym_state_cell <- internals$get.output(which(internals$outputs %in%  "rnn.state.cell"))
sym_output <- internals$get.output(which(internals$outputs %in% "loss_output"))
symbol <- mx.symbol.Group(sym_output, sym_state, sym_state_cell)

pred_length <- 100
predicted <- numeric()
data <- mx.nd.array(trainX[, , 6, drop = F])
label <- mx.nd.array(trainY[, 6, drop = F])
infer.data <- mx.io.arrayiter(data = data, 
                              label = label, 
                              batch.size = 1, 
                              shuffle = FALSE)
infer <- mx.infer.rnn.one(infer.data = infer.data, 
                          symbol = symbol, 
                          arg.params = model$arg.params,
                          aux.params = model$aux.params, 
                          input.params = NULL, 
                          ctx = ctx)
actual <- trainY[, 6]
## Iterate one by one over timestamps
for (i in 1:10) {
    data <- mx.nd.array(trainX[, i, 6, drop = F])
    label <- mx.nd.array(trainY[i, 6, drop = F])
    infer.data <- mx.io.arrayiter(data = data, 
                                  label = label, 
                                  batch.size = 1, 
                                  shuffle = FALSE)
    ## use previous RNN state values
    infer <- mx.infer.rnn.one(infer.data = infer.data,
                              symbol = symbol,
                              ctx = ctx, 
                              arg.params = model$arg.params,
                              aux.params = model$aux.params, 
                              input.params = 
                              list(rnn.state=infer[[2]], 
                              rnn.state.cell = infer[[3]]))
    pred <- infer[[1]]
    predicted <- c(predicted, as.numeric(as.array(pred)))
}
deger1 <- actual *2
deger2 <- predicted+actual
deger1 <- deger1 *100
deger2 <- deger2 *100
indis1 <- length(deger1)
indis2 <- length(deger2)
list1 <- 1:indis1
list2 <- 1:indis2
toplam1 <- 0
toplam2 <- 0
for(i in list1){
	toplam1 <- toplam1 + deger1[i]
}
for(i in list2){
	toplam2 <- toplam2 + deger2[i]
}
toplam1 <- round(toplam1)
toplam2 <- round(toplam2)
list(Score = mod(toplam2,toplam1),
Pred = 0)
}#Test_fun end
##################################
###################################
## Set larger init_points and n_iter for better optimization result
hesapla <- function(vektor){
OPT_Res <- BayesianOptimization(Test_Fun,
bounds = list(x = c(0.1, 0.2,0.25,0.27,0.3,0.4,0.45,0.5)),
init_points = 2, n_iter = 1,
acq = "ucb", kappa = 2.576, eps = 0.0,
verbose = TRUE)
vektor[1] <- vektor[1]+1
}

#####################################
library("parallel")
vektor <- c(1)
system.time(save2 <- mclapply(vektor, hesapla))