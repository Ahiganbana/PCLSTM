
library(nloptr)
Test_Fun <- function(x) {
mydata <- read.csv("sudeste.csv")
df <- mydata
df[is.na(df)] = 0
# Normalizing features
df <- matrix(as.matrix(df),
             ncol = ncol(df),
             dimnames = NULL)
rangenorm <- function(x) {
    (x - min(x))/(max(x) - min(x))
}
df <- apply(df, 2, rangenorm)
df <- t(df)
n_dim <- 16
seq_len <- 100
num_samples <- 18
# extract only required data from dataset
trX <- df[1:n_dim, 1:1800]
# the label data(next output value) should be one time 
# step ahead of the current output value
trY <- df[16, 1:1800]
trainX <- trX
dim(trainX) <- c(n_dim, seq_len, num_samples)
trainY <- trY
dim(trainY) <- c(seq_len, num_samples)
batch.size <- 32
train_ids <- 1:8
eval_ids <- 9:18
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
                                      num.round = 3, 
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
data <- mx.nd.array(trainX[, , 18, drop = F])
label <- mx.nd.array(trainY[, 18, drop = F])
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
actual <- trainY[, 16]
## Iterate one by one over timestamps
for (i in 1:100) {
    data <- mx.nd.array(trainX[, i, 18, drop = F])
    label <- mx.nd.array(trainY[i, 18, drop = F])
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
sonuc <- toplam2-toplam1
return(sonuc)
}#Test_fun end
##################################

library(nloptr)
# Bounded version of Nelder-Mead

S <- neldermead(0.1, Test_Fun, 0.1, 0.5, nl.info = TRUE)
# $xmin = c(0.7085595, 0.5000000, 0.2500000)
# $fmin = 0.3353605
## End(Not run)
###################################




#####################################

