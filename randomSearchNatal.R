library(nloptr)
library("mxnet")
library("abind")
library("numbers")

Test_Fun <- function(x1) {
dropoutliste <- c()
sonucliste <- c()
count <-1 
j <- 1
while(count<11){
 x1 <- runif(1, 0.1, 0.9)
mydata <- read.csv("SaoPaulo.csv")
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
num_samples <- 30
# extract only required data from dataset
trX <- df[1:n_dim, 1:300]
# the label data(next output value) should be one time 
# step ahead of the current output value
trY <- df[6, 1:300]
trainX <- trX
dim(trainX) <- c(n_dim, seq_len, num_samples)
trainY <- trY
dim(trainY) <- c(seq_len, num_samples)
batch.size <- 32
train_ids <- 1:20
eval_ids <- 21:30
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
                    dropout = x1, 
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
data <- mx.nd.array(trainX[, , 30, drop = F])
label <- mx.nd.array(trainY[, 30, drop = F])
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
    data <- mx.nd.array(trainX[, i, 18,drop = F])
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

indis <- length(actual)
liste <- 1:indis
toplam1 <- 0
toplam2 <- 0
for( i in liste){
	toplam1 <- toplam1 + actual[i]
	toplam2 <- toplam2 + predicted[i]
}
sonuc <- toplam1 - toplam2
if(sonuc<0) sonuc <- sonuc * -1
dropoutliste[j] <- x1
sonucliste[j] <- sonuc
j <- j+1
count <- count+1
}
#######en iyi dropout bul###########################
j <- 1
dizi <- 1:10
sonucIndis <- 1
min <- sonucliste[1]
for(j in dizi)
{
	if(sonucliste[j]<min){
		min <- sonucliste[j]
	sonucIndis <- j
	}
}
print(dropoutliste[sonucIndis])
}##Test_fun end line
###########################
hesapla <- function(vektor){
 x1 <- runif(1, 0.1, 0.9)
Test_Fun(x1)
vektor[1] <- vektor[1]+1
}
library("parallel")
vektor <- c(1)
system.time(save2 <- mclapply(vektor, hesapla))