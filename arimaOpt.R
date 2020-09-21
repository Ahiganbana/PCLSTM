library("tsbox")
dta <- read.csv("sudeste.csv")
 dta <- dta[,-1]
library(tsbox)
d <- ts_ts(dta)
d
#############################
#############################
###########################
##############################
library("rlist")
liste <- c(0,1,2)
sonuclar1 <- c()
sonuclar2 <- c()
for (i in liste){
system.time({
library(ggplot2)
library(forecast)
WWWusage %>%
Arima(order=c(3,1,0)) %>%
forecast(h=20) %>%
autoplot
# Fit model to first few years of AirPassengers data
air.model <- Arima(window(d,end=2020+11/12),order=c(0,1,2),
seasonal=list(order=c(0,1,2),period=12),lambda=0)
plot(forecast(air.model,h=48),ylim=c(10,50))
lines(d)
# Apply fitted model to later data
air.model2 <- Arima(window(d,start=1990),model=air.model)
# Forecast accuracy measures on the log scale.
# in-sample one-step forecasts.
accuracy(air.model)
# out-of-sample one-step forecasts.
sonuc <- accuracy(air.model2)})
sonuclar1 <- list.append(sonuclar1,sonuc[2])
sonuclar2 <- list.append(sonuclar2,i)
}
###################################
liste2 <- c(2,3)
indis <-1
min <- sonuclar1[1]
for(i in liste2){
if(sonuclar1[i]< min){
min <- sonuclar1[i]
indis <- i
}
}
liste[indis]
###############PARALEL KISIM############################################
hesapla <- function(vektor){
system.time({
library(ggplot2)
library(forecast)
WWWusage %>%
Arima(order=c(3,1,0)) %>%
forecast(h=20) %>%
autoplot
# Fit model to first few years of AirPassengers data
air.model <- Arima(window(d,end=2020+11/12),order=c(0,1,2),
seasonal=list(order=c(0,1,2),period=12),lambda=0)
plot(forecast(air.model,h=48),ylim=c(10,50))
lines(d)
# Apply fitted model to later data
air.model2 <- Arima(window(d,start=1990),model=air.model)
# Forecast accuracy measures on the log scale.
# in-sample one-step forecasts.
accuracy(air.model)
# out-of-sample one-step forecasts.
sonuc <- accuracy(air.model2)})
vektor[1] <- vektor[1]+1
}
###########################################
library("parallel")
vektor <- c(1)
system.time(save2 <- mclapply(vektor, hesapla))
