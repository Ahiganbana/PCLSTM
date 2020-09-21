library("tsbox")
dta <- read.csv("Natal.csv")
 dta <- dta[,-1]
library(tsbox)
d <- ts_ts(dta)
d
#############################
#############################
###########################
##############################
system.time({
library(ggplot2)
library(forecast)
WWWusage %>%
Arima(order=c(3,1,0)) %>%
forecast(h=20) %>%
autoplot
# Fit model to first few years of AirPassengers data
air.model <- Arima(window(d,end=2020+11/12),order=c(0,1,1),
seasonal=list(order=c(0,1,1),period=12),lambda=0)
plot(forecast(air.model,h=48))
lines(d)
# Apply fitted model to later data
air.model2 <- Arima(window(d,start=1990),model=air.model)
# Forecast accuracy measures on the log scale.
# in-sample one-step forecasts.
accuracy(air.model)
# out-of-sample one-step forecasts.
accuracy(air.model2)})