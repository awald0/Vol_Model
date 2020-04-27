#install.packages(c("quantmod","rugarch","rmgarch"))   # only needed in case you have not yet installed these packages
library(quantmod)
library(rugarch)
#library(rmgarch)


# replace with your directory and uncomment
# setwd("YOUR/COPLETE/DIRECTORY/PATH") 


startDate = as.Date("2007-01-03") #Specify period of time we are interested in
endDate = as.Date("2018-04-30")
 
mydata<-getSymbols("CAT", from = startDate, to = endDate,auto.assign=F)



chartSeries(mydata)


rmydata <- dailyReturn(mydata)

ug_spec = ugarchspec()
print(ug_spec)


ug_spec <- ugarchspec(mean.model=list(armaOrder=c(1,0)))
ugfit = ugarchfit(spec = ug_spec, data = rmydata)
print(ugfit)


ugfit@fit$coef

ug_var <- ugfit@fit$var   # save the estimated conditional variances
ug_res2 <- (ugfit@fit$residuals)^2   # save the estimated squared residuals

plot(ug_res2, type = "l")
lines(ug_var, col = "green")

ugfore <- ugarchforecast(ugfit, n.ahead = 10)
print(ugfore)

ug_f <- ugfore@forecast$sigmaFor
plot(ug_f, type = "l")

ug_var_t <- c(tail(ug_var,20),rep(NA,10))  # gets the last 20 observations
ug_res2_t <- c(tail(ug_res2,20),rep(NA,10))  # gets the last 20 observations
ug_f <- c(rep(NA,20),(ug_f)^2)

plot(ug_res2_t, type = "l")
lines(ug_f, col = "orange")
lines(ug_var_t, col = "green")
