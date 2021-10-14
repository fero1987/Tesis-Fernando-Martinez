#############   Modelo de Predicción de Ventas mediante técnicas de  Data Mining  ############# 
#############   y Series Temporales en Empresa Alimenticia de Venta Directa       #############

# Alumno: Fernando Martínez
# Tutor: Rodrigo Del Rosso

# Carga de las librerías
  library(tseries)
  library(forecast)
  library(ggplot2)
  library(gridExtra)
  library(car)
  library(nortest)
  library(AdequacyModel)
  library(lmtest)
  library(quantmod)
  library(dygraphs)
  library(lessR)
  library(PASWR2)
  library(dplyr)
  library(psych)
  library(pastecs)
  library(astsa)
  library(tseries)
  library(zoo)
  library(xts)
  library(fma)
  library(expsmooth)
  library(Quandl)
  library(fpp)
  library(urca)
  library(AER)
  library(fUnitRoots)
  library(CADFtest)
  library(fpp2)
  library(datasets)
  library(tidyverse)
  library(magrittr)
  library(ggfortify)
  library(gamlss.data)
  library(vars)
  library(urca)
  library(lmtest)
  library(forecast)
  library(ggplot2)
  library(reshape2)
  library(ggfortify)
  library(readxl)
  library(psych)
  library(DataExplorer)
  library(timetk)
  library(keras)
  library(dplyr)
  library(uroot)
  library(rugarch)

# Limpieza de la memoria
rm( list=ls() )
gc()

# Carga del Dataset
base <- read_xlsx(file.choose()) # Base a usar: Dataset Agregado
head(base)
tail(base)

# Análisis exploratorio de la base
psych::describe(base)
plot_histogram(base) 

# Se define la serie temporal de las ventas
ts <- ts(base[,2], start= 2016, freq = 12)   
ts

# Gráfica de la serie de tiempo
autoplot(ts, ts.colour = 'dark blue')+ ggtitle("Ventas en Unidades") + ylab("")

# Descomposición de la serie temporal
autoplot(decompose(ts, type = "additive"))
autoplot(decompose(ts, type = "multiplicative"))

# Gráfico de la FAS, FAC y FACP de la serie
acf(ts,type = "covariance",plot = T)
ggAcf(ts) + ggtitle("FAC") 
ggAcf(ts,type = "partial") + ggtitle("FACP") 

# Test de Ljung-Box. Si se rechaza H0 significa que hay coeficientes de autocorrelación distintos a cero
Incorrelation(ts,"Ljung-Box") # Ver función autocorrelación al final del script
inco_wn = Incorrelation(ts,"Ljung-Box")
autoplot(ts(inco_wn$P_Value)) + ggtitle("Test de Ljung-Box", subtitle = "P-Value") + ylab("") # Gráfico de los p-value para distintos lags
# El p-value se ubica en 0 por lo que se rechaza H0 y se puede considerar que es una serie no estacionaria 

# Test de raíces unitarias KPSS
kpss.test(ts, null = 'Trend') # 0.01027 - Rechazo H0, no estacionario.
kpss.test(ts, null = 'Level') # 0.1 - No rechazo H0, estacionario.
# Test de raíces unitarias Zivot-Andrews
ur.za(ts, model="both", lag=NULL)
summary(ur.za(ts, model="both", lag=NULL)) 
plot(ur.za(ts, model="both", lag=NULL)) # t: -4.0181 - Estadístico mayor que valores críticos (no rechazo H0), no estacionario.
# Test de raíces unitarias Phillips-Perron
pp.test(ts) # 0.3164 - No rechazo H0, no estacionario.
# Test de raíces unitarias Dickey-Fuller Aumentado
adf.test(ts) # 0.05925 - No rechazo H0, no estacionario.

# Se diferencia 12 lags para remover la estacionalidad
dts0 <- diff(ts, lag = 12)

# Gráfico de la FAS, FAC y FACP de la serie
acf(dts0,type = "covariance",plot = T)
ggAcf(dts0) + ggtitle("FAC") 
ggAcf(dts0,type = "partial") + ggtitle("FACP") 

# Test de Ljung-Box. Si se rechaza H0 significa que hay coeficientes de autocorrelación distintos a cero
Incorrelation(dts0,"Ljung-Box") # Ver función autocorrelación al final del script
inco_wn = Incorrelation(dts0,"Ljung-Box")
autoplot(dts0(inco_wn$P_Value)) + ggtitle("Test de Ljung-Box", subtitle = "P-Value") + ylab("") # Gráfico de los p-value para distintos lags
# El p-value se ubica en 0 por lo que rechazo H0 y puedo considerar que es una serie no estacionaria 

# Test de raíces unitarias KPSS
kpss.test(dts0, null = 'Trend') # 0.1 - No rechazo H0, estacionario.
kpss.test(dts0, null = 'Level') # 0.01 - Rechazo H0, no estacionario
# Test de raíces unitarias Zivot-Andrews
ur.za(dts0, model="both", lag=NULL)
summary(ur.za(dts0, model="both", lag=NULL)) 
plot(ur.za(dts0, model="both", lag=NULL)) # t: -3.3412 - Estadístico mayor que valores críticos (no rechazo H0), no estacionario.
# Test de raíces unitarias Phillips-Perron
pp.test(dts0) # 0.4241 - No rechazo H0, no estacionario.
# Test de raíces unitarias Dickey-Fuller Aumentado
adf.test(dts0) # 0.3442 - No rechazo H0, no estacionario.

# Se realiza la primera diferencia de la serie anterior
dts <- diff(dts0)

# Gráfico de la serie diferenciada
autoplot(dts, ts.colour = 'dark blue')+ ggtitle("Ventas Unidades", subtitle = "Primera Diferencia") + ylab("")

# Gráfico de la FAS, FAC y FACP de la serie diferenciada
acf(dts,type = "covariance",plot = T)
ggAcf(dts) + ggtitle("FAC 1ra dif Ventas") # La serie decrece exponencialmente, lo que indica estacionariedad
ggAcf(dts,type = "partial") + ggtitle("FACP 1ra dif Ventas") 

# Test de Ljung-Box. Si se rechaza H0 significa que hay coeficientes de autocorrelación distintos a cero
Incorrelation(dts,"Ljung-Box") 
inco_wn = Incorrelation(dts,"Ljung-Box")
autoplot(ts(inco_wn$P_Value)) + ggtitle("Test de Ljung-Box", subtitle = "P-Value") + ylab("") # Gráfico de los p-value para distintos lags
# El p-value se ubica por encima de 0.05 por lo que no rechazo H0 y puedo considerar que es una serie estacionaria 

# Test de raíces unitarias KPSS
kpss.test(dts, null = 'Trend') # 0.04395 - Rechazo H0, estacionario.
kpss.test(ts, null = 'Level') # 0.1 - No rechazo H0, estacionario.
# Test de raíces unitarias Zivot-Andrews
ur.za(dts, model="both", lag=NULL)
summary(ur.za(ts, model="both", lag=NULL)) 
plot(ur.za(dts, model="both", lag=NULL)) # -8.4761 - Estadístico menor que valores críticos (rechazo H0), estacionario.
# Test de raíces unitarias Phillips-Perron
pp.test(dts) # 0.01 - Rechazo H0, estacionario.
# Test de raíces unitarias Dickey-Fuller Aumentado
adf.test(dts) # 0.04395 - Rechazo H0, estacionario.

######### Forecast: Se quiere predecir el valor de las ventas para los proximos 7 meses del 07-20 al 01-21

# Conjunto de entrenamiento de la serie original: del 01-16 al 06-20
original_train <- window(ts,start = c(2016,1), end = c(2020,6))
# Conjunto de testeo de la serie original: del 07-20 al 01-21
original_test <- window(ts,start = c(2020,7), end = c(2021,1))

# Gráfico  del conjunto de entrenamiento y el conjunto de testeo de la serie original
ts.plot(original_train,original_test, gpars = list(col = c("black", "red")))

# Conjunto de entrenamiento de la serie diferenciada: del 01-16 al 06-20
train <- window(dts,start = c(2016,1), end = c(2020,6))
# Conjunto de testeo de la serie diferenciada: del 07-20 al 01-21
test <- window(dts,start = c(2020,7), end = c(2021,1))

# Graficamos el conjunto de entrenamiento y el conjunto de testeo de la serie diferenciada
ts.plot(train,test, gpars = list(col = c("black", "red")))

########## Modelo Bayes Ingenuo con Estacionalidad ########## 

# Se plantea el modelo
mie <- snaive(train,h = 7)
summary(mie)
autoplot(mie)

# Gráfico del conjunto de testeo con las prediciones
ts.plot(test,mie$mean , gpars = list(col = c("black", "red")))

# Se realiza la diferencia inversa para llevar las predicciones a la serie original
pred <- c(mie$x, mie$mean)
pred <- cumsum(c(dts0[1], pred))
pred <- diffinv(pred, lag = 12)
pred <- pred + ts[0:12]
tspred <- ts(pred, start= 2016, freq = 12)   # Serie de tiempo de las predicciones
tspred_test <- window(tspred,start = c(2020,7), end = c(2021,1)) # Ventana temporal de la serie de testeo
tspred_test

# Gráfico del conjunto de testeo original con las prediciones
ts.plot(original_test,tspred_test, gpars = list(col = c("black", "red")))

# Gráfico de la serie original con las prediciones
ts.plot(ts,tspred_test, gpars = list(col = c("black", "red")))

# Cálculo del forecast accuracy para testeo
abs <- sum(abs(tspred_test - original_test)) # Diferencia absoluta
real_sales <- sum(original_test) # Ventas reales
(real_sales -abs )/real_sales # 87,64%

########## Modelo de Redes Neuronales NNAR ########## 

# Se inicializa y entrena la red
nn1 = nnetar(y = train, p = 7, P = 0, size = 2) # p = 7 períodos
print(nn1)

# Predicciones para 7 meses
fc1 = forecast(nn1,h=7)
fc1.PI = forecast(nn1,h=7, PI = T)
autoplot(fc1.PI) + 
  xlab("Year") + 
  ylab("Ventas diferenciadas") + 
  ggtitle("Forecast NNAR")

# Verificación de los intervalos
fc1.PI

# Gráfico del conjunto de testeo junto con la predicción
ts.plot(test,fc1$mean , gpars = list(col = c("black", "red")))

# Se realiza la diferencia inversa para llevar las predicciones a la serie original
pred <- c(fc1$x, fc1$mean)
pred <- cumsum(c(dts0[1], pred))
pred <- diffinv(pred, lag = 12)
pred <- pred + ts[0:12]
tspred <- ts(pred, start= 2016, freq = 12)   # Serie de tiempo de las predicciones
tspred_test <- window(tspred,start = c(2020,7), end = c(2021,1)) # Ventana temporal de la serie de testeo
tspred_test

# Gráfico del conjunto de testeo original con las prediciones
ts.plot(original_test,tspred_test, gpars = list(col = c("black", "red")))

# Gráfico de la serie original con las prediciones
ts.plot(ts,tspred_test, gpars = list(col = c("black", "red")))

# Cálculo del forecast accuracy para testeo
abs <- sum(abs(tspred_test - original_test)) #Diferencia absoluta
real_sales <- sum(original_test) # Ventas reales
(real_sales -abs )/real_sales # 72,88%

########## Modelo ARIMA ########## 

# Inicialización y entrenamiento del modelo
ARIMAfit <- auto.arima(train)
autoplot(forecast(ARIMAfit))

# Predicciones para los próximos 7 meses
pred <- forecast(ARIMAfit, 7)
pred

# Gráfico del conjunto de testeo con las prediciones
ts.plot(test,pred$mean , gpars = list(col = c("black", "red")))

# Se realiza la diferencia inversa para llevar las predicciones a la serie original
pred <- c(pred$x, pred$mean)
pred <- cumsum(c(dts0[1], pred))
pred <- diffinv(pred, lag = 12)
pred <- pred + ts[0:12]
tspred <- ts(pred, start= 2016, freq = 12)   # Serie de tiempo de las predicciones
tspred_test <- window(tspred,start = c(2020,7), end = c(2021,1)) # Ventana temporal de la serie de testeo
tspred_test

# Gráfico del conjunto de testeo original con las prediciones
ts.plot(original_test,tspred_test, gpars = list(col = c("black", "red")))

# Gráfico de la serie original con las prediciones
ts.plot(ts,tspred_test, gpars = list(col = c("black", "red")))

# Cálculo del forecast accuracy para testeo
abs <- sum(abs(tspred_test - original_test)) #Diferencia absoluta
real_sales <- sum(original_test) # Ventas reales
(real_sales -abs )/real_sales # 87,21%

########## Modelo SARIMA ########## 

# Inicialización y entrenamiento del modelo
mod1<-sarima(train, 0,1,0,0,0,0,12)

# Predicciones para los próximos 7 meses
pred <- sarima.for(train, 7,0,0,0,0,1,0,12)
pred

# Gráfico del conjunto de testeo con las prediciones
ts.plot(test,pred$pred , gpars = list(col = c("black", "red")))

# Se realiza la diferencia inversa para llevar las predicciones a la serie original
pred <- c(train, pred$pred)
pred <- cumsum(c(dts0[1], pred))
pred <- diffinv(pred, lag = 12)
pred <- pred + ts[0:12]
tspred <- ts(pred, start= 2016, freq = 12)   # Serie de tiempo de las predicciones
tspred_test <- window(tspred,start = c(2020,7), end = c(2021,1)) # Ventana temporal de la serie de testeo
tspred_test

# Gráfico del conjunto de testeo original con las prediciones
ts.plot(original_test,tspred_test, gpars = list(col = c("black", "red")))

# Gráfico de la serie original con las prediciones
ts.plot(ts,tspred_test, gpars = list(col = c("black", "red")))

# Cálculo del forecast accuracy para testeo
abs <- sum(abs(tspred_test - original_test)) #Diferencia absoluta
real_sales <- sum(original_test) # Ventas reales
(real_sales -abs )/real_sales # 85,17%

########## Modelo GARCH ########## 

# Inicialización y entrenamiento del modelo
ugarch1 = ugarchspec()
ugfit = ugarchfit(spec = ugarch1, data = train)

# Predicciones para los próximos 7 meses
pred <- ugarchforecast(ugfit, n.ahead = 7)
pred <- fitted(pred)
pred <- ts(pred, start= c(2020,7), freq = 12) 

# Gráfico del conjunto de testeo con las prediciones
ts.plot(test,pred, gpars = list(col = c("black", "red")))

# Se realiza la diferencia inversa para llevar las predicciones a la serie original
pred <- c(train, pred)
pred <- cumsum(c(dts0[1], pred))
pred <- diffinv(pred, lag = 12)
pred <- pred + ts[0:12]
tspred <- ts(pred, start= 2016, freq = 12)   # Serie de tiempo de las predicciones
tspred_test <- window(tspred,start = c(2020,7), end = c(2021,1)) # Ventana temporal de la serie de testeo
tspred_test

# Gráfico del conjunto de testeo original con las prediciones
ts.plot(original_test,tspred_test, gpars = list(col = c("black", "red")))

# Gráfico de la serie original con las prediciones
ts.plot(ts,tspred_test, gpars = list(col = c("black", "red")))

# Cálculo del forecast accuracy para testeo
abs <- sum(abs(tspred_test - original_test)) #Diferencia absoluta
real_sales <- sum(original_test) # Ventas reales
(real_sales -abs )/real_sales # 82,19%

########## Redes Neuronales LSTM ########## 

# Se escalan los datos para la red neuronal
serie <- as.data.frame(train)
scaled_train <- (serie$`Sales Units` - mean(serie$`Sales Units`))/sd(serie$`Sales Units`) # Se escala mediante la media y el desvío estándar
scaled_train <- as.matrix(scaled_train) # Se define una matriz

# Se define el horizonte de predicción
prediction <- 7
lag <- prediction

# Se arma una matriz con columnas de valores rezagados
x_train_data <- t(sapply(
  1:(length(scaled_train) - lag - prediction + 1),
  function(x) scaled_train[x:(x + lag - 1), 1]
))
dim(x_train_data)

# Se transforma a un formato array
x_train_arr <- array(
  data = as.numeric(unlist(x_train_data)),
  dim = c(nrow(x_train_data),lag,1)
)
dim(x_train_arr)

# Se calculan los valores de la predicción
y_train_data <- t(sapply(
  (1 + lag):(length(scaled_train) - prediction + 1),
  function(x) scaled_train[x:(x + prediction - 1)]
))
dim(y_train_data)

y_train_arr <- array(
  data = as.numeric(unlist(y_train_data)),
  dim = c(nrow(y_train_data),prediction,1)
)
dim(y_train_arr)

# Se preparan los datos para la predicción
x_test <- as.data.frame(test)
x_test_scaled <- (x_test$`Sales Units` - mean(serie$`Sales Units`))/sd(serie$`Sales Units`)  Se escala mediante la media y el desvío estándar
x_test_scaled <- as.matrix(x_test_scaled) # Se define una matriz

# Se transforma a un formato array
x_pred_arr <- array(
  data = x_test_scaled,
  dim = c(1,lag,1)
)

# Se crea el modelo
lstm_model <- keras_model_sequential()
lstm_model %>%
  layer_lstm(units = 9, # tamaño de la capa
             batch_input_shape = c(1, 7, 1), # tamaño de lote, períodos de tiempo, variables
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 9,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  time_distributed(keras::layer_dense(units = 1))

# Se compila el modelo
lstm_model %>%
  compile(loss = 'mae', 
          optimizer = 'adam', 
          metrics = 'accuracy')

# Análisis de la estructura del modelo
summary(lstm_model)

# Se entrena el modelo
lstm_model %>% fit(
  x = x_train_arr,
  y = y_train_arr,
  batch_size = 1,
  epochs = 20,
  verbose = 0,
  shuffle = TRUE #,
)

# Predicciones del modelo
lstm_forecast <- lstm_model %>%
  predict(x_pred_arr, batch_size = 1) %>%
  .[, , 1]

# Se reescalan las predicciones
lstm_forecast <- lstm_forecast * sd(serie$`Sales Units`)+ mean(serie$`Sales Units`)

# Predicciones con los datos de entrenamiento
fitted <- predict(lstm_model, x_train_arr, batch_size = 1) %>%
  .[, , 1]

# Se transforman los datos para tener una predicción por cada fecha
if(dim(fitted)[2] > 1){
  fit <- c(fitted[, 1], fitted[dim(fitted)[1], 2:dim(fitted)[2]])
} else {
  fit <- fitted[, 1]
}

# Se vuelven a reescalar los datos
fitted <- fit * sd(serie$`Sales Units`) + mean(serie$`Sales Units`)
length(fitted)

# Se especifican los primeros valores como NA
fitted <- c(rep(NA, lag), fitted)

# Configuración de las predicciones como serie temporal
lstm_forecast <- tk_ts(lstm_forecast,
                       start = c(2020, 7),
                       end = c(2021, 1),
                       frequency = 12)

# Visualización las predicciones
lstm_forecast

# Gráfico del conjunto de testeo con las prediciones
ts.plot(test,lstm_forecast, gpars = list(col = c("black", "red")))

# Se realiza la diferencia inversa para llevar las predicciones a la serie original
pred <- c(train, lstm_forecast)
pred <- cumsum(c(dts0[1], pred))
pred <- diffinv(pred, lag = 12)
pred <- pred + ts[0:12]
tspred <- ts(pred, start= 2016, freq = 12)   # Serie de tiempo de las predicciones
tspred_test <- window(tspred,start = c(2020,7), end = c(2021,1)) # Ventana temporal de la serie de testeo
tspred_test

# Gráfico del conjunto de testeo original con las prediciones
ts.plot(original_test,tspred_test, gpars = list(col = c("black", "red")))

# Gráfico de la serie original con las prediciones
ts.plot(ts,tspred_test, gpars = list(col = c("black", "red")))

# Cálculo del forecast accuracy para testeo
abs <- sum(abs(tspred_test - original_test)) #Diferencia absoluta
real_sales <- sum(original_test) # Ventas reales
(real_sales -abs )/real_sales # 85,51%


# Modelo para la totalidad de los datos
lstm_forecast2 <- tk_ts(lstm_forecast,
                       start = c(2016, 1),
                       end = c(2021, 1),
                       frequency = 12)

#---------------------------------------------- 0 ----------------------------------------------
# FUNCION INCORRELACION

# Cargo la siguiente función de incorrelación que realiza un test de Ljung-Box o Box-Pierce para distintos lags

Incorrelation <- function(ts, type = c("Ljung-Box","Box-Pierce"), fitdf = 0){
  p_ljung_box = NULL
  s_ljung_box = NULL
  for(i in 0:(length(ts)/4)){
    p_ljung_box[i] = Box.test(ts,lag = i,type = type,fitdf = fitdf)$p.value
    s_ljung_box[i] = Box.test(ts,lag = i,type = type,fitdf = fitdf)$statistic
  }
  table = data.frame(j = 1:(length(ts)/4),
                     P_Value = p_ljung_box,
                     Statistic = s_ljung_box)
  return(table)
}

#---------------------------------------------- 0 ----------------------------------------------
# FUNCION TEST DE NORMALIDAD
Normality_Test <- function(ts,type = c("JB", "AD", "SW")){
  require(tseries)
  require(nortest)
  if(type == "JB"){
    p_val = jarque.bera.test(ts)$p.value
    stat  = jarque.bera.test(ts)$statistic
  } else if(type == "AD"){
    p_val = ad.test(ts)$p.value
    stat  = ad.test(ts)$statistic
  } else {
    p_val = shapiro.test(ts)$p.value
    stat  = shapiro.test(ts)$statistic
  }
  
  table = data.frame(P_Value = p_val,
                     Statistic = stat)
  return(table)
}

