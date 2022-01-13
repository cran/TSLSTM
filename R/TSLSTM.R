

#' @title Long Short Term Memory (LSTM) Model for Time Series Forecasting
#' @description The LSTM (Long Short-Term Memory) model is a Recurrent Neural Network (RNN) based architecture that is widely used for time series forecasting. Min-Max transformation has been used for data preparation. Here, we have used one LSTM layer as a simple LSTM model and a Dense layer is used as the output layer. Then, compile the model using the loss function, optimizer and metrics. This package is based on Keras and TensorFlow modules.
#' @param ts Time series data
#' @param xreg Exogenous variables
#' @param tsLag Lag of time series data
#' @param xregLag Lag of exogenous variables
#' @param LSTMUnits Number of unit in LSTM layer
#' @param DropoutRate Dropout rate
#' @param Epochs Number of epochs
#' @param CompLoss Loss function
#' @param CompMetrics Metrics
#' @param ActivationFn Activation function
#' @param SplitRatio Training and testing data split ratio
#' @param ValidationSplit Validation split ration
#'
#' @import keras tensorflow tsutils stats
#' @return
#' \itemize{
#'   \item TrainFittedValue: Fitted value of train data
#'   \item TestPredictedValue: Predicted value of test data
#'   \item AccuracyTable: RMSE and MAPE of train and test data
#' }
#' @export
#'
#' @examples
#' \donttest{
#'y<-rnorm(100,mean=100,sd=50)
#'x1<-rnorm(100,mean=50,sd=50)
#'x2<-rnorm(100, mean=50, sd=25)
#'x<-cbind(x1,x2)
#'TSLSTM<-ts.lstm(ts=y,xreg = x,tsLag=2,xregLag = 0,LSTMUnits=5, Epochs=2)
#'}
#' @references
#' Paul, R.K. and Garai, S. (2021). Performance comparison of wavelets-based machine learning technique for forecasting agricultural commodity prices, Soft Computing, 25(20), 12857-12873

ts.lstm<-function(ts,xreg=NULL,tsLag,xregLag=0,LSTMUnits, DropoutRate=0.00, Epochs=10, CompLoss="mse",CompMetrics="mae",
                  ActivationFn='tanh',SplitRatio=0.8, ValidationSplit=0.1)
{

### Lag selection################################

  ## data matrix preparation

  feature_mat<-NULL
  if (is.null(xreg)){
    lag_y<-lagmatrix(as.ts(ts),lag = c(0:(tsLag)))
    all_feature<-cbind(lag_y,feature_mat)
  } else {
    exo<-xreg
    exo_v<-dim((exo))[2]
    for (var in 1:exo_v) {
      lag_x<-lagmatrix(as.ts(exo[,var]),lag=c(0:xregLag))
      feature_mat<-cbind(feature_mat,lag_x)
    }
    lag_y<-lagmatrix(as.ts(ts),lag = c(0:(tsLag)))
    all_feature<-cbind(lag_y,feature_mat)
  }
  if(xregLag>=tsLag){
    data_all<-all_feature[-c(1:xregLag),]
  } else {
    data_all<-all_feature[-c(1:tsLag),]
  }
  data<-data_all[,-1]
  feature<-ncol(data)

  a<- 1/(max(data[,1])-min(data[,1]))
  b<- min(data[,1])/(max(data[,1])-min(data[,1]))

  normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }

  denormalize <- function(x) {
    return ((x + b) / a)
  }
  data_normalise <- apply(data,2, normalize)

  ##################### Data Split #################################

  train_normalise <- data_normalise[c(1:(nrow(data_normalise)*SplitRatio)),]
  test_normalise<- data_normalise[-c(1:(nrow(data_normalise)*SplitRatio)),]

  n_train<-nrow(train_normalise)
  n_test<-nrow(test_normalise)

  #Data Array Preparation
  train_data<-timeseries_generator(data = train_normalise,targets = train_normalise[,1],length = 1,sampling_rate = 1, batch_size = 1)
  test_data<-timeseries_generator(data = test_normalise,targets = test_normalise[,1],length = 1,sampling_rate = 1, batch_size = 1)

  # LSTM model
  lstm_model <- keras_model_sequential() %>%
    layer_lstm(units =LSTMUnits, input_shape = c(1,feature),activation=ActivationFn,dropout=DropoutRate,return_sequences = TRUE) %>%
    layer_dense(units = 1)

  lstm_model %>% compile(optimizer = optimizer_rmsprop(), loss=CompLoss, metrics=CompMetrics)

  summary(lstm_model)

  lstm_history <- lstm_model %>% fit(
    train_data,
    batch_size = 1,
    epochs = Epochs, validation.split=ValidationSplit
  ) #

  # LSTM fitted value
  lstm_model %>% evaluate(train_data)
  lstm_fiited_norm <- lstm_model %>%
    predict(train_data)
  train_lstm_fiited<-denormalize(lstm_fiited_norm)


  # LSTM  Prediction
  lstm_model %>% evaluate(test_data)
  lstm_predicted_norm <- lstm_model %>%
    predict(test_data)
  test_lstm_predicted<-denormalize(lstm_predicted_norm)

  ### accuracy measurements ##################

  actual_data<-data_all[,2]
  train_actual<-actual_data[c((1+1):n_train)]
  test_actual<-actual_data[c((1+n_train+1):(n_train+n_test))]

  AccuracyTable<-matrix(nrow=2, ncol=2)
  AccuracyTable[1,1]<-round(sqrt(mean((train_actual-train_lstm_fiited)^2)), digits = 4)
  AccuracyTable[1,2]<-round(mean(abs((train_actual-train_lstm_fiited)/train_actual)), digits = 4)
  AccuracyTable[2,1]<-round(sqrt(mean((test_actual-test_lstm_predicted)^2)), digits = 4)
  AccuracyTable[2,2]<-round(mean(abs((test_actual-test_lstm_predicted)/test_actual)), digits = 4)
  row.names(AccuracyTable)<-c("Train", "Test")
  colnames(AccuracyTable)<-c("RMSE", "MAPE")
  return(list(TrainFittedValue=train_lstm_fiited,TestPredictedValue=test_lstm_predicted, AccuracyTable=AccuracyTable))
}
