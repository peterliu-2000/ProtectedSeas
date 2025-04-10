library(readr) # basic R functions
library(plyr) # basic R functions
library(dplyr) # basic R functions
library(raster) # helpful GIS functions
library(rgeos) # helpful GIS functions
library(rgdal) # helpful GIS functions
library(sf) # helpful GIS functions
library(sp) # helpful GIS functions
library(xgboost)
library(stringr)
library(caret)
library(car)
library(data.table)
library(Matrix)
library(mltools)
library(mlr)
library(parallel)
library(parallelMap)
library(reshape2)

drive = "E"
folder = "ResearchProjects\\SpeedAccuracy"

if (drive == "C") {
  path = "C:\\Users\\saman\\Anthropocene Dropbox\\Samantha Cope\\"
} else {
  path = "E:\\Dropbox (Anthropocene)\\"
}
working_directory = paste0(path, "AnthInst_Sam\\M2\\", folder)
setwd(working_directory)

sites <- c(26, 19, 37, 43, 29, 10, 31, 28, 22, 42)

# import AIS tracks with associations
datalist = list()

for (i in sites) {
  site = as.character(i)
  df = as.data.frame(read.csv(paste(site, "tracks_ais.csv",
                                       sep = "_")))
  datalist[[i]] = df
}
df_all_ais = do.call(rbind, datalist)

datalist = list()

for (i in sites) {
  site = as.character(i)
  df = as.data.frame(read.csv(paste(site, "tracks_radar.csv",
                                    sep = "_")))
  datalist[[i]] = df
}
df_all_radar = do.call(rbind, datalist)

# AIS data, only Class A with all 4 dimensions

df_length <- df_all_ais %>% filter(!is.na(dim_a)) %>%
  filter(!is.na(dim_b)) %>%
  filter(!is.na(dim_c)) %>%
  filter(!is.na(dim_d))

df_class_a <- subset(df_length, dim_a > 0 &
                       dim_b > 0 &
                       dim_c > 0 &
                       dim_d > 0)
df_class_a$loa <- df_class_a$dim_a + df_class_a$dim_b
df_class_a$sa <- (df_class_a$dim_c + df_class_a$dim_d) * df_class_a$loa

df = df_class_a %>%
  dplyr::select(id_track,
                id_site,
                loa,
                min_speed,
                max_speed,
                curviness,
                turning_mean,
                turning_std,
                heading_std,
                avg_speed,
                speed_diff_mean,
                speed_diff_std,
                dist_diff_mean,
                dist_diff_std)

df$min_speed = as.numeric(as.character(df$min_speed))
df$max_speed = as.numeric(as.character(df$max_speed))
df$curviness = as.numeric(as.character(df$curviness))
df$turning_mean = as.numeric(as.character(df$turning_me))
df$avg_speed = as.numeric(as.character(df$avg_speed))
df$turning_std = as.numeric(as.character(df$turning_st))
df$heading_std = as.numeric(as.character(df$heading_st))
df$speed_diff_mean = as.numeric(as.character(df$speed_diff_mean))
df$speed_diff_std = as.numeric(as.character(df$speed_diff_std))
df$dist_diff_mean = as.numeric(as.character(df$dist_diff_mean))
df$dist_diff_std = as.numeric(as.character(df$dist_diff_std))
df = na.omit(df)

id <- which(c("id_track", "id_site") %in% colnames(df))
id_valid <- which(c("id_track", "id_site", "loa") %in% colnames(df))

# evaluate variable variance
nzv <- nearZeroVar(df[,-id_valid], saveMetrics= TRUE)
# variables with TRUE nzv are a near zero variance predictor, remove

# check correlations
# Spearman correlation
#According to Tabachnick & Fidell (1996) the independent variables with a bivariate correlation more than 0.70 should not be included in multiple regression analysis.
cormat = round(cor(df[,-id_valid], use = "complete.obs", method = "spearman"), 2)

get_lower_tri<-function(cormat){
  cormat[lower.tri(cormat)] <- NA
  return(cormat)
}
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
cormat <- reorder_cormat(cormat)
lower_tri <- get_lower_tri(cormat)
melted_cormat <- melt(lower_tri, na.rm = TRUE)
ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab") +
  theme_bw()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        panel.border = element_blank(),
        panel.grid = element_blank(),
        legend.position = "none")+
  coord_fixed()

df_corr <- subset(melted_cormat, value >= 0.7 &
                    Var1 != Var2)

# NOTE CORRELATED FEATURES

model_eval = function(data) {
  df_train = sample_n(data, round(0.75*nrow(data), 0))
  df_test = dplyr::setdiff(data, df_train)
  train <- data.table(df_train[,-id], keep.rownames = F)
  test = data.table(df_test[,-id], keep.rownames = F)
  
  #running internal CV to tune hyperparameters
  traintask <- makeRegrTask(data = df_train[,-id],target = "loa")
  testtask <- makeRegrTask(data = df_test[,-id],target = "loa")
  
  lrn <- makeLearner("regr.xgboost", predict.type = "response")
  lrn$par.vals <- list(objective="reg:squarederror", eval_metric="rmse")
  
  params <- makeParamSet(makeDiscreteParam("booster", values = c("gbtree")),
                         makeIntegerParam("nrounds",lower = 50L,upper = 300L),
                         makeIntegerParam("max_depth",lower = 3L,upper = 10L),
                         makeIntegerParam("gamma",lower = 0L,upper = 10L),
                         makeNumericParam("eta",lower = 0.1,upper = 0.5),
                         makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                         makeNumericParam("subsample",lower = 0.5,upper = 1),
                         makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
  rdesc <- makeResampleDesc("CV",stratify = F,iters=5L)
  ctrl <- makeTuneControlRandom(maxit = 10L)
  parallelStartSocket(cpus = detectCores())
  mytune <- tuneParams(learner = lrn,
                       task = traintask,
                       resampling = rdesc,
                       measures = mse,
                       par.set = params,
                       control = ctrl,
                       show.info = T)
  lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)
  
  # print ideal parameters
  print(mytune$x)
  params <- as.data.frame(mytune$x)
  
  # train model and evaluate accuracy on test data
  xgtrain <- xgb.DMatrix(data = as.matrix(train[,!"loa"]), label = train$loa)
  xgtest <- xgb.DMatrix(data = as.matrix(test[,!"loa"]), label = test$loa)
  bst <- xgboost(data = xgtrain,
                 objective = "reg:squarederror",
                 booster = "gbtree",
                 nrounds = params$nrounds,
                 max_depth = params$max_depth,
                 gamma = params$gamma,
                 eta = params$eta,
                 min_child_weight = params$min_child_weight,
                 subsample = params$subsample,
                 colsample_bytree = params$colsample_bytree)
  pred <- as.data.frame(predict(bst, xgtest))
  names(pred) = c("loa_pred")
  result = cbind(df_test, pred)
  return(list(df1=result, df2=bst))
}

# inital model train and test
results = model_eval(df)
im <- xgb.importance(model = results$df2)

# of the correlated feature pairs, remove the feature with a lower gain in importance matrix
# identify features to remove
feat_remove = c("speed_diff_std", "curviness", "turning_std", "avg_speed", "dist_diff_mean")

df = df_class_a %>%
  dplyr::select(id_track,
                id_site,
                loa,
                min_speed,
                max_speed,
                curviness,
                turning_mean,
                turning_std,
                heading_std,
                avg_speed,
                speed_diff_mean,
                speed_diff_std,
                dist_diff_mean,
                dist_diff_std)

df$min_speed = as.numeric(as.character(df$min_speed))
df$max_speed = as.numeric(as.character(df$max_speed))
df$curviness = as.numeric(as.character(df$curviness))
df$turning_mean = as.numeric(as.character(df$turning_me))
df$avg_speed = as.numeric(as.character(df$avg_speed))
df$turning_std = as.numeric(as.character(df$turning_st))
df$heading_std = as.numeric(as.character(df$heading_st))
df$speed_diff_mean = as.numeric(as.character(df$speed_diff_mean))
df$speed_diff_std = as.numeric(as.character(df$speed_diff_std))
df$dist_diff_mean = as.numeric(as.character(df$dist_diff_mean))
df$dist_diff_std = as.numeric(as.character(df$dist_diff_std))
df = na.omit(df)

df = df %>%
  dplyr::select(-feat_remove)

id <- which(c("id_track", "id_site") %in% colnames(df))
id_valid <- which(c("id_track", "id_site", "loa") %in% colnames(df))

results = model_eval(df)
result = results$df1
im <- xgb.importance(model = results$df2)
im$order <- 1:nrow(im) 

r = subset(im, order <= 1)
feat = c(r$Feature)
df_new = df %>%
  dplyr::select('id_track', 'id_site', 'loa', feat)
test = model_eval(df_new)$df1
rmse <- RMSE(test$loa_pred, test$loa)

datalist2 = list()
for (i in unique(im$order)) {
  r = subset(im, order <= i)
  feat = c(r$Feature)
  df_new = df %>%
    dplyr::select('id_track', 'id_site', 'loa', feat)
  test = model_eval(df_new)$df1
  rmse <- RMSE(test$loa_pred, test$loa)
  datalist2[[i]] = rmse
}
accur_results = do.call(rbind, datalist2)
accur_results = cbind(as.data.frame(im$Feature), accur_results)
names(accur_results) = c("feature", "RMSE")
accur_results$order <- 1:nrow(accur_results) 
accur_results$feature = reorder(accur_results$feature, accur_results$order)

ggplot(accur_results, aes(x = feature, y = RMSE, group = 1)) +
  geom_point() +
  geom_line() +
  theme_bw()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none")

test$diff <- abs(test$loa - test$loa_pred)

plot(test$loa, test$loa_pred)

# results, 28,172 records
RMSE(test$loa, test$loa_pred) # 52m
quantile(test$diff, probs = 0.9) # 88m

