



CVmaster = function(tidymodel, train_features, training_labels, K, metrics, split_func,
                    .scale=T, .tune_grid=NULL, excld_cols = c('X', 'Y', 'image', 'Label')){
  
  # --------------- Create Formula ---------------
  excld_cols = train_features %>% colnames() %>% intersect(excld_cols)
  feature_names = train_features %>% dplyr::select(-excld_cols) %>% colnames()
  formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 
  
  # ensure label is factor
  training_labels = training_labels %>% as.factor()
  
  # full train data with labels
  full_train = train_features %>% dplyr::select(feature_names) %>% mutate(Label=training_labels)
  
  # --------------- Create Recipe ---------------
  model_recipe = recipe(formula, data = full_train)
  if (.scale){
    model_recipe = model_recipe %>% step_normalize(all_numeric_predictors())
  }
  
  # --------------- Create Resampler ---------------
  cv_splits = split_func(train_features, K=K)
  cv_settings = list(v = K, strata = FALSE, repeats = 1)
  resample_split = list()
  fold_id = paste('Fold ', 1:K)
  N = dim(train_features)[1] 
  
  for (k in 1:K){
    val_idx = cv_splits[[k]]
    train_idx = seq(1, N)[-val_idx]
    resample_split[[k]] = make_splits(list(analysis=train_idx, assessment=val_idx), data=full_train)
  }
  resample_settings = new_rset(splits = resample_split,
                               ids = fold_id,
                               subclass = c("vfold_cv", "rset"),
                               attrib = cv_settings)
  
  # --------------- Create Workflow & Fit CV---------------
  if (!is.null(.tune_grid)){
    res = workflow() %>%
      add_recipe(model_recipe) %>%
      add_model(tidymodel) %>%
      tune_grid(
        resample_settings,
        grid=.tune_grid,
        control=control_grid(verbose=T),
        metrics=metrics
      )
  }else{
    res = workflow() %>%
      add_recipe(model_recipe) %>%
      add_model(tidymodel) %>%
      fit_resamples(resamples = resample_settings, metrics = metrics)
  }
  
  # --------------- Extract CV Loss---------------
  fold_loss = data.frame()
  for (i in 1:K){
    fold_loss = rbind(fold_loss, res$.metrics[[i]])
  }
  # If there is tuning parameters, select the best one. If there are multiple metrics, the first one is used to select.
  if (!is.null(.tune_grid)){
    fold_loss = fold_loss %>% dplyr::filter(.config %in% (res%>%select_best(metric = "roc_auc")%>%pull(.config)))
  }
  
  N_loss =  fold_loss %>% distinct(.metric)%>%pull()%>%length()
  fold_loss = fold_loss %>% dplyr::select(-c(.config, .estimator)) %>%
    mutate(fold = rep(1:K, each=N_loss)) %>%
    pivot_wider(names_from = .metric, values_from = .estimate)
  
  output = list(fold_loss = fold_loss, 
                avg = (fold_loss %>%  dplyr::select(-fold) %>% colMeans()),
                workflow=res)
}


block_split = function(data, K){
  N =data %>% dim() %>% .[1]
  fold_n = round(N/ K)
  remainder = N - (K-1) * fold_n
  block_id = rep(1:(K-1), each= fold_n)
  block_id = c(block_id, rep(K, remainder))
  data['idx'] = 1:(dim(data)[1])
  data = data %>% arrange(X, Y)
  data['block']= block_id
  
  idx = vector(mode = 'list',length = K)
  for (i in 1:K){
    idx[[i]] = data %>% filter(block==i) %>% pull(idx) 
  }
  return(idx)
}

Kmeans_block_split = function(data, K){
  idx = vector(mode = 'list',length = K)
  res = kmeans(data %>%  dplyr::select(Y,X) ,K)
  for (i in 1:K){
    idx[[i]] = which(res$cluster == i)
  }
  return (idx)
}




