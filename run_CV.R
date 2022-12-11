source("CVmaster.R")

list.of.packages <- c("tidyverse", "kableExtra",'patchwork', 'GGally', 'corrplot', 
                      'RColorBrewer', 'tidymodels', 'discrim', 'ranger', 'xgboost')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

for (p in list.of.packages){
  library(p, character.only = TRUE)
}


set.seed(666)


dir.create("data",recursive = TRUE,showWarnings = FALSE)
dir.create("pic",recursive = TRUE,showWarnings = FALSE)


img1 = read.table("./data/imagem1.txt")
img2 = read.table("./data/imagem2.txt")
img3 = read.table("./data/imagem3.txt")

col_name = c("Y","X","Label","NDAI","SD","CORR","Rad_Df",
             "Rad_Cf","Rad_Bf","Rad_Af","Rad_An")
colnames(img1) = col_name
colnames(img2) = col_name
colnames(img3) = col_name

img1$image = 1
img2$image = 2
img3$image = 3

full_img = rbind(img1, img2, img3)
full_img_no_unlabeled = full_img %>% filter(Label!=0)

## Create data split using split functions
create_split = function(data, split_func, filter_unlabeled=T){
  idx = split_func(data,3)
  
  for (i in 1:3){
    data[idx[[i]],'block']=i
  }
  
  unique_blocks = data %>% distinct(image, block)
  block_n = unique_blocks %>% dim() %>% .[1]
  val_test_samples = sample(1:block_n, 2)
  
  val_img_block = unique_blocks[val_test_samples[1], ]
  test_img_block = unique_blocks[val_test_samples[2], ]
  
  val = data %>% filter(image==val_img_block$image & block==val_img_block$block)
  test = data %>% filter(image==test_img_block$image & block==test_img_block$block)
  train = data %>% setdiff(val) %>% setdiff(test)
  
  if (filter_unlabeled){
    val = val%>% filter(Label!=0) %>% dplyr::select(-block)
    test = test%>% filter(Label!=0) %>% dplyr::select(-block)
    train = train%>% filter(Label!=0) %>% dplyr::select(-block)
  }
  return(list(val = val,
              test = test,
              train = train))
  
}

## Create split
set.seed(12344)
kmeans_split = create_split(full_img, Kmeans_block_split, filter_unlabeled=T)
bk_split = create_split(full_img, block_split, filter_unlabeled=T)


## ---------------------------- Set up Models -------------------------------------
lr_model = logistic_reg() %>% set_engine('glm')

lda_model = discrim_linear() %>% set_engine('MASS')

qda_model = discrim_quad() %>% set_engine('MASS')

rf_model = rand_forest(mode='classification', mtry=tune(), trees=300, min_n = tune()) %>% 
  set_engine("ranger", num.threads=24)

NB_model = naive_Bayes(smoothness = 1.48, Laplace = 1.08)%>%set_mode('classification')
#NB_model = naive_Bayes(smoothness = tune(), Laplace = tune())%>%set_mode('classification')

xgboost_model = boost_tree(tree_depth = tune(), trees = 300, learn_rate = tune(), min_n = tune(), loss_reduction = tune(), sample_size = tune(), stop_iter = tune())%>%set_engine('xgboost', num.threads=24) %>%set_mode('classification')



## ---------------------------- Run CV on K-means split -------------------------------------
kmean_train_data = rbind(kmeans_split$train, kmeans_split$val) %>% mutate(Label = factor(Label))
lr_res = CVmaster(lr_model,
                  train_features = kmean_train_data,
                  training_labels = kmean_train_data$Label,
                  K = 5,
                  split_func = Kmeans_block_split,
                  metrics = metric_set(accuracy, roc_auc))
write_rds(lr_res, file='data/Kmeans_lr_res.rds')


lda_res = CVmaster(lda_model,
                   train_features = kmean_train_data,
                   training_labels = kmean_train_data$Label,
                   K = 5,
                   split_func = Kmeans_block_split,
                   metrics = metric_set(accuracy, roc_auc))
write_rds(lda_res, file='data/Kmeans_lda_res.rds')

qda_res = CVmaster(qda_model,
                   train_features = kmean_train_data,
                   training_labels = kmean_train_data$Label,
                   K = 5,
                   split_func = Kmeans_block_split,
                   metrics = metric_set(accuracy, roc_auc))
write_rds(qda_res, file='data/Kmeans_qda_res.rds')

rf_res = CVmaster(rf_model,
                  train_features = kmean_train_data,
                  training_labels = kmean_train_data$Label,
                  K = 5,
                  split_func = Kmeans_block_split,
                  metrics = metric_set(accuracy, roc_auc), .tune_grid = 30)
write_rds(rf_res, file='data/Kmeans_rf_res.rds')

nb_res = CVmaster(NB_model,
                  train_features = kmean_train_data,
                  training_labels = kmean_train_data$Label,
                  K = 5,
                  split_func = Kmeans_block_split,
                  metrics = metric_set(accuracy, roc_auc))
write_rds(nb_res, file='data/Kmeans_nb_res.rds')

xgboost_res = CVmaster(xgboost_model,
                       train_features = kmean_train_data,
                       training_labels = kmean_train_data$Label,
                       K = 5,
                       split_func = Kmeans_block_split,
                       metrics = metric_set(accuracy, roc_auc), .tune_grid = 30)
write_rds(xgboost_res, file='data/Kmeans_xgboost_res.rds')

## ---------------------------- Run CV on Block split -------------------------------------

bk_train_data = rbind(bk_split$train, bk_split$val) %>% mutate(Label = factor(Label))

lr_res = CVmaster(lr_model,
                  train_features = bk_train_data,
                  training_labels = bk_train_data$Label,
                  K = 5,
                  split_func = block_split,
                  metrics = metric_set(accuracy, roc_auc))
write_rds(lr_res, file='data/Block_lr_res.rds')

lda_res = CVmaster(lda_model,
                   train_features = bk_train_data,
                   training_labels = bk_train_data$Label,
                   K = 5,
                   split_func = block_split,
                   metrics = metric_set(accuracy, roc_auc))
write_rds(lda_res, file='data/Block_lda_res.rds')

qda_res = CVmaster(qda_model,
                   train_features = bk_train_data,
                   training_labels = bk_train_data$Label,
                   K = 5,
                   split_func = block_split,
                   metrics = metric_set(accuracy, roc_auc))
write_rds(qda_res, file='data/Block_qda_res.rds')

rf_res = CVmaster(rf_model,
                  train_features = bk_train_data,
                  training_labels = bk_train_data$Label,
                  K = 5,
                  split_func = block_split,
                  metrics = metric_set(accuracy, roc_auc), .tune_grid = 30)
write_rds(rf_res, file='data/Block_rf_res.rds')

nb_res = CVmaster(NB_model,
                  train_features = bk_train_data,
                  training_labels = bk_train_data$Label,
                  K = 5,
                  split_func = block_split,
                  metrics = metric_set(accuracy, roc_auc), .tune_grid = 30)
write_rds(nb_res, file='data/Block_nb_res.rds')

xgboost_res = CVmaster(xgboost_model,
                       train_features = bk_train_data,
                       training_labels = bk_train_data$Label,
                       K = 5,
                       split_func = block_split,
                       metrics = metric_set(accuracy, roc_auc), .tune_grid = 30)
write_rds(xgboost_res, file='data/Block_xgboost_res.rds')

