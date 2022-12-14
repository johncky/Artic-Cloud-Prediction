list.of.packages <- c("tidyverse", "kableExtra",'patchwork', 'GGally', 'corrplot', 
                      'RColorBrewer', 'tidymodels', 'discrim', 'ranger')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

for (p in list.of.packages){
  library(p, character.only = TRUE)
}



set.seed(666)


dir.create("data",recursive = TRUE,showWarnings = FALSE)
dir.create("pic",recursive = TRUE,showWarnings = FALSE)


# --------------------------------- Load data ----------------------------------
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

set.seed(12344)

kmeans_split = create_split(full_img, Kmeans_block_split, filter_unlabeled=T)
bk_split = create_split(full_img, block_split, filter_unlabeled=T)

# Extract best Xgboost & Random Forest Parameters
## ------ K-means Split  ------
kmeans_best_xgboost = read_rds('./data/Kmeans_xgboost_res.rds')$fold_loss
kmeans_best_xgboost_param = list(min_n = kmeans_best_xgboost$min_n[1],
                                 tree_depth = kmeans_best_xgboost$tree_depth[1],
                                 learn_rate = kmeans_best_xgboost$learn_rate[1],
                                 loss_reduction = kmeans_best_xgboost$loss_reduction[1],
                                 sample_size = kmeans_best_xgboost$sample_size[1],
                                 stop_iter = kmeans_best_xgboost$stop_iter[1])

kmeans_best_rf = read_rds('./data/Kmeans_rf_res.rds')$fold_loss
kmeans_best_rf_param = list(mtry = kmeans_best_rf$mtry[1],
                            min_n = kmeans_best_rf$min_n[1])


## ------ Block Split ------
block_best_xgboost = read_rds('./data/block_xgboost_res.rds')$fold_loss
block_best_xgboost_param = list(min_n = block_best_xgboost$min_n[1],
                                tree_depth = block_best_xgboost$tree_depth[1],
                                learn_rate = block_best_xgboost$learn_rate[1],
                                loss_reduction = block_best_xgboost$loss_reduction[1],
                                sample_size = block_best_xgboost$sample_size[1],
                                stop_iter = block_best_xgboost$stop_iter[1])


block_best_rf = read_rds('./data/block_rf_res.rds')$fold_loss
block_best_rf_param = list(mtry = block_best_rf$mtry[1],
                           min_n = block_best_rf$min_n[1])


# --------- Models using best params from CV result above ---------
lr_model = logistic_reg() %>% set_engine('glm')
lr_reg_model = logistic_reg(mixture =tune(), penalty=tune()) %>% set_engine('glmnet')
lda_model = discrim_linear() %>% set_engine('MASS')
qda_model = discrim_quad() %>% set_engine('MASS')

rf_model = rand_forest(mode='classification', mtry=kmeans_best_rf_param$mtry, trees=300, min_n = kmeans_best_rf_param$min_n) %>%
  set_engine("ranger", num.threads=12)

NB_model = naive_Bayes(smoothness = 1.48, Laplace = 1.08)%>%set_mode('classification')

xgboost_model = boost_tree(tree_depth = kmeans_best_xgboost_param$tree_depth	, trees = 300,
                           learn_rate = kmeans_best_xgboost_param$learn_rate,
                           min_n = kmeans_best_xgboost_param$min_n,
                           loss_reduction= kmeans_best_xgboost_param$loss_reduction,
                           sample_size = kmeans_best_xgboost_param$sample_size,
                           stop_iter = kmeans_best_xgboost_param$stop_iter)%>%set_engine('xgboost') %>%set_mode('classification')

# --------- K means Split ----------
train_data = rbind(kmeans_split$train, kmeans_split$val) %>% mutate(Label = factor(Label))
test_data = kmeans_split$test %>% mutate(Label = factor(Label))
feature_names = train_data %>% dplyr::select(-c(X,Y,Label, image)) %>% colnames()
formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 

kmean_lr_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data) %>% step_normalize(all_numeric_predictors())) %>%
  add_model(lr_model)%>%
  fit(data = train_data) %>%
  augment(new_data = test_data)
kmean_lr_roc_test =  kmean_lr_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="Logistic - test")


kmean_lda_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data) %>% step_normalize(all_numeric_predictors())) %>% add_model(lda_model)%>%
  fit(data = train_data) %>% augment(new_data = test_data)
kmean_lda_roc_test = kmean_lda_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="LDA - test")


kmean_qda_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data) %>% step_normalize(all_numeric_predictors())) %>% add_model(qda_model)%>%
  fit(data = train_data) %>% augment(new_data =test_data)
kmean_qda_roc_test = kmean_qda_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="QDA - test")


kmean_rf_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data) %>% step_normalize(all_numeric_predictors())) %>% add_model(rf_model)%>%
  fit(data = train_data) %>% augment(new_data =test_data)
kmean_rf_roc_test = kmean_rf_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="Random Forest - test")


kmean_nb_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data) %>% step_normalize(all_numeric_predictors())) %>% add_model(NB_model)%>%
  fit(data = train_data) %>% augment(new_data = test_data)
kmean_nb_roc_test = kmean_nb_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="Naive Bayes - test")


kmean_xgboost_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data) %>% step_normalize(all_numeric_predictors())) %>% add_model(xgboost_model)%>%
  fit(data = train_data) %>% augment(new_data = test_data)
kmean_xgboost_roc_test = kmean_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="XGboost - test")

p = rbind(kmean_lr_roc_test,
          kmean_lda_roc_test,
          kmean_qda_roc_test,
          kmean_rf_roc_test,
          kmean_nb_roc_test,
          kmean_xgboost_roc_test)%>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color=name))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() + labs(
    title='K-means Split: Test ROC'
  )
ggsave('./pic/kmeans_test_roc.jpg', p)

max_pt = rbind(kmean_lr_roc_test,
               kmean_lda_roc_test,
               kmean_qda_roc_test,
               kmean_rf_roc_test,
               kmean_nb_roc_test,
               kmean_xgboost_roc_test) %>% group_by(name) %>% mutate(angle=specificity^2 + sensitivity^2) %>% filter(angle == max(angle))


pt_data = rbind(kmean_lr_roc_test,
                kmean_lda_roc_test,
                kmean_qda_roc_test,
                kmean_rf_roc_test,
                kmean_nb_roc_test,
                kmean_xgboost_roc_test)
p1 = ggplot()+
  geom_line(aes(x = 1 - pt_data$specificity, y = pt_data$sensitivity, color=pt_data$name))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  geom_point(aes(x=1 - max_pt$specificity, y=max_pt$sensitivity), shape='x', color='red', size=4)+
  coord_equal() + labs(
    # title='K-means split',
    x = '1 - specificity', y='sensitivity') +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",  text = element_text(size = 20))

### ------------------------------ Test Performance using K-means split ----------------------------------------
accu = rbind(
  kmean_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='XGBoost'),
  kmean_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='LDA'),
  kmean_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='Logistic'),
  kmean_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='Naive Bayes'),
  kmean_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='QDA'),
  kmean_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('Accuracy' = .estimate, 'Model' = model)%>%
  select('Model', 'Accuracy') 

auc = rbind(
  kmean_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')   %>% mutate(model='XGBoost'),
  kmean_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')   %>% mutate(model='LDA'),
  kmean_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')   %>% mutate(model='Logistic'),
  kmean_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')   %>% mutate(model='Naive Bayes'),
  kmean_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')   %>% mutate(model='QDA'),
  kmean_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('AUC' = .estimate, 'Model' = model)%>%
  select('Model', 'AUC') 

f_meas =  rbind(
  kmean_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='XGBoost'),
  kmean_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='LDA'),
  kmean_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='Logistic'),
  kmean_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='Naive Bayes'),
  kmean_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='QDA'),
  kmean_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('F measure' = .estimate, 'Model' = model)%>%
  select('Model', 'F measure') 

kap = rbind(
  kmean_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='XGBoost'),
  kmean_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='LDA'),
  kmean_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='Logistic'),
  kmean_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='Naive Bayes'),
  kmean_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='QDA'),
  kmean_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('Kappa' = .estimate, 'Model' = model)%>%
  select('Model', 'Kappa') 

accu %>% left_join(auc) %>% left_join(f_meas) %>% left_join(kap) %>% mutate_if(is.numeric, ~round(., 3)) %>%
  kbl(
    # caption = "<center><strong>K-means Split: Test Performance<center><strong>",
    escape = FALSE,
    format = 'html',) %>%
  kable_classic(full_width = F, html_font = "Cambria")  %>% save_kable(file='pic/kmeans_test_err.pdf')

# ------ Model Params ------
rf_model = rand_forest(mode='classification', mtry=block_best_rf_param$mtry, trees=300, min_n = block_best_rf_param$min_n) %>%
  set_engine("ranger", num.threads=12)

xgboost_model = boost_tree(tree_depth = block_best_xgboost_param$tree_depth	, trees = 300,
                           learn_rate = block_best_xgboost_param$learn_rate,
                           min_n = block_best_xgboost_param$min_n,
                           loss_reduction= block_best_xgboost_param$loss_reduction,
                           sample_size = block_best_xgboost_param$sample_size,
                           stop_iter = block_best_xgboost_param$stop_iter)%>%set_engine('xgboost') %>%set_mode('classification')


# --------- Block Split ----------
train_data_bk = rbind(bk_split$train, bk_split$val) %>% mutate(Label = factor(Label))
test_data_bk = bk_split$test %>% mutate(Label = factor(Label))
feature_names = train_data_bk %>% dplyr::select(-c(X,Y,Label, image)) %>% colnames()
formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 

bk_lr_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>%
  add_model(lr_model)%>%
  fit(data = train_data_bk) %>%
  augment(new_data = test_data_bk)
bk_lr_roc_test =  bk_lr_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="Logistic")


bk_lda_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(lda_model)%>%
  fit(data = train_data_bk) %>% augment(new_data = test_data_bk)
bk_lda_roc_test = bk_lda_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="LDA")


bk_qda_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(qda_model)%>%
  fit(data = train_data_bk) %>% augment(new_data =test_data_bk)
bk_qda_roc_test = bk_qda_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="QDA")


bk_rf_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(rf_model)%>%
  fit(data = train_data_bk) %>% augment(new_data =test_data_bk)
bk_rf_roc_test = bk_rf_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="Random Forest")


bk_nb_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(NB_model)%>%
  fit(data = train_data_bk) %>% augment(new_data = test_data_bk)
bk_nb_roc_test = bk_nb_roc_perf %>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="Naive Bayes")


bk_xgboost_roc_perf = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(xgboost_model)%>%
  fit(data = train_data_bk) %>% augment(new_data = test_data_bk)
bk_xgboost_roc_test = bk_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>%
  yardstick::roc_curve(factor(Label),'.pred_-1') %>%
  mutate(name="XGboost")

max_pt2 = rbind(bk_lr_roc_test,
                bk_lda_roc_test,
                bk_qda_roc_test,
                bk_rf_roc_test,
                bk_nb_roc_test,
                bk_xgboost_roc_test) %>% group_by(name) %>% mutate(angle=specificity^2 + sensitivity^2) %>% filter(angle == max(angle))


pt_data2 = rbind(bk_lr_roc_test,
                 bk_lda_roc_test,
                 bk_qda_roc_test,
                 bk_rf_roc_test,
                 bk_nb_roc_test,
                 bk_xgboost_roc_test)

p2 = ggplot()+
  geom_line(aes(x = 1 - pt_data2$specificity, y = pt_data2$sensitivity, color=pt_data2$name))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  geom_point(aes(x=1 - max_pt2$specificity, y=max_pt2$sensitivity), shape='x', color='red', size=4)+
  coord_equal() + labs(
    # title='Block Split',
    x = '1 - specificity', y='sensitivity') +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 20))+
  guides(color=guide_legend(title="Model"))

## -------------------------------------------- Test ROC Curves ------------------------------
p = p1 + p2 + plot_layout(guides = 'collect') + plot_annotation(
  # title='Test Set ROC',
  theme=theme(plot.title = element_text(hjust = 0.4),
              text = element_text(size = 20)))

ggsave('./pic/test_roc.jpg', p,width = 20, height = 20)

### ------------------------------ Test Performance using Block split ----------------------------------------
accu = rbind(
  bk_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='XGBoost'),
  bk_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='LDA'),
  bk_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='Logistic'),
  bk_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='Naive Bayes'),
  bk_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='QDA'),
  bk_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::accuracy(Label,'.pred_class')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('Accuracy' = .estimate, 'Model' = model)%>%
  select('Model', 'Accuracy') 

auc = rbind(
  bk_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='XGBoost'),
  bk_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='LDA'),
  bk_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='Logistic'),
  bk_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='Naive Bayes'),
  bk_qda_roc_perf%>%select(Label, starts_with(".pred")) %>%yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='QDA'),
  bk_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::roc_auc(Label,'.pred_-1')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('AUC' = .estimate, 'Model' = model)%>%
  select('Model', 'AUC') 

f_meas =  rbind(
  bk_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='XGBoost'),
  bk_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='LDA'),
  bk_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='Logistic'),
  bk_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='Naive Bayes'),
  bk_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='QDA'),
  bk_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::f_meas(Label,'.pred_class')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('F measure' = .estimate, 'Model' = model)%>%
  select('Model', 'F measure') 

kap = rbind(
  bk_xgboost_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='XGBoost'),
  bk_lda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='LDA'),
  bk_lr_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='Logistic'),
  bk_nb_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='Naive Bayes'),
  bk_qda_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='QDA'),
  bk_rf_roc_perf%>%select(Label, starts_with(".pred")) %>% yardstick::kap(Label,'.pred_class')  %>% mutate(model='Random Forest'))%>%
  arrange(desc(.estimate))%>%mutate('Kappa' = .estimate, 'Model' = model)%>%
  select('Model', 'Kappa') 

accu %>% left_join(auc) %>% left_join(f_meas) %>% left_join(kap) %>% mutate_if(is.numeric, ~round(., 3)) %>% 
  kbl(
    # caption = "<center><strong>Block Split: Test Performance<center><strong>",
    escape = FALSE,
    format = 'html',) %>%
  kable_classic(full_width = F, html_font = "Cambria") %>% save_kable('./pic/block_test_table.pdf')