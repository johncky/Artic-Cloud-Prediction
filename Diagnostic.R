
source("CVmaster.R")

set.seed(666)

list.of.packages <- c("tidyverse", "kableExtra",'patchwork', 'vip', 'plotly', 'tidymodels', 'ranger')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

for (p in list.of.packages){
  library(p, character.only = TRUE)
}



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


# Load Best Parameters
block_best_rf_workflow = read_rds('./data/block_rf_res.rds')$workflow
block_best_rf = read_rds('./data/block_rf_res.rds')$fold_loss
block_best_rf_param = list(mtry = block_best_rf$mtry[1],
                           min_n = block_best_rf$min_n[1])

# Train Model
train_data_bk = rbind(bk_split$train, bk_split$val) %>% mutate(Label = factor(Label))
test_data_bk = bk_split$test %>% mutate(Label = factor(Label))
feature_names = train_data_bk %>% dplyr::select(-c(X,Y,Label, image)) %>% colnames()
formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 


rf_model = rand_forest(mode='classification', mtry=block_best_rf_param$mtry, trees=300, min_n = block_best_rf_param$min_n) %>%
  set_engine("ranger", num.threads=12, importance = "impurity")

bk_rf_fit = workflow() %>% add_recipe(recipe(formula, data = train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(rf_model)%>% fit(data = train_data_bk) 

bk_rf_pred = bk_rf_fit %>% augment(new_data =test_data_bk) 

# Diagnostic
p1 = block_best_rf_workflow %>% show_best(metric='roc_auc', 100) %>%
  ggplot(aes(x=mtry, y=mean, color=min_n)) + 
  geom_point()+
  scale_color_viridis_c()+
  labs(y='AUC', title='AUC')+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_line(stat="smooth",method = "lm", formula = y~poly(x,3),
            alpha = 0.5, color='blue')

p2 = block_best_rf_workflow %>% show_best(metric='accuracy', 100) %>%
  ggplot(aes(x=mtry, y=mean, color=min_n)) + 
  geom_point()+
  scale_color_viridis_c()+
  labs(y='Accuracy', title='Accuracy')+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_line(stat="smooth",method = "lm", formula = y~poly(x,3),
            alpha = 0.5, color='blue')

p = p1 + p2 + plot_layout(guides = "collect") + plot_annotation(
  # title='Random Forest: 5-Fold CV Performance',
  theme = theme(plot.title = element_text(hjust = 0.5)))
ggsave('./pic/diagnostic_cv_params.jpg', width=12, height=4, p)

## RF CV test: ntree convergence figure
rf_model = rand_forest(mode='classification', mtry=2, trees=tune(), min_n = 6) %>% 
  set_engine("ranger", num.threads=24)

rf_res = CVmaster(rf_model,
                  train_features = train_data_bk,
                  training_labels = train_data_bk$Label,
                  K = 5,
                  split_func = block_split,
                  metrics = metric_set(accuracy, roc_auc), .tune_grid = data.frame(trees=c(10, 50, 150, 300, 500, 800, 1500, 3000)))

res = do.call(rbind, rf_res$workflow %>% group_by(id) %>% pull(.metrics) ) %>% group_by(trees) %>% filter(.metric=='accuracy') 
res$Fold = fold=rep(1:8, time=5)
p1 = res %>% group_by(trees) %>% summarise(accuracy = mean(.estimate)) %>% ggplot() + geom_point(aes(x=trees, y=accuracy))+
  labs(y='Accuracy')

res = do.call(rbind, rf_res$workflow %>% group_by(id) %>% pull(.metrics) ) %>% group_by(trees) %>% filter(.metric=='roc_auc') 
res$Fold = fold=rep(1:8, time=5)
p2 = res %>% group_by(trees) %>% summarise(accuracy = mean(.estimate)) %>% ggplot() + geom_point(aes(x=trees, y=accuracy))+
  labs(y='AUC')

p = p1 + p2 + plot_layout(nrow=2)+ plot_annotation(
  theme = theme(plot.title = element_text(hjust = 0.5)))

ggsave('./pic/tree_convergence.jpg', p)

## Feature Importance
p = bk_rf_fit %>% 
  extract_fit_parsnip() %>% 
  # Make VIP plot
  vip() + labs(
    # title='Feature Importance by Impurity'
  )+
  theme(plot.title = element_text(hjust = 0.5))
ggsave('./pic/block_feature_importance.jpg', p)


## Decision Boundary
l = list(NDAI = seq(min(train_data_bk$NDAI), max(train_data_bk$NDAI), length.out=100),
         SD = seq(min(train_data_bk$SD), quantile(train_data_bk$SD,0.9), length.out=100),
         CORR = seq(min(train_data_bk$CORR), max(train_data_bk$CORR), length.out=100))
grid = do.call(expand.grid, l)
grid_data  = grid %>% mutate(Rad_Df = mean(train_data_bk$Rad_Df),
                             Rad_Cf = mean(train_data_bk$Rad_Cf),
                             Rad_Bf = mean(train_data_bk$Rad_Bf),
                             Rad_Af = mean(train_data_bk$Rad_Af),
                             Rad_An = mean(train_data_bk$Rad_An))

grid_pred = bk_rf_fit %>% augment(new_data=grid_data)


grid_pred = grid_pred %>% mutate(Prediction = factor(ifelse(.pred_class==1, 'Cloud', 'No Cloud')))
p = plot_ly(data=grid_pred, x=~NDAI,y=~SD, z=~CORR, color=~Prediction, alpha=0.01, 
            colors=c('white', 'red')) %>%add_markers(size=1)
p%>% layout(title = 'Random Forest: Decision Boundary')


## Plot Misclassified points
plot_misclass = function(test_data, predictions){
  
  my.cols = c('grey', 'white',"black")
  FP = predictions %>% filter(Label==-1 & .pred_class == 1)
  FN = predictions %>% filter(Label==1 & .pred_class == -1)
  
  for (img in 1:3){
    data = test_data %>% filter(image==img)
    
    if (dim(data)[1] > 0){
      require(data.table)
      allpossible <- do.call(CJ, lapply(data[, c("X", "Y")], unique))
      data = data  %>% 
        mutate(color = ifelse(Label==0, 'Unlabelled', ifelse(Label==1, 'Cloudy', 'Cloud-free')))
      p = ggplot() +
        geom_raster(aes(x=data$X, y=data$Y, fill=data$color)) +
        scale_fill_manual(values=my.cols)+
        geom_raster(data = allpossible[!data, on = c("X", "Y")], mapping=aes(X, Y), color = "black")+
        geom_point(mapping=aes(x=FP$X, y=FP$Y), color='red')+
        geom_point(mapping=aes(x=FN$X, y=FN$Y), color='blue')+
        theme(plot.title = element_text(hjust = 0.5),  legend.position="none")+
        labs(
          # title=paste0('Test Set (Image ', img,'): FP (red) & FN (blue)'),
          x='X', y='Y')+
        scale_color_manual(name='Error Type',
                           breaks=c('False Positive', 'False Negative'),
                           values=c('False Positive'='red', 'False Negative'='blue'))+
        scale_fill_manual(values=my.cols)
      
      ggsave('./pic/block_misclassified.jpg')
    }
    
  }
  
}
plot_misclass(test_data_bk, bk_rf_pred)

# Feature range of misclassified points
error_data = bk_rf_pred %>% mutate(Type = ifelse(Label==-1 & .pred_class == 1, 'False Positive', 
                                                 ifelse(Label==1 & .pred_class == -1, 'False Negative', 
                                                        ifelse(Label==1 & .pred_class == 1, 'True Positive', 'True Negative' ))))
p = error_data %>% mutate_at(4:11, funs(c(scale(.))))  %>% filter(Type %in% c('True Positive', 'False Positive'))%>%
  pivot_longer(4:11)%>% ggplot()+
  geom_boxplot(aes(x = name, y=value, fill=Type), outlier.size = 0.1, outlier.alpha = 0.5) + 
  labs(
    # title='Test Set Feature Range (Standardized)'
  )+theme(plot.title = element_text(hjust = 0.5))

ggsave('./pic/block_FP_feature_range.jpg')

tmp = error_data %>% pivot_longer(Rad_Df:Rad_An) %>% mutate(index = ifelse(name=='Rad_Af', 1, 
                                                                           ifelse(name=='Rad_An', 2,
                                                                                  ifelse(name=='Rad_Bf',3,
                                                                                         ifelse(name=='Rad_Cf',4,5)))))%>% group_by(X, Y) %>% mutate(deg_2_coeff = lm(value ~ poly(index,2) )%>% 
                                                                                                                                                       .$coefficients %>% .[3] %>% as.numeric())

p = tmp%>% 
  ggplot()+
  geom_density(aes(deg_2_coeff, fill=Type), alpha=0.5)+
  labs(x='Degree 2 Coefficient',
       # title='Density of Degree 2 Coefficient by Type'
  )

ggsave('./pic/block_deg2.jpg', p)

## New improved model
new_train_data_bk = rbind(bk_split$train, bk_split$val) %>% mutate(Label = factor(Label)) %>% pivot_longer(Rad_Df:Rad_An) %>% mutate(index = ifelse(name=='Rad_Af', 1, 
                                                                                                                                                    ifelse(name=='Rad_An', 2,
                                                                                                                                                           ifelse(name=='Rad_Bf',3,
                                                                                                                                                                  ifelse(name=='Rad_Cf',4,5)))))%>% group_by(X, Y) %>%
  mutate(deg_2_coeff = lm(value ~ poly(index,2) )%>% .$coefficients %>% .[3] %>% as.numeric()) %>% select(-index) %>% pivot_wider(names_from = name, values_from = value)%>% ungroup()


new_test_data_bk = bk_split$test %>% mutate(Label = factor(Label)) %>% pivot_longer(Rad_Df:Rad_An) %>% mutate(index = ifelse(name=='Rad_Af', 1, 
                                                                                                                             ifelse(name=='Rad_An', 2,
                                                                                                                                    ifelse(name=='Rad_Bf',3,
                                                                                                                                           ifelse(name=='Rad_Cf',4,5)))))%>% group_by(X, Y) %>%
  mutate(deg_2_coeff = lm(value ~ poly(index,2) )%>% .$coefficients %>% .[3] %>% as.numeric()) %>% select(-index) %>% pivot_wider(names_from = name, values_from = value) %>% ungroup()

feature_names = new_train_data_bk %>% dplyr::select(-c(X,Y,Label, image)) %>% colnames()
formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 

new_rf_model = rand_forest(mode='classification', mtry=block_best_rf_param$mtry, trees=500, min_n = block_best_rf_param$min_n) %>%
  set_engine("ranger", num.threads=12, importance = "impurity")

new_bk_rf_fit = workflow() %>% add_recipe(recipe(formula, data = new_train_data_bk) %>% step_normalize(all_numeric_predictors())) %>% add_model(new_rf_model)%>% fit(data = new_train_data_bk) 

new_bk_rf_pred = new_bk_rf_fit %>% augment(new_data =new_test_data_bk) 

new_bk_rf_pred%>% roc_auc(Label, `.pred_-1`)
new_bk_rf_pred %>% accuracy(Label, .pred_class)

## Feature importance of new model
p = new_bk_rf_fit %>% 
  extract_fit_parsnip() %>% 
  # Make VIP plot
  vip() +
  theme(plot.title = element_text(hjust = 0.5))
ggsave('./pic/block_deg2_feature_importance.jpg', p)

## ---------------- 5-fold CV ROCs for different folds of ALL models
cv_fold_roc = function(data, model, K){
  cv_splits = block_split(data, K=K)
  N = dim(data)[1]
  
  feature_names = data %>% dplyr::select(-c(X,Y,Label, image)) %>% colnames()
  formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 
  
  res = list()
  for (k in 1:K){
    
    val_idx = cv_splits[[k]]
    train_idx = seq(1, N)[-val_idx]
    train_data = data[train_idx, ]
    val_data = data[val_idx, ]
    
    
    res[[k]] = workflow() %>%
      add_recipe(recipe(formula, data = data) %>% step_normalize(all_numeric_predictors()))%>%
      add_model(model) %>% fit(data = train_data)  %>% augment(new_data =val_data)
    
    
  }
  res
}

lr_model = logistic_reg() %>% set_engine('glm')
lda_model = discrim_linear() %>% set_engine('MASS')
qda_model = discrim_quad() %>% set_engine('MASS')
rf_model = rand_forest(mode='classification', mtry=block_best_rf_param$mtry, trees=300, min_n = block_best_rf_param$min_n) %>%
  set_engine("ranger", num.threads=12, importance = "impurity")
NB_model = naive_Bayes(smoothness = 1.48, Laplace = 1.08)%>%set_mode('classification')

block_best_xgboost_workflow = read_rds('./data/block_xgboost_res.rds')$workflow
block_best_xgboost = read_rds('./data/block_xgboost_res.rds')$fold_loss
block_best_xgboost_param = list(tree_depth = block_best_xgboost$tree_depth[1],
                                min_n = block_best_xgboost$min_n[1],
                                learn_rate = block_best_xgboost$learn_rate[1],
                                loss_reduction = block_best_xgboost$loss_reduction[1],
                                sample_size = block_best_xgboost$sample_size[1],
                                stop_iter = block_best_xgboost$stop_iter[1]
)

xgboost_model = boost_tree(tree_depth = block_best_xgboost_param$tree_depth,
                           trees = 300,
                           learn_rate = block_best_xgboost_param$learn_rate, 
                           min_n = block_best_xgboost_param$min_n,
                           loss_reduction = block_best_xgboost_param$loss_reduction,
                           sample_size = block_best_xgboost_param$sample_size,
                           stop_iter = block_best_xgboost_param$stop_iter)%>%set_engine('xgboost', num.threads=24) %>%set_mode('classification')

cv_fold_roc_res = cv_fold_roc(data=train_data_bk, model=lr_model, K=5)
plot_data = c()
for (k in 1:5){
  auc = round(cv_fold_roc_res[[k]] %>% roc_auc(factor(Label),'.pred_-1') %>% pull(.estimate),3)
  d = cv_fold_roc_res[[k]] %>%
    yardstick::roc_curve(factor(Label),'.pred_-1') %>%
    mutate(Fold=paste0("Fold",k, ', AUC: ', auc))
  
  plot_data = rbind(plot_data, d)
}

p = plot_data%>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color=Fold))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() + labs(
  )
ggsave('./pic/block_CV_ROC_lr.jpg', p)




cv_fold_roc_res = cv_fold_roc(data=train_data_bk, model=lda_model, K=5)
plot_data = c()
for (k in 1:5){
  auc = round(cv_fold_roc_res[[k]] %>% roc_auc(factor(Label),'.pred_-1') %>% pull(.estimate),3)
  d = cv_fold_roc_res[[k]] %>%
    yardstick::roc_curve(factor(Label),'.pred_-1') %>%
    mutate(Fold=paste0("Fold",k, ', AUC: ', auc))
  
  plot_data = rbind(plot_data, d)
}

p = plot_data%>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color=Fold))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() + labs(
  )
ggsave('./pic/block_CV_ROC_lda.jpg', p)

cv_fold_roc_res = cv_fold_roc(data=train_data_bk, model=qda_model, K=5)
plot_data = c()
for (k in 1:5){
  auc = round(cv_fold_roc_res[[k]] %>% roc_auc(factor(Label),'.pred_-1') %>% pull(.estimate),3)
  d = cv_fold_roc_res[[k]] %>%
    yardstick::roc_curve(factor(Label),'.pred_-1') %>%
    mutate(Fold=paste0("Fold",k, ', AUC: ', auc))
  
  plot_data = rbind(plot_data, d)
}

p = plot_data%>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color=Fold))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() + labs(
  )
ggsave('./pic/block_CV_ROC_qda.jpg', p)


cv_fold_roc_res = cv_fold_roc(data=train_data_bk, model=NB_model, K=5)
plot_data = c()
for (k in 1:5){
  auc = round(cv_fold_roc_res[[k]] %>% roc_auc(factor(Label),'.pred_-1') %>% pull(.estimate),3)
  d = cv_fold_roc_res[[k]] %>%
    yardstick::roc_curve(factor(Label),'.pred_-1') %>%
    mutate(Fold=paste0("Fold",k, ', AUC: ', auc))
  
  plot_data = rbind(plot_data, d)
}

p = plot_data%>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color=Fold))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() + labs(
  )
ggsave('./pic/block_CV_ROC_nb.jpg', p)


cv_fold_roc_res = cv_fold_roc(data=train_data_bk, model=xgboost_model, K=5)
plot_data = c()
for (k in 1:5){
  auc = round(cv_fold_roc_res[[k]] %>% roc_auc(factor(Label),'.pred_-1') %>% pull(.estimate),3)
  d = cv_fold_roc_res[[k]] %>%
    yardstick::roc_curve(factor(Label),'.pred_-1') %>%
    mutate(Fold=paste0("Fold",k, ', AUC: ', auc))
  
  plot_data = rbind(plot_data, d)
}

p = plot_data%>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color=Fold))+
  geom_path(lwd = 0.6, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() + labs(
  )
ggsave('./pic/block_CV_ROC_xgboost.jpg', p)



