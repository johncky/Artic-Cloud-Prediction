source("CVmaster.R")

require(tidyverse)
require(kableExtra)
require(patchwork)
require(tidymodels)
require(ranger)
require(vip)
require(plotly)
set.seed(666)

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
kmeans_best_rf_workflow = read_rds('./data/Kmeans_rf_res.rds')$workflow
kmeans_best_rf = read_rds('./data/Kmeans_rf_res.rds')$fold_loss
kmeans_best_rf_param = list(mtry = kmeans_best_rf$mtry[1],
                            min_n = kmeans_best_rf$min_n[1])


# Train Model
train_data_kmeans = rbind(kmeans_split$train, kmeans_split$val) %>% mutate(Label = factor(Label))
test_data_kmeans = kmeans_split$test %>% mutate(Label = factor(Label))
feature_names = train_data_kmeans %>% dplyr::select(-c(X,Y,Label, image)) %>% colnames()
formula = as.formula(paste0('Label ~ ', paste(feature_names, collapse = ' + '))) 


rf_model = rand_forest(mode='classification', mtry=kmeans_best_rf_param$mtry, trees=300, min_n = kmeans_best_rf_param$min_n) %>%
  set_engine("ranger", num.threads=12, importance = "impurity")

kmeans_rf_fit = workflow() %>% add_recipe(recipe(formula, data = train_data_kmeans) %>% step_normalize(all_numeric_predictors())) %>% add_model(rf_model)%>% fit(data = train_data_kmeans) 

kmeans_rf_pred = kmeans_rf_fit %>% augment(new_data =test_data_kmeans) 

kmeans_rf_pred%>% accuracy(Label, .pred_class)

kmeans_rf_pred%>% roc_auc(Label, `.pred_-1`)

# Varying parameter CV result plto
p1 = kmeans_best_rf_workflow %>% show_best(metric='roc_auc', 100) %>%
  ggplot(aes(x=mtry, y=mean, color=min_n)) + 
  geom_point()+
  scale_color_viridis_c()+
  labs(y='AUC', title='AUC')+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_line(stat="smooth",method = "lm", formula = y~poly(x,3),
            alpha = 0.5, color='blue')

p2 = kmeans_best_rf_workflow %>% show_best(metric='accuracy', 100) %>%
  ggplot(aes(x=mtry, y=mean, color=min_n)) + 
  geom_point()+
  scale_color_viridis_c()+
  labs(y='Accuracy', title='Accuracy')+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_line(stat="smooth",method = "lm", formula = y~poly(x,3),
            alpha = 0.5, color='blue')

p = p1 + p2 + plot_layout(guides = "collect") + plot_annotation(
  # title='Random Forest: 5-Fold CV (K-means Split) Performance',
  theme = theme(plot.title = element_text(hjust = 0.5)))
ggsave('./pic/diagnostic_cv_params_kmeans.jpg', p)

## Varying tree numbers plot
rf_model = rand_forest(mode='classification', mtry=2, trees=tune(), min_n = 6) %>% 
  set_engine("ranger", num.threads=24)

rf_res = CVmaster(rf_model,
                  train_features = train_data_kmeans,
                  training_labels = train_data_kmeans$Label,
                  K = 5,
                  split_func = Kmeans_block_split,
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

ggsave('./pic/tree_convergence_kmeans.jpg', p)


## Feature Importance plot
p = kmeans_rf_fit %>% 
  extract_fit_parsnip() %>% 
  # Make VIP plot
  vip() +
  theme(plot.title = element_text(hjust = 0.5))
ggsave('./pic/kmeans_feature_importance.jpg', p)

## Decision Boundary
l = list(NDAI = seq(min(train_data_kmeans$NDAI), max(train_data_kmeans$NDAI), length.out=100),
         SD = seq(min(train_data_kmeans$SD), quantile(train_data_kmeans$SD,0.9), length.out=100),
         CORR = seq(min(train_data_kmeans$CORR), max(train_data_kmeans$CORR), length.out=100))
grid = do.call(expand.grid, l)
grid_data  = grid %>% mutate(Rad_Df = mean(train_data_kmeans$Rad_Df),
                             Rad_Cf = mean(train_data_kmeans$Rad_Cf),
                             Rad_Bf = mean(train_data_kmeans$Rad_Bf),
                             Rad_Af = mean(train_data_kmeans$Rad_Af),
                             Rad_An = mean(train_data_kmeans$Rad_An))

grid_pred = kmeans_rf_fit %>% augment(new_data=grid_data)

grid_pred = grid_pred %>% mutate(Prediction = factor(ifelse(.pred_class==1, 'Cloud', 'No Cloud')))
p = plot_ly(data=grid_pred, x=~NDAI,y=~SD, z=~CORR, color=~Prediction, alpha=0.01, 
            colors=c('white', 'red')) %>%add_markers(size=1)
p%>% layout(title = 'Random Forest: Decision Boundary (K-means Split)')


## Plot misclassified
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
        labs(x='X', y='Y')+
        scale_color_manual(name='Error Type',
                           breaks=c('False Positive', 'False Negative'),
                           values=c('False Positive'='red', 'False Negative'='blue'))+
        scale_fill_manual(values=my.cols)
      
      ggsave('./pic/kmeans_misclassified.jpg')
      print(p)
    }
    
  }
  
}
plot_misclass(test_data_kmeans, kmeans_rf_pred)


