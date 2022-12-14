## -------------------------------  Load Data & Package ---------------------------------
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

## ------------------------------- Pairwise Correlation ----------------------------------
pdf(file = "./pic/corr.pdf")


col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
full_img %>% dplyr::select(-c(X,Y,Label, image)) %>% cor() %>%
  corrplot::corrplot(method="color", col=col(200),  
                     diag=T,  tl.srt=45, tl.col = 'black', tl.cex = 0.75,mar=c(0,0,2,0),
                     type="upper", order="hclust", 
                     # title='Feature Correlation', 
                     addCoef.col = "black",  outline=T,
  )


dev.off()
## -------------------------------  Label % By Image Table ---------------------------------
tbl1= full_img %>% filter(image==1) %>% group_by(Label) %>% count() %>%
  ungroup() %>% 
  mutate(Image1 = round(n / sum(n)*100,1)) %>% dplyr::select(Label, Image1)
tbl2= full_img %>% filter(image==2) %>% group_by(Label) %>% count() %>%
  ungroup() %>% 
  mutate(Image2 = round(n / sum(n)*100,1)) %>% dplyr::select(Label, Image2)
tbl3= full_img %>% filter(image==3) %>% group_by(Label) %>% count() %>%
  ungroup() %>% 
  mutate(Image3 = round(n / sum(n)*100,1)) %>% dplyr::select(Label, Image3)

tbl1 %>% left_join(tbl2, by='Label') %>% left_join(tbl3, by='Label') %>%
  mutate(Label=ifelse(Label==1, 'Cloudy (1)', ifelse(Label==-1, 'Cloud-free (-1)', 'Unlabeled (0)')))%>%
  arrange(Label)%>%
  kbl(
    # caption = "<center><strong>Pixels By Label (%)<center><strong>",
    escape = FALSE,
    format = 'html',) %>%
  kable_classic(full_width = F, html_font = "Cambria") %>% save_kable(file='pic/table_1.pdf')



## -------------------------------  Image Map ---------------------------------
p = full_img %>% filter(image==1) %>% 
  mutate(Label=
           as.factor(ifelse(Label==1, 'Cloudy (1)',
                            ifelse(Label==-1, 'Cloud-free (-1)',
                                   'Unlabeled (0)')))) %>% 
  ggplot() +
  geom_raster(aes(x=X, y=Y, fill=Label)) +
  scale_fill_manual(values=c("Cloudy (1)"='white', 
                             'Cloud-free (-1)'='grey',
                             'Unlabeled (0)'='black'))+
  labs(
    # title=paste0('Image ',1,' Label'),
    x='X', y='Y')+
  theme(plot.title = element_text(hjust = 0.5), legend.position="none")
ggsave(paste0('./pic/Image', 1,'_map.jpg'), p)


p = full_img %>% filter(image==2) %>% 
  mutate(Label=
           as.factor(ifelse(Label==1, 'Cloudy (1)',
                            ifelse(Label==-1, 'Cloud-free (-1)',
                                   'Unlabeled (0)')))) %>% 
  ggplot() +
  geom_raster(aes(x=X, y=Y, fill=Label)) +
  scale_fill_manual(values=c("Cloudy (1)"='white', 
                             'Cloud-free (-1)'='grey',
                             'Unlabeled (0)'='black'))+
  labs(
    # title=paste0('Image ',2,' Label'),
    x='X', y='Y')+
  theme(plot.title = element_text(hjust = 0.5), legend.position="none")
ggsave(paste0('./pic/Image', 2,'_map.jpg'), p)



p = full_img %>% filter(image==3) %>% 
  mutate(Label=
           as.factor(ifelse(Label==1, 'Cloudy (1)',
                            ifelse(Label==-1, 'Cloud-free (-1)',
                                   'Unlabeled (0)')))) %>% 
  ggplot() +
  geom_raster(aes(x=X, y=Y, fill=Label)) +
  scale_fill_manual(values=c("Cloudy (1)"='white', 
                             'Cloud-free (-1)'='grey',
                             'Unlabeled (0)'='black'))+
  labs(
    # title=paste0('Image ',3,' Label'),
    x='X', y='Y')+
  theme(plot.title = element_text(hjust = 0.5), legend.position="none")
ggsave(paste0('./pic/Image', 3,'_map.jpg'), p)


## -------------------------------  Feature Correlation with Label ---------------------------------
tbl_data = full_img %>% dplyr::select(-image)%>% filter(Label!=0) %>% cor() %>% .[-c(1,2,3),3] %>% round(.,2)

data.frame('value'=tbl_data) %>% arrange(desc(value))%>%kbl(
  # caption = "<center><strong>Correlation with Label<center><strong>",
  escape = FALSE,
  format = 'html',
  table.attr = "style='width:25%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",)  %>% save_kable(file='pic/table_2.pdf')

plot_data = full_img_no_unlabeled %>%
  mutate(Label = ifelse(Label==1, 'Cloudy', 'Cloud-free'))


## -------------------------------  Feature Density  ---------------------------------
p1 = plot_data%>%
  ggplot() + geom_density(aes(x=CORR, fill=Label), alpha=0.5) 
p2 = plot_data%>%
  ggplot() + geom_density(aes(x=NDAI, fill=Label), alpha=0.5) 
p3 = plot_data%>%
  ggplot() + geom_density(aes(x=log(SD), fill=Label), alpha=0.5) 
p4 = plot_data%>%
  ggplot() + geom_density(aes(x=Rad_Af, fill=Label), alpha=0.5) 

p = p1 + p2 + p3 + p4 + plot_layout(nrow=2, guides = "collect") + 
  plot_annotation(
    # title = 'Density of Features By Label',
    theme=theme(plot.title = element_text(hjust = 0.5))) 
ggsave(paste0('./pic/feature_density.jpg'), p)


## -------------------------------  Feature K-S Stat ---------------------------------
KS_stat = function(data){
  data_cloudy = data %>% filter(Label==1)
  data_not_cloudy = data %>%filter(Label==-1)
  
  rmv_cols = intersect(colnames(data), c('X', 'Y', 'Label', 'image'))
  
  var_names = data %>% dplyr::select(-rmv_cols) %>% colnames() 
  
  KS_stat = matrix(nrow=length(var_names))
  rownames(KS_stat) = var_names
  colnames(KS_stat) = 'Statistics'
  
  for (var in var_names){
    cloudy_density = data_cloudy %>% dplyr::select(var) %>% as.matrix()%>% density()
    not_cloudy_density = data_not_cloudy %>% dplyr::select(var)%>% as.matrix()%>% density()
    
    min_x = min(cloudy_density$x , not_cloudy_density$x)
    max_x = max(cloudy_density$x , not_cloudy_density$x)
    
    cloudy_density = data_cloudy %>% dplyr::select(var) %>% as.matrix()%>% 
      density(from=min_x, to=max_x)
    
    not_cloudy_density = data_not_cloudy %>% dplyr::select(var)%>% as.matrix()%>%
      density(from=min_x, to=max_x)
    
    x_diff = not_cloudy_density$x %>% diff() %>% .[1]
    
    KS = tibble(x=cloudy_density$x, cloudy=cloudy_density$y, not_cloudy= not_cloudy_density$y)%>%
      mutate(cloudy = cumsum(cloudy *x_diff),
             not_cloudy= cumsum(not_cloudy *x_diff))%>%
      mutate(diff = abs(cloudy - not_cloudy))%>%
      dplyr::select(diff)%>%max()
    KS_stat[var,1] = KS
  }
  KS_stat
}


stat = KS_stat(full_img)

stat %>% round(.,2) %>% data.frame() %>% arrange(desc(Statistics))%>%kbl(
  #caption = "<center><strong>Two-sample K-S Test: cloudy VS cloud-free<center><strong>",
  escape = FALSE,
  format = 'html',
  table.attr = "style='width:45%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",) %>% save_kable(file='pic/ks_stat.pdf')


## -------------------------------  PCA ---------------------------------
pca_data = full_img %>%dplyr::select(-c(X,Y,Label, image))
pca = prcomp(pca_data, scale.  =T)
explained_var = (pca$sdev^2) %>% cumsum()
explained_var = round(explained_var / explained_var[8]*100,1)
p = ggplot()+
  geom_point(aes(x=seq(1,8), y=explained_var))+
  geom_line(aes(x=seq(1,8), y=explained_var))+
  labs(x='PC', y='Cumulative explained variance (%)', 
       # title='Screeplot'
  )
ggsave(paste0('./pic/explained_var.jpg'),width=8, height=3, p)



pca$rotation %>% round(2) %>%kbl(
  # caption = "<center><strong>PCA<center><strong>",
  escape = FALSE,
  format = 'html',
  table.attr = "style='width:45%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",)%>% save_kable(file='pic/pca.pdf')


full_pca = tibble(data.frame(pca$x))
full_pca$X = full_img$X
full_pca$Y = full_img$Y
full_pca$Label = ifelse(full_img$Label==1,'Cloudy',ifelse(full_img$Label==-1,'Cloud-free',
                                                          'Unlabeled'))
full_pca$image = full_img$image

full_pca %>% dplyr::select(-image)%>% filter(Label!="Unlabeled")%>% 
  mutate(Label=ifelse(Label=='Cloudy', 1, -1)) %>%
  cor() %>% .[1:8,11] %>% round(.,2) %>% 
  data.frame('value'=.) %>% arrange(desc(value))%>%kbl(
    # caption = "<center><strong>Correlation with Label<center><strong>",
    escape = FALSE,
    format = 'html',
    table.attr = "style='width:25%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",) %>% save_kable(file='pic/pca_correlation_label.pdf')

## -------------------------------  Block Split Method ---------------------------------
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

plot_block = function(data, split_func, K, img, name){
  res = split_func(data, K)
  
  data['block'] = NA
  for (i in 1:K){
    data[res[[i]],'block'] = i
  }
  my.cols = brewer.pal(K, "Paired")
  my.cols = c(my.cols, "white")
  p = data %>% filter(image==img) %>% 
    mutate(color = ifelse(Label!=0, block, 'Cloud-free'))%>%
    ggplot() +
    geom_raster(aes(x=X, y=Y, fill=color)) +
    labs(
      # title='Image 1: blocks',
      x='X', y='Y')+
    scale_fill_manual(values=my.cols)+
    theme(plot.title = element_text(hjust = 0.5), legend.position="none")
  ggsave(paste0('./pic/', name, '.jpg'), p)
}

plot_block(full_img,block_split, 3, 1, 'blocksplit')
plot_block(full_img,Kmeans_block_split, 3, 1, 'kmeanssplit')

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


## -------------------------------  After-Split, Data point % by Label ---------------------------------
split_ratio = function(full_img_no_unlabeled, kmeans_split, bk_split){
  N = full_img_no_unlabeled %>% dim() %>% .[1]
  k_train_pct = (kmeans_split$train %>%dim() %>% .[1]) / N
  k_val_pct = (kmeans_split$val %>%dim() %>% .[1]) / N
  k_test_pct = (kmeans_split$test %>%dim() %>% .[1]) / N
  
  
  block_train_pct = (bk_split$train %>%dim() %>% .[1]) / N
  block_val_pct = (bk_split$val %>%dim() %>% .[1]) / N
  block_test_pct = (bk_split$test %>%dim() %>% .[1]) / N
  
  split_ratio = matrix(nrow = 3, ncol = 2)
  
  split_ratio[1,1] = k_train_pct
  split_ratio[2,1] = k_test_pct
  split_ratio[3,1] = k_val_pct
  split_ratio[1,2] = block_train_pct
  split_ratio[2,2] = block_test_pct
  split_ratio[3,2] = block_val_pct
  return(split_ratio)
}


tbl_data = data.frame(split_ratio(full_img_no_unlabeled, kmeans_split, bk_split) * 100 ) %>% round(.,1) 
colnames(tbl_data) = c('K means', 'Block split')
rownames(tbl_data) = c('Train', 'Test', 'Validation')

tbl_data %>%kbl(
  # caption = "<center><strong>Data Split Ratio (%) Excluding Unlabeled Points<center><strong>",
  escape = FALSE,
  format = 'html',
  table.attr = "style='width:45%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",) %>% save_kable(file='pic/train_test_val_split_ratio.pdf')


## -------------------------------  Baseline Accuracy ---------------------------------
baseline_accuracy = function(bk_split, kmeans_split){
  kmeans_valid_accuracy <- round(mean((kmeans_split$val  %>% pull(Label) )== -1),2)
  kmeans_test_accuracy <- round(mean((kmeans_split$test %>% pull(Label) )== -1),2)
  
  block_valid_accuracy <- round(mean((bk_split$val %>% pull(Label) )== -1),2)
  block_test_accuracy <- round(mean((bk_split$test %>% pull(Label) )== -1),2)
  
  res = matrix(nrow=4, ncol=1, dimnames = list(c('K-means Test', 
                                                 'K-means Validation',
                                                 'Block Test',
                                                 'Block Validation'), 'Accuracy'))
  
  res[1,1] = kmeans_test_accuracy
  res[2,1] = kmeans_valid_accuracy
  res[3,1] = block_test_accuracy
  res[4,1] = block_test_accuracy
  
  res %>% kbl(
    # caption = "<center><strong>Baseline Accuracy By Split Method<center><strong>",
    escape = FALSE,
    format = 'html',
    table.attr = "style='width:45%;'") %>%
    kable_classic(full_width = T, html_font = "Cambria",) %>% save_kable(file='pic/baseline_accu.pdf')
}

baseline_accuracy(bk_split, kmeans_split )


## -------------------------------  First order importance ---------------------------------
first_order_importance = function(full_img, kmeans_split, bk_split){
  variable_names = full_img %>% dplyr::select(-c(X, Y, Label, image)) %>% colnames()
  lr_model = logistic_reg() %>% set_engine('glm')
  
  Ktrain_data = kmeans_split$train %>% mutate(Label = as.factor(ifelse(Label==1, 'Cloudy', 'Cloudfree')))
  Kvalid_data = kmeans_split$val %>% mutate(Label = as.factor(ifelse(Label==1, 'Cloudy', 'Cloudfree')))
  
  Blocktrain_data = bk_split$train %>% mutate(Label = as.factor(ifelse(Label==1, 'Cloudy', 'Cloudfree')))
  Blockvalid_data = bk_split$val  %>% mutate(Label = as.factor(ifelse(Label==1, 'Cloudy', 'Cloudfree')))
  
  res = matrix(NA, nrow=length(variable_names), ncol = 3)
  rownames(res) = variable_names
  colnames(res) = c('AUC: K-means split', 'AUC: Block split', 'AUC: Avg')
  
  coeffs = matrix(NA, nrow=length(variable_names), ncol = 3)
  rownames(coeffs) = variable_names
  colnames(coeffs) = c('K-means split', 'Block split', 'Avg')
  
  stderrs = matrix(NA, nrow=length(variable_names), ncol = 3)
  rownames(stderrs) = variable_names
  colnames(stderrs) = c('K-means split', 'Block split', 'Avg')
  
  for (var in variable_names){
    # K-means
    lr_recipe = recipe(as.formula(paste0('Label ~ ', var)), data = Ktrain_data) %>%
      step_normalize(all_numeric_predictors())
    lr_work = workflow() %>% add_model(lr_model) %>% add_recipe(lr_recipe)
    lr_fit = lr_work %>% fit(data = Ktrain_data)
    lr_perf = lr_fit %>% augment(new_data = Kvalid_data) %>%
      dplyr::select(Label, starts_with('.pred')) 
    res[var,1] = lr_perf %>% roc_auc(Label, .pred_Cloudfree) %>% pull(.estimate)
    coeffs[var,1] = broom::tidy(lr_fit) %>% pull(estimate)%>%.[2]
    stderrs[var,1] = broom::tidy(lr_fit) %>% pull(std.error)%>%.[2]
    
    # BLock
    lr_recipe = recipe(as.formula(paste0('Label ~ ', var)), data = Blocktrain_data) %>%
      step_normalize(all_numeric_predictors())
    lr_fit = lr_work %>% fit(data = Blocktrain_data)
    lr_perf = lr_fit %>% augment(new_data = Blockvalid_data) %>%
      dplyr::select(Label, starts_with('.pred')) 
    res[var,2] = lr_perf %>% roc_auc(Label, .pred_Cloudfree) %>% pull(.estimate)
    coeffs[var,2] = broom::tidy(lr_fit) %>% pull(estimate)%>%.[2]
    stderrs[var,2] = broom::tidy(lr_fit) %>% pull(std.error)%>%.[2]
    
    res[var,3] = res[var,1]/2 + res[var,2]/2
    coeffs[var,3] = coeffs[var,1]/2 + coeffs[var,2]/2
    stderrs[var,3] = stderrs[var,1]/2 + stderrs[var,2]/2
    
  }
  return(list(res=res, coeffs = coeffs, stderrs=stderrs))
}

foi = first_order_importance(full_img, kmeans_split, bk_split)

(foi$res*100) %>% round(.,1) %>% as.data.frame() %>% arrange(desc('AUC: Avg')) %>% kbl(
  # caption = "<center><strong>Logistic Regression Performance<center><strong>",
  escape = FALSE,
  format = 'html',
  table.attr = "style='width:50%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",) %>%  save_kable(file='pic/foi_lr_auc.pdf')

foi$coeffs %>% round(.,2) %>% as.data.frame() %>% arrange(desc('Beta: Avg')) %>% kbl(
  # caption = "<center><strong>Logistic Regression Coefficient Estimate<center><strong>",
  escape = FALSE,
  format = 'html',
  table.attr = "style='width:50%;'") %>%
  kable_classic(full_width = T, html_font = "Cambria",) %>%  save_kable(file='pic/foi_lr_coeff.pdf')



## -------------------------------  First Order Importance Confidence Interval ---------------------------------
se_df = foi$stderrs%>% as.data.frame() %>% mutate(features=rownames(.)) %>% pivot_longer(1:2, names_to = 'split', values_to = 'se') %>%
  dplyr::select(-contains('Avg'))

p = foi$coeffs %>% as.data.frame() %>% mutate(features=rownames(.)) %>% pivot_longer(1:2, names_to = 'split', values_to = 'coef') %>%
  dplyr::select(-contains('Avg'))%>%
  left_join(se_df, by=c('features', 'split')) %>%
  mutate(lb=coef - 1.96 * se,
         ub = coef + 1.96 * se,
         features = factor(features, levels=(foi$coeffs %>%as.data.frame()%>%arrange(desc(Avg))%>%rownames())))%>%
  ggplot(aes(features, coef, color=split)) +
  geom_point(aes(shape=split),size=1, position=position_dodge(width=0.7)) +
  scale_color_manual(name="split",values=c("coral","steelblue")) +
  scale_shape_manual(name="split",values=c(17,19)) +
  theme_bw() +
  geom_errorbar(aes(ymin=lb,ymax=ub),width=0.7,position=position_dodge(width=0.7))+
  labs(
    # title="95% CI of Logistic Regression Coefficient",
    y='Beta')+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_color_discrete(name = "Split Method")+
  scale_shape_discrete(name="Split Method")
ggsave(paste0('./pic/CI_lr_coeff.jpg'), width=8, height=3,p)

