#-------------------------------------------------------------------------------------------------------------
library(tidyquant)
library(highcharter)
library(glue)
library(rlang)
library(htmltools)
library(tidyverse)

setwd("C:/Users/a.huseynli/Desktop/ML/R DA/R_1_3")
#Datanı read_delim-nən oxumaq daha sürətlidir. delim isə datanı göstərilən simvol üzrə sütunları ayırır
df<-read_delim('bank-full.csv',delim = ',')

#Datanı sətrlər üzrə qarışdırmaq. 
df[sample(nrow(df)),]->df

#Output olan sütunu faktor etmək.
df$y<-as.factor(df$y)

#---------------------------------------------Modelling------------------------------------------------------
library(h2o)

# Run h2O cluster
h2o.init()

# datanı h2o-ya çevirin
h2o_data<-as.h2o(df)

#datanı 3 hissəyə bolün
h2o_data<-h2o.splitFrame(h2o_data,ratios = c(0.7,0.15),seed=1)

# hesablama hissəsi
train<-h2o_data[[1]]

#proqnoz vermə hissəsi
test<-h2o_data[[2]]

#təsdiqləmə hissəsi
validation<-h2o_data[[3]]

# süni zəkaya hansı sütunu proqnoz verəcəyini göstəririk
outcome<-'y'

# bu kod isə yuxarıda göstərilən sütunu digərlərindən ayırır
# çünki biz yuxarıdakı sütunu çıxmaqla, digərləri əsasında həmin sütunu proqnoz edəcəyik.
features<-setdiff(colnames(df),outcome)

# modelin qurulması, 5 alqoritm ilə
# y -- bizə hansı sütunu proqnoz edəcəyimizi
# training_frame - hesablama hissəsi
# validation_frame - təsdiqləmə hissəsi
# leaderboard_frame  - proqnoz vermə hissəsi
# seed - təsadüfi ədədlərin yaradılması
# max_runtime_secs - modellərin qurulması üçün maksimum vaxtın verilməsi
# max_models - isə max_runtime_secs-in əvəzinə model sayını bildirir. Bu sayda model  qurulduqdan sonra süni zəka dayanır
# exclude_algos - isə həmin alqoritmləri hesablamadan çıxarır
aml<-h2o.automl(y=outcome,
                training_frame = train,
                validation_frame = validation,
                leaderboard_frame = test,seed=3,max_runtime_secs = 120)#,max_models = 2)#,exclude_algos = c("DRF", "GBM","GLM","DeepLearning","StackedEnsemble"))

xgboost<-h2o.xgboost(y=outcome,
                training_frame = train,
                validation_frame = validation,ntrees = 50,seed = 50)

# lider alqoritmi göstərir
aml@leader

#liderlərin sihasını əks etdirir.
aml@leaderboard %>% as.tibble() %>% head(.,20)

#---------------------------------------------Vizuallaşdırma------------------------------------------------------
# leaderboard-dan ilk nümunənin, yəni liderin adının çıxarılması
# . işarəsi datanı bildirir, ardıcılıqda yazıldığı üçün nöqtə ilə qeyd olunur
# [,1] - vergülün rəqəmnən qabaq olması sütun sayını, rəqəmnən sonra isə sətr sayı bildirir.
# str_split - veriləni müəyyən məsafəyə əsasən bölür. Bu nümunədə işarə kimi '_' qeyd olunub.

aml@leaderboard %>% as.tibble() %>% select(model_id) %>% .[,1] %>% .[1,] %>% 
  str_split(.,'_',simplify = T) %>% .[,1:1]->leader

# Confusion Matris səhvləri göstərir, output-da yazılan faktorlarl select(no,yes) yelərinə yazırıq.
# bizim nümunədə output no və yes olduğu üçün, select(np, yes) olubdur.
# ikinci olaraq isə modelin adını yazmalıyıq. Bizim nümunədə bu aml@leader-dir
h2o.confusionMatrix(aml@leader,test) %>% as.tibble() %>% select(no,yes) %>% as.matrix() %>% .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("#ed3b3b", "#0099ff"),
               margin = 1,main = paste(leader,
                                       round(sum(diag(.))/sum(.)*100,0),"%",sep = ' '))

#modeli save etmək üçün, slice sətri göstərir, pull çıxarmaq, get model - modeli əldə etmək,
# save isə yaddaşa qeyd etmək, path isə 'newww' qovluğunu yaradaraq modeli içinə yerləşdirir.
# slice daxilində olan 3 sətri bildirir, yəni 3-cü sətrdə olan model save olunsun
aml@leaderboard %>% as.tibble() %>% slice(3) %>% pull(model_id) %>%  h2o.getModel() %>% 
  h2o.saveModel(path = 'newww')

# modeli komyuterin yaddaşından yenidən açmaq üçün, amma bunun üçün ilk növbədə 58-83 sətrlərini RUN edin,
# əgər bu hissədən başlamaq istəyirsinizsə, yanı model artıq qurulubsa.
load<-h2o.loadModel('model_name')


# Area Under Curve (AUC) plotunun yaradılması üçün nəticələrin çıxarılması
h2o.performance(aml@leader,newdata = test) %>% h2o.metric() %>% select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(400,min=0.003,max=1)) %>% mutate(fpr_r=tpr_r) %>% arrange(tpr_r,fpr_r)->deep_metrics

# AUC göstəricisinin alınması
perf<-h2o.performance(aml@leader,newdata = test) %>% h2o.auc() %>% round(2)


# AUC plotunun çəkilməsi
highchart() %>% 
  # Data
  hc_add_series(deep_metrics, "scatter", hcaes(y = tpr, x = fpr), color='green',name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y = tpr_r, x = fpr_r), color='red',name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      list(
        point = list(
          xAxis = 0,
          yAxis = 0,
          x = 0.75,
          y = 0.25
        ),
        text = "Worse than guessing"
      ),  list( point = list(
        xAxis = 0,
        yAxis = 0,
        x = 0.3,
        y = 0.6
      ),
      text = glue('Better than guessing, AUC = {enexpr(perf)}')))) %>%
  hc_title(text = "ROC Curve") %>% hc_subtitle(text = "Model is performing much better than random guessing") 


# Kəsilmə xətti. Threshold və ya cutoff point deyilir. Yəni həmin bu nöqtədən sonra case-lər Yes qrupuna
# yəni 1-ə aid olacaqdır.
h2o.find_threshold_by_max_metric(h2o.performance(aml@leader,newdata = test),'f1') ->intercept




# Precision və Recall arasında tradeoff
highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y = precision,x=threshold), color='blue',name='Precision') %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y = recall,x=threshold), color='red',name='Recall') %>% 
  hc_xAxis(
    opposite = F,
    plotLines = list(
      list(label = list(text = glue('Max_threshold - {round(intercept,2)}')),
           color = "#FF0000",
           width = 3,
           value = glue('{intercept}'))))

# Modellərin siyahısı
aml@leaderboard %>% as.tibble() %>% select(model_id,auc,logloss) %>% 
  mutate_if(is.numeric,round,2)-> auc_log

# AUC və logloss-a görə sıralama, çoxdab-aza doğru
auc_log %>% gather(key=key,value = value, -model_id)  %>% 
  mutate(model = str_split(model_id, '_', simplify = T) %>% .[,1]) -> color_deep 

# Modellərin vizuallaşdırılması
hchart(auc_log %>% gather(key=key,value = value, -model_id),
       "bar", hcaes(x = model_id, y = value, group=key)) %>% 
  hc_plotOptions(series = list(stacking = "normal")) %>% 
  hc_xAxis(visible=T) %>% hc_yAxis(visible=T) %>% hc_colors(colors = c('green','red'))




#Gain plot
h2o.gainsLift(h2o.performance(aml@leader,newdata = test)) %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>% 
  select(-contains('lift')) %>% 
  mutate(base = cumulative_data_fraction) %>% 
  rename(gain = cumulative_capture_rate) %>% 
  gather(key = key, value = value, gain, base) %>% as.tibble()%>% mutate(choose_col=case_when(
    key == 'gain' ~ 'red',
    TRUE  ~ 'black'
  )) ->gain_chart

highchart() %>% hc_add_series(gain_chart,"line",hcaes(y=value,x=cumulative_data_fraction,group=choose_col),
                              showInLegend=F,name='Gain') %>% hc_yAxis(title=list(text = "Gain")) %>% 
  hc_xAxis(title=list(text='Cumulative Data Fraction')) %>% 
  hc_title(text='Gain chart', useHTML=T, align="center") %>% hc_add_theme(hc_theme_ffx(title = list(
    style = list(color = "red"))))%>% hc_colors(colors = c('black','red'))


#Lift plot 

h2o.gainsLift(h2o.performance(aml@leader,newdata = test)) %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>% 
  select(-cumulative_capture_rate) %>% 
  mutate(baseline = 1) %>% 
  rename(lift = cumulative_lift) %>% 
  gather(key = key, value = value, lift, baseline) %>% mutate(choose_col=case_when(
    value == 1 ~ 'black',
    TRUE  ~ 'red'
  )) ->lift_chart


highchart() %>% 
  hc_add_series(lift_chart,"line",hcaes(y=value,x=cumulative_data_fraction,group=choose_col),showInLegend=F,
                              name='Lift') %>% hc_yAxis(title=list(text = "Lift")) %>% 
  hc_xAxis(title=list(text='Cumulative Data Fraction')) %>% 
  hc_title(text='Lift chart', useHTML=T ,align="center") %>% hc_add_theme(hc_theme_ffx(title = list(
    style = list(
      color = "red",align='center')))) %>% hc_colors(colors = c('black','red'))



