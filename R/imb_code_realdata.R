########################################################
#Simple, DownSampling, UpSampling y SMOTE for Real Data#
########################################################
rm(list=ls())
library(MASS) # LDA
library(e1071)# SVM
library(rpart) # CART
library(randomForest) # RandomForest
source("functions_mcv3.R") # SAMME  You should copy this script in your working directory
library(pROC)#ROC y AUC
library(DMwR)
library(caret)

set.seed(2020)

iter=300


datos<-read.csv("dataFull_coli_Feb2020_pMathias.csv")
datos=datos[complete.cases(datos),]

##Simple
cf_cat=as.factor(datos$cf_cat)
datos=cbind(cf_cat,datos[,-1])
datos=datos[,-c(2:8,10:12,22,25,29,30,31)]#chequear
datalearn=datos[which(datos$isTEST=="NO_TEST"),-ncol(datos)]#42 es la columna de Notest/test
datatest=datos[which(datos$isTEST=="TEST"),-ncol(datos)]
#########



###Down
#cf_cat=as.factor(datos$cf_cat)
#datos=cbind(cf_cat,datos[,-1])
#datos=datos[,-c(2:8,10:12,22,25,29,30,31)]#chequear
#datalearn=datos[which(datos$isTEST=="NO_TEST"),-ncol(datos)]#42 es la columna de Notest/test
#datatest=datos[which(datos$isTEST=="TEST"),-ncol(datos)]
#datalearn=downSample(y=as.factor(datalearn$cf_cat),x=datalearn[,-1])
#cf_cat=as.factor(datalearn$Class)
#datalearn=cbind(cf_cat,datalearn[-c(1,ncol(datalearn))])
#######
########

###Up
#cf_cat=as.factor(datos$cf_cat)
#datos=cbind(cf_cat,datos[,-1])
#datos=datos[,-c(2:8,10:12,22,25,29,30,31)]#chequear
#datalearn=datos[which(datos$isTEST=="NO_TEST"),-ncol(datos)]#42 es la columna de Notest/test
#datatest=datos[which(datos$isTEST=="TEST"),-ncol(datos)]
#datalearn=upSample(y=as.factor(datalearn$cf_cat),x=datalearn[,-1])
#cf_cat=as.factor(datalearn$Class)
#datalearn=cbind(cf_cat,datalearn[-c(1,ncol(datalearn))])
#######


###SMOTE
#cf_cat=as.factor(datos$cf_cat)
#datos=cbind(cf_cat,datos[,-1])
#datos=datos[,-c(2:8,10:12,22,25,29,30,31)]#chequear
#datalearn=datos[which(datos$isTEST=="NO_TEST"),-ncol(datos)]#42 es la columna de Notest/test
#datatest=datos[which(datos$isTEST=="TEST"),-ncol(datos)]
#datalearn=SMOTE(cf_cat ~ ., data=datalearn , k=3)
#########




CUTOFF<- 0.5 # If posterior probability is > CUTOFF then classifier will choose the class- typical majority vote ==0.5
#CUTOFF=(sum(datalearn[,1]==1))/nrow(datalearn)

vif_calc(datos[,-c(1,2,6,8,19,42)]) #para hacer el vif saco las categoricas


dataglm=datos[,-c(2,6,8,9,10,17,22,23,24,25,26,27,28,29,30,31,32)]  #saco las que tienen un VIF grande (chequear)
dataglmlearn=dataglm[which(dataglm$isTEST=="NO_TEST"),-ncol(dataglm)]#33 es la columna Notest/test
dataglmtest=dataglm[which(dataglm$isTEST=="NO_TEST"),-ncol(dataglm)]


yobstst=as.numeric(datatest[,1])
yobstst1=as.numeric(dataglmtest[,1])



errores=as.list(0)
auc=as.list(0)
truePositive=as.list(0)
falsePositive=as.list(0)


  lda_error<-tryCatch(lda(cf_cat~.,data=dataglmlearn),error=function(e)e,finally="ERROR") # si da error, la prediccion es 0
  
  if(any(class(lda_error)=="error")){evdis=rep(0, nrow(dataglmtest)) } else {
    dis=lda(cf_cat~.,data=dataglmlearn); 
    pr.dis<-as.numeric(predict(dis,newdata=dataglmtest,type="prob")$posterior[,"1"])
    evdis=numeric()#as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
    evdis[pr.dis > CUTOFF]<- 2
    evdis[pr.dis <= CUTOFF]<- 1 # Acomoda para que las clases sean clasificadas como 1 o 2 segun el cutoff
    #evdis=as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
  }# When error, assume error in all predictions. Caused by error in lda because no variance
  
  #LR
  glm1<-stepAIC(glm(cf_cat~., data=dataglmlearn,family=binomial(link="logit")),trace=0)
  pr.glm<-predict(glm1, newdata=dataglmtest[,-1],type="response")
  evglm<-numeric()
  evglm[pr.glm >  CUTOFF]<-2 #Esta bien eso, el glm devuelve la proba de la clase minoritoria
  evglm[pr.glm <= CUTOFF]<-1
  
  
  #SVM
  #svmaa=tune(svm,train.x=dataglmlearn[,-1],train.y=dataglmlearn[,1],data=dataglmlearn)#No sé 
  #por qué no funciona.
  #svma=svm(datalearn[,1]~.,data=datalearn[,-1],cost=svmaa$best.model$cost,gamma=svmaa$best.model$gamma,probability=TRUE) 
  svma=svm(datalearn[,1]~.,data=datalearn[,-1],probability=TRUE)
  
  pr.svm<- as.numeric(attributes(predict(svma,newdata=datatest,probability=TRUE))$probabilities[,"1"])# Probability for class 1
  evsvm<-numeric()
  evsvm[pr.svm > CUTOFF]<-2
  evsvm[pr.svm <= CUTOFF]<-1
  
  #Non lineal
  #CART
  arbol=cart(datalearn,datatest,st=F,ms=5,cpopt=0.00000001)
  #cp.opt = arbol$cptable[which.min(arbol$cptable[,"xerror"]),"CP"] #Ya lo hace dentro de la funcion
  #arbol = cart(datalearn,datatest,st=F,ms=5,cpopt=cp.opt)
  pr.cart<-arbol$probTest
  ecart<-numeric()
  ecart[pr.cart >  CUTOFF]<-2
  ecart[pr.cart <= CUTOFF]<-1
  
  #SAMME
  samme=lastsamme(datalearn,datatest,nbiter=iter,st=F,ms=5,cpopt=0.0000001,cutoff=CUTOFF) 
  evsamme=samme$prevtst
  
  #RANDOM FOREST SIMPLE
  rf=randomForest(y=datalearn[,1],x=datalearn[,-1],data=datalearn,ntree=100)
 # varImpPlot(rf,main='Variable Importance',n.var=6)
  pr.rf= as.numeric(predict(rf,newdata=datatest,type="prob")[,"1"])
  evrf<-numeric()
  evrf[pr.rf >0.5]<-2
  evrf[pr.rf <= 0.5]<-1 # Aqui no puse el CUTOFF por que sino el metodo es equivalente a rf_cutoff que esta abajo (sirve como una especie de control)
  
  
  
  #Random Forest modificados para mejorar la prediccion
  # Ver https://stats.stackexchange.com/questions/168415/random-forest-in-r-using-unbalanced-data
  #RANDOM FOREST cutoff
  rf_cutoff=randomForest(y=datalearn[,1],x=datalearn[,-1],data=datalearn,ntree=iter, cutoff=c(CUTOFF, 1-CUTOFF)) # Following the receipe from internet...
  evrfcf= as.numeric(predict(rf_cutoff,newdata=datatest,type="class")) # n
  
  #RANDOM FOREST Stratified sampling
  sampsize<-min(table(datalearn[,1])) # Add this because gives error because random selection of learn set
  if(sampsize>5){
    rf_stratsamp=randomForest(y=datalearn[,1],x=datalearn[,-1],data=datalearn,ntree=iter, sampsize=c(sampsize, sampsize)) # 
    pr.rf_stratsamp= as.numeric(predict(rf_stratsamp,newdata=datatest,type="class"))
  } else pr.rf_stratsamp=NaN
  
  pr.rf_ss= predict(rf_stratsamp,newdata=datatest,type="prob")[,"1"]
  erfst<-numeric()
  erfst[pr.rf_ss > CUTOFF]<-2
  erfst[pr.rf_ss <=CUTOFF]<-1
  #table(erfst,yobstst)
  
  #RANDOM FOREST con classwt
  rf_wt=randomForest(y=datalearn[,1],x=datalearn[,-1],data=datalearn,ntree=iter, classwt=c(10,0.001)) # 
  pr.rfwt= as.numeric(predict(rf_wt,newdata=datatest,type="prob")[,"1"])
  erfcw<-numeric()
  erfcw[pr.rfwt >CUTOFF]<-2
  erfcw[pr.rfwt <= CUTOFF]<-1
  
  ### CALCULOS DE CALIDAD DE AJUSTE
  
  
  errores=cbind(LDA=sum(evdis!=yobstst1)/nrow(dataglmtest),
                  SVM=sum(evsvm!=yobstst)/nrow(datatest), 
                  LR=sum(evglm!=yobstst1)/nrow(dataglmtest), 
                  CART=sum(ecart !=yobstst)/nrow(datatest), 
                  SAMME=sum(evsamme!=yobstst)/nrow(datatest),
                  RF=sum(evrf !=yobstst)/nrow(datatest), 
                  RFcutoff=sum(evrfcf!=yobstst)/nrow(datatest), 
                  RF_strat=sum(erfst!=yobstst)/nrow(datatest), 
                  RF_classwt=sum(erfcw!=yobstst)/nrow(datatest))
  round(errores,2)
  
  
  # AUC
  auc=c(auc(as.numeric(yobstst1),evdis),
          auc(as.numeric(yobstst),evsvm),
          auc(as.numeric(yobstst1),evglm),
          auc(as.numeric(yobstst),evsamme),
          auc(as.numeric(yobstst),ecart),
          auc(as.numeric(yobstst),evrf),
          auc(as.numeric(yobstst),evrfcf),
          auc(as.numeric(yobstst),erfst),
          auc(as.numeric(yobstst),erfcw))
  
  # TRUE POSITIVE RATE
  truePositive= cbind(LDA=TP_fun(yobstst1,evdis)$TPR, SVM=TP_fun(yobstst,evsvm)$TPR, LR=TP_fun(yobstst1,evglm)$TPR,
                        CART=TP_fun(yobstst,ecart)$TPR, SAMME=TP_fun(yobstst,evsamme)$TPR, 
                        RF=TP_fun(yobstst,evrf)$TPR,RFcutoff=TP_fun(yobstst,evrfcf)$TPR, 
                        RF_strat=TP_fun(yobstst,erfst)$TPR, RF_classwt=TP_fun(yobstst,erfcw)$TPR)
  #FALSE POSITIVE RATE
  falsePositive= cbind(LDA=TP_fun(yobstst1,evdis)$FPR, SVM=TP_fun(yobstst,evsvm)$FPR, LR=TP_fun(yobstst1,evglm)$FPR,
                         CART=TP_fun(yobstst,ecart)$FPR, SAMME=TP_fun(yobstst,evsamme)$FPR, 
                         RF=TP_fun(yobstst,evrf)$FPR,RFcutoff=TP_fun(yobstst,evrfcf)$FPR, 
                         RF_strat=TP_fun(yobstst,erfst)$FPR, RF_classwt=TP_fun(yobstst,erfcw)$FPR)


ModNames<-c("LDA", "SVM", "LR","CART", "SAMME", "RF","RFcutoff", "RF_strat", "RF_classwt")



Acc<-round(1-errores,4)

TP<-round(truePositive,4)

FP<-round(falsePositive,4)

AUC<-round(auc,4)


Acc
TP
FP
AUC

