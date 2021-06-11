##################################
#DownSampling, UpSampling y SMOTE#
##################################
rm(list=ls())
library(MASS) # LDA
library(e1071)# SVM

library(rpart) # CART
#library(tree)  # CART
library(randomForest) # RandomForest
source("functions_mcv3.R") # SAMME  You should copy this script in your working directory

#install.packages("pROC")
library(pROC)#ROC y AUC



#############################################################################################################################
#############################################################################################################################
## DATOS SIMULADOS PARA USAR COMO EJEMPLO - Tshiribarni ISRL pg 315
#############################################################################################################################
#############################################################################################################################
Nobs<- 200
set.seed(69)
datay<-sample(letters[1:2], Nobs,replace=T, prob=c(0.10,0.90))
datax<-matrix(runif(Nobs*4,-1,1),ncol=4,nrow=Nobs)

## Datos no lineales
sel1<-which(datax[,1]>0.5 & datax[,2]>0.5)   #A
sel2<-which(datax[,1]<0.5)                   #B
sel3<-which(datax[,2]<0.5)                   #B

if(length(sel1)>0) 		datay[sel1]<-      "a"
if(length(c(sel2)>0)) 	        datay[sel2]<-      "b"
if(length(sel3)>0) 		datay[sel3]<-      "b"

## ADD some noise (change majority class to minority class to avoid perfect classification)
set.seed(69)
selb<-which(datay=="b")
datay[selb[sample(selb,3)]]<-"a" 

coly<-as.factor(datay)
levels(coly)<- c("red","green")

## Datos lineales
datayl<-datay
sel1l<-which(datax[,1]<datax[,2])   				#B
sel2l<-which(datax[,1]>datax[,2] & datax[,1]-datax[,2]<1) 	#B
sel3l<-which(datax[,1]>datax[,2] & datax[,1]-datax[,2]>1) 	#A

if(length(sel1l)>0) datayl[sel1l]<- "b"
if(length(sel2l)>0) datayl[sel2l]<- "b"
if(length(sel3l)>0) datayl[sel3l]<- "a"

## ADD some noise (change majority class to minority class to avoid perfect classification)
set.seed(69)
selbl<-which(datayl=="b")
datayl[selbl[sample(selbl,3)]]<- "a"


colyl<-as.factor(datayl)
levels(colyl)<- c("red","green")

## FIGURE Structure of simulated data

#pdf("FIGURAS/Figure_3_simulatedData.pdf", height=8, width=8)
x11(); 
par(mfrow=c(2,2))
plot(datax[,1],datax[,2], pch=datayl, col=as.character(colyl),xlab="", ylab="", cex.lab=1.5)
mtext(expression(x^(1)),1,line=3, cex=1.5)
mtext(expression(x^(2)),2,line=2, cex=1.5)
title("Linear data (L set)")
abline(a=-1,b=1,lty=2)

plot(datax[,1],datax[,2], pch=datay, col=as.character(coly),xlab="", ylab="")
mtext(expression(x^(1)),1,line=3, cex=1.5)
mtext(expression(x^(2)),2,line=2, cex=1.5)
title("Non-linear data (NL set)")
abline(h=0.5,v=0.5,lty=2)

plot(datax[,1],datax[,3], pch=datayl, col=as.character(colyl),xlab="", ylab="")
mtext(expression(x^(1)),1,line=3, cex=1.5)
mtext(expression(x^(3)),2,line=2, cex=1.5)

plot(datax[,1],datax[,3], pch=datay, col=as.character(coly),xlab="", ylab="")
mtext(expression(x^(1)),1,line=3, cex=1.5)
mtext(expression(x^(3)),2,line=2, cex=1.5)

dev.off()
#http://stats.stackexchange.com/questions/92157/compute-and-graph-the-lda-decision-boundary


datax 
table(datay)
table(datayl)




##############
#DownSampling#
##############

library(DMwR)
library(caret)
simul_data_nl<-data.frame(datay,datax) # 

errores=as.list(0)
nbiter=300

nsim=100 # numero de simulaciones para evaluar distribucion de errores

ERRORES<-matrix(NaN, ncol=6, nrow=nsim) # modelos x simulaciones
AUCs<-matrix(NaN, ncol=6, nrow=nsim) # Matrix con las Areas bajo la curva
TP_MAT<-matrix(NaN, ncol=6, nrow=nsim) # Matriz con los true positive calculados por TP_fun
FP_MAT<-matrix(NaN, ncol=6, nrow=nsim) # Matriz con los true positive calculados por FP_fun

ksamme=numeric()
CUTOFF=0.5

for(k in 1:nsim){
  
  print(k)
  n=nrow(simul_data_nl)
  s=sample(n,n/3)
  dlearn=simul_data_nl[-s,]
  while(dim(table(dlearn[,1]))==1){dlearn=simul_data_nl[-s,]}
  ## AGREGA downSample
  dlearn=downSample(x = dlearn[,-1],y=dlearn[,1])
  dlearn<-dlearn[,c(ncol(dlearn):1)]
  colnames(dlearn)[1]<-"datay"
  cl=dlearn[,1]
  dtest=simul_data_nl[s,]
  dtest=dtest[complete.cases(dtest),]
  yobstst=dtest[ ,1]
  yobstst=as.numeric(yobstst)
  
  #DISCRIMINANT ANALYSIS
  lda_error<-tryCatch(lda(datay~.,data=dlearn),error=function(e)e,finally="ERROR") # si da error, la prediccion es 0
  
  if(any(class(lda_error)=="error")){evdis=rep(0, nrow(dtest)) } else {
    dis=lda(datay~.,data=dlearn); 
    pr.dis<-as.numeric(predict(dis,newdata=dtest,type="prob")$posterior[,"a"])
    evdis=numeric()#as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
    evdis[pr.dis > CUTOFF]<- 1
    evdis[pr.dis <= CUTOFF]<- 2 # Acomoda para que las clases sean clasificadas como 1 o 2 segun el cutoff
    #evdis=as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
  }# When error, assume error in all predictions. Caused by error in lda because no variance
  
  #LR
  glm1<-stepAIC(glm(datay~., data=dlearn,family=binomial(link="logit")),trace=0)
  pr.glm<-predict(glm1, newdata=dtest[,-1],type="response")
  evglm<-numeric()
  evglm[pr.glm >  CUTOFF]<-2
  evglm[pr.glm <= CUTOFF]<-1
  
  
  #SVM
  svmaa=tune(svm,train.x=dlearn[,-1],train.y=dlearn[,1],data=dlearn[,-1])
  svma=svm(dlearn[,1]~.,data=dlearn[,-1],cost=svmaa$best.model$cost,gamma=svmaa$best.model$gamma,probability=TRUE)
  pr.svm<- as.numeric(attributes(predict(svma,newdata=dtest,probability=TRUE))$probabilities[,"a"])# Probability for class 1
  evsvm<-numeric()
  evsvm[pr.svm > CUTOFF]<-1
  evsvm[pr.svm <= CUTOFF]<-2
  
  #Non lineal
  #CART
  arbol=cart(dlearn,dtest,st=F,ms=5,cpopt=0.000001)
  pr.cart<-arbol$probTest
  ecart<-numeric()
  ecart[pr.cart >  CUTOFF]<-1
  ecart[pr.cart <= CUTOFF]<-2
  
  #SAMME
  samme_error<-tryCatch(lastsamme(dlearn,dtest,nbiter=300,st=F,ms=5,cpopt=0,cutoff=CUTOFF),error=function(e)e,finally="ERROR") 

  if(any(class(samme_error)=="error")){evsamme=rep(2, nrow(dtest));ksamme=c(ksamme,k) } else {
  evsamme=samme_error$prevtst}
  
  #samme=lastsamme(dlearn,dtest,nbiter=300,st=F,ms=5,cpopt=0,cutoff=CUTOFF) 
  #evsamme=samme$prevtst
  
  #RANDOM FOREST SIMPLE
  rf=randomForest(y=dlearn[,1],x=dlearn[,-1],data=dlearn,ntree=1000)
  pr.rf= as.numeric(predict(rf,newdata=dtest,type="prob")[,"a"])
  evrf<-numeric()
  evrf[pr.rf >0.5]<-1
  evrf[pr.rf <= 0.5]<-2 # Aqui no puse el CUTOFF por que sino el metodo es equivalente a rf_cutoff que esta abajo (sirve como una especie de control)
  
  
  
  ### CALCULOS DE CALIDAD DE AJUSTE
  
  errores=cbind(LDA=sum(evdis!=yobstst)/nrow(dtest),
                  SVM=sum(evsvm!=yobstst)/nrow(dtest), 
                  LR=sum(evglm!=yobstst)/nrow(dtest), 
                  CART=sum(ecart!=yobstst)/nrow(dtest),
                 SAMME=sum(evsamme!=yobstst)/nrow(dtest),
                  RF=sum(evrf!=yobstst)/nrow(dtest))#
  round(errores,2)
  
  ERRORES[k,]<-errores
  
  # AUC
  auc=c(auc(as.numeric(yobstst),evdis),
          auc(as.numeric(yobstst),evsvm),
          auc(as.numeric(yobstst),evglm),
         auc(as.numeric(yobstst),evsamme),
          auc(as.numeric(yobstst),ecart),
          auc(as.numeric(yobstst),evrf))
  AUCs[k,]<-auc
  
  # TRUE POSITIVE
  truePositive= cbind(LDA=TP_fun(yobstst,evdis)$TPR, SVM=TP_fun(yobstst,evsvm)$TPR, 
                        LR=TP_fun(yobstst,evglm)$TPR,CART=TP_fun(yobstst,ecart)$TPR, 
                        SAMME=TP_fun(yobstst,evsamme)$TPR,
                        RF=TP_fun(yobstst,evrf)$TPR)
  TP_MAT[k,]<-truePositive
  
  falsePositive= cbind(LDA=TP_fun(yobstst,evdis)$FPR, SVM=TP_fun(yobstst,evsvm)$FPR, LR=TP_fun(yobstst,evglm)$FPR,
                         CART=TP_fun(yobstst,ecart)$FPR, 
                         SAMME=TP_fun(yobstst,evsamme)$FPR, 
                         RF=TP_fun(yobstst,evrf)$FPR)
  FP_MAT[k,]<-falsePositive
  
  
}# END LOOP      


ModNames<-cbind("LDA", "SVM", "LR","CART", "SAMME", "RF")
#ModNames<-cbind("LDA", "SVM", "LR","CART", "RF")

ERRORES=ERRORES[-ksamme,]
TP_MAT=TP_MAT[-ksamme,]
FP_MAT=FP_MAT[-ksamme,]
AUCs=AUCs[-ksamme,]

MeanAcc<-apply(1-ERRORES, 2, mean)
MeanAcc=round(MeanAcc,4)
SDAcc<-round(apply(1-ERRORES, 2, sd),2)
Acctab<-t(data.frame(paste(MeanAcc, SDAcc, sep="Ypm")))
colnames(Acctab)=ModNames
rownames(Acctab)=c("ACC")

MeanTP_MAT<-apply(TP_MAT, 2, mean)
MeanTP_MAT=round(MeanTP_MAT,4)
SDTP_MAT<-round(apply(TP_MAT, 2, sd),2)
TPRtab<-t(data.frame(paste(MeanTP_MAT, SDTP_MAT, sep="Ypm")))
colnames(TPRtab)=ModNames
rownames(TPRtab)=c("TPR")

MeanFP_MAT<-apply(FP_MAT, 2, mean)
MeanFP_MAT=round(MeanFP_MAT,4)
SDFP_MAT<-round(apply(FP_MAT, 2, sd),2)
FPRtab<-t(data.frame(paste(MeanFP_MAT, SDFP_MAT, sep="Ypm")))
colnames(FPRtab)=ModNames
rownames(FPRtab)=c("FPR")


MeanAUC<-apply(AUCs, 2, mean)
MeanAUC=round(MeanAUC,4)
SDAUC<-round(apply(AUCs, 2, sd),2)
AUCtab<-t(data.frame(paste(MeanAUC, SDAUC, sep="Ypm")))
colnames(AUCtab)=ModNames
rownames(AUCtab)=c("AUC")

tabla_down_nl=rbind(Acctab,AUCtab,TPRtab,FPRtab)


fic = paste("resdownnl.txt",sep="")

dump(paste("ERRORES"),file=fic,append=T)
dump(paste("TP_MAT"),file=fic,append=T)
dump(paste("FP_MAT"),file=fic,append=T)
dump(paste("AUCs"),file=fic,append=T)
dump(paste("tabla_down_nl"),file=fic,append=T)

xtable(tabla_down_nl,caption = "Tabla Down No Lineal")





############
#UpSampling#
############
library(DMwR)
library(caret)
simul_data_nl<-data.frame(datay,datax) # 

errores=as.list(0)
nbiter=300

nsim=100 # numero de simulaciones para evaluar distribucion de errores

ERRORES<-matrix(NaN, ncol=6, nrow=nsim) # modelos x simulaciones
AUCs<-matrix(NaN, ncol=6, nrow=nsim) # Matrix con las Areas bajo la curva
TP_MAT<-matrix(NaN, ncol=6, nrow=nsim) # Matriz con los true positive calculados por TP_fun
FP_MAT<-matrix(NaN, ncol=6, nrow=nsim) # Matriz con los true positive calculados por FP_fun

CUTOFF=0.5

for(k in 1:nsim){
  
  print(k)
  n=nrow(simul_data_nl)
  s=sample(n,n/3)
  dlearn=simul_data_nl[-s,]
  
 
  dlearn=upSample(x = dlearn[,-1],y=dlearn[,1])
  dlearn<-dlearn[,c(ncol(dlearn):1)]
  colnames(dlearn)[1]<-"datay"
  cl=dlearn[,1]
  dtest=simul_data_nl[s,]
  dtest=dtest[complete.cases(dtest),]
  yobstst=dtest[ ,1]
  yobstst=as.numeric(yobstst)
  
  #DISCRIMINANT ANALYSIS
  lda_error<-tryCatch(lda(datay~.,data=dlearn),error=function(e)e,finally="ERROR") # si da error, la prediccion es 0
  
  if(any(class(lda_error)=="error")){evdis=rep(0, nrow(dtest)) } else {
    dis=lda(datay~.,data=dlearn); 
    pr.dis<-as.numeric(predict(dis,newdata=dtest,type="prob")$posterior[,"a"])
    evdis=numeric()#as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
    evdis[pr.dis > CUTOFF]<- 1
    evdis[pr.dis <= CUTOFF]<- 2 # Acomoda para que las clases sean clasificadas como 1 o 2 segun el cutoff
    #evdis=as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
  }# When error, assume error in all predictions. Caused by error in lda because no variance
  
  #LR
  glm1<-stepAIC(glm(datay~., data=dlearn,family=binomial(link="logit")),trace=0)
  pr.glm<-predict(glm1, newdata=dtest[,-1],type="response")
  evglm<-numeric()
  evglm[pr.glm >  CUTOFF]<-2
  evglm[pr.glm <= CUTOFF]<-1
  
  
  #SVM
  svmaa=tune(svm,train.x=dlearn[,-1],train.y=dlearn[,1],data=dlearn[,-1])
  svma=svm(dlearn[,1]~.,data=dlearn[,-1],cost=svmaa$best.model$cost,gamma=svmaa$best.model$gamma,probability=TRUE)
  pr.svm<- as.numeric(attributes(predict(svma,newdata=dtest,probability=TRUE))$probabilities[,"a"])# Probability for class 1
  evsvm<-numeric()
  evsvm[pr.svm > CUTOFF]<-1
  evsvm[pr.svm <= CUTOFF]<-2
  
  #Non lineal
  #CART
  arbol=cart(dlearn,dtest,st=F,ms=5,cpopt=0.000001)
  pr.cart<-arbol$probTest
  ecart<-numeric()
  ecart[pr.cart >  CUTOFF]<-1
  ecart[pr.cart <= CUTOFF]<-2
  
  #SAMME
  samme=lastsamme(dlearn,dtest,nbiter=300,st=F,ms=5,cpopt=0.00001,cutoff=CUTOFF) 
  evsamme=samme$prevtst
  
  #RANDOM FOREST SIMPLE
  rf=randomForest(y=dlearn[,1],x=dlearn[,-1],data=dlearn,ntree=1000)
  pr.rf= as.numeric(predict(rf,newdata=dtest,type="prob")[,"a"])
  evrf<-numeric()
  evrf[pr.rf >0.5]<-1
  evrf[pr.rf <= 0.5]<-2 # Aqui no puse el CUTOFF por que sino el metodo es equivalente a rf_cutoff que esta abajo (sirve como una especie de control)
  
  
  
  ### CALCULOS DE CALIDAD DE AJUSTE
  
  errores=cbind(LDA=sum(evdis!=yobstst)/nrow(dtest),
                SVM=sum(evsvm!=yobstst)/nrow(dtest), 
                LR=sum(evglm!=yobstst)/nrow(dtest), 
                CART=sum(ecart!=yobstst)/nrow(dtest),
                SAMME=sum(evsamme!=yobstst)/nrow(dtest),
                RF=sum(evrf!=yobstst)/nrow(dtest))#
  round(errores,2)
  
  ERRORES[k,]<-errores
  
  # AUC
  auc=c(auc(as.numeric(yobstst),evdis),
        auc(as.numeric(yobstst),evsvm),
        auc(as.numeric(yobstst),evglm),
        auc(as.numeric(yobstst),evsamme),
        auc(as.numeric(yobstst),ecart),
        auc(as.numeric(yobstst),evrf))
  AUCs[k,]<-auc
  
  # TRUE POSITIVE
  truePositive= cbind(LDA=TP_fun(yobstst,evdis)$TPR, SVM=TP_fun(yobstst,evsvm)$TPR, 
                      LR=TP_fun(yobstst,evglm)$TPR,CART=TP_fun(yobstst,ecart)$TPR, 
                      SAMME=TP_fun(yobstst,evsamme)$TPR,
                      RF=TP_fun(yobstst,evrf)$TPR)
  TP_MAT[k,]<-truePositive
  
  falsePositive= cbind(LDA=TP_fun(yobstst,evdis)$FPR, SVM=TP_fun(yobstst,evsvm)$FPR, LR=TP_fun(yobstst,evglm)$FPR,
                       CART=TP_fun(yobstst,ecart)$FPR, 
                       SAMME=TP_fun(yobstst,evsamme)$FPR, 
                       RF=TP_fun(yobstst,evrf)$FPR)
  FP_MAT[k,]<-falsePositive
  
  
}# END LOOP      


ModNames<-cbind("LDA", "SVM", "LR","CART", "SAMME", "RF")
#ModNames<-cbind("LDA", "SVM", "LR","CART", "RF")



MeanAcc<-apply(1-ERRORES, 2, mean)
MeanAcc=round(MeanAcc,4)
SDAcc<-round(apply(1-ERRORES, 2, sd),2)
Acctab<-t(data.frame(paste(MeanAcc, SDAcc, sep="Ypm")))
colnames(Acctab)=ModNames
rownames(Acctab)=c("ACC")

MeanTP_MAT<-apply(TP_MAT, 2, mean)
MeanTP_MAT=round(MeanTP_MAT,4)
SDTP_MAT<-round(apply(TP_MAT, 2, sd),2)
TPRtab<-t(data.frame(paste(MeanTP_MAT, SDTP_MAT, sep="Ypm")))
colnames(TPRtab)=ModNames
rownames(TPRtab)=c("TPR")

MeanFP_MAT<-apply(FP_MAT, 2, mean)
MeanFP_MAT=round(MeanFP_MAT,4)
SDFP_MAT<-round(apply(FP_MAT, 2, sd),2)
FPRtab<-t(data.frame(paste(MeanFP_MAT, SDFP_MAT, sep="Ypm")))
colnames(FPRtab)=ModNames
rownames(FPRtab)=c("FPR")


MeanAUC<-apply(AUCs, 2, mean)
MeanAUC=round(MeanAUC,4)
SDAUC<-round(apply(AUCs, 2, sd),2)
AUCtab<-t(data.frame(paste(MeanAUC, SDAUC, sep="Ypm")))
colnames(AUCtab)=ModNames
rownames(AUCtab)=c("AUC")

tabla_up_nl=rbind(Acctab,AUCtab,TPRtab,FPRtab)


fic = paste("resupnl.txt",sep="")

dump(paste("ERRORES"),file=fic,append=T)
dump(paste("TP_MAT"),file=fic,append=T)
dump(paste("FP_MAT"),file=fic,append=T)
dump(paste("AUCs"),file=fic,append=T)
dump(paste("tabla_up_nl"),file=fic,append=T)


xtable(tabla_up_nl,caption = "Tabla Down No Lineal")













#######
#SMOTE#
#######

library(DMwR)
library(caret)
simul_data_nl<-data.frame(datay,datax) # 

errores=as.list(0)
nbiter=300

nsim=100 # numero de simulaciones para evaluar distribucion de errores

ERRORES<-matrix(NaN, ncol=6, nrow=nsim) # modelos x simulaciones
AUCs<-matrix(NaN, ncol=6, nrow=nsim) # Matrix con las Areas bajo la curva
TP_MAT<-matrix(NaN, ncol=6, nrow=nsim) # Matriz con los true positive calculados por TP_fun
FP_MAT<-matrix(NaN, ncol=6, nrow=nsim) # Matriz con los true positive calculados por FP_fun

CUTOFF=0.5

for(k in 1:nsim){
  
  print(k)
  n=nrow(simul_data_nl)
  s=sample(n,n/3)
  dlearn=simul_data_nl[-s,]
  
 
  dlearn=SMOTE(datay~., data = dlearn, k=3)
  colnames(dlearn)[1]<-"datay"
  dtest=simul_data_nl[s,]
  dtest=dtest[complete.cases(dtest),]
  yobstst=dtest[ ,1]
  yobstst=as.numeric(yobstst)
  
  #DISCRIMINANT ANALYSIS
  lda_error<-tryCatch(lda(datay~.,data=dlearn),error=function(e)e,finally="ERROR") # si da error, la prediccion es 0
  
  if(any(class(lda_error)=="error")){evdis=rep(0, nrow(dtest)) } else {
    dis=lda(datay~.,data=dlearn); 
    pr.dis<-as.numeric(predict(dis,newdata=dtest,type="prob")$posterior[,"a"])
    evdis=numeric()#as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
    evdis[pr.dis > CUTOFF]<- 1
    evdis[pr.dis <= CUTOFF]<- 2 # Acomoda para que las clases sean clasificadas como 1 o 2 segun el cutoff
    #evdis=as.numeric(predict(dis,newdata=dtest,type="class")$class) # Esto lo uso si clasifico con Majority vote (CUTOFF==0.5)
  }# When error, assume error in all predictions. Caused by error in lda because no variance
  
  #LR
  glm1<-stepAIC(glm(datay~., data=dlearn,family=binomial(link="logit")),trace=0)
  pr.glm<-predict(glm1, newdata=dtest[,-1],type="response")
  evglm<-numeric()
  evglm[pr.glm >  CUTOFF]<-2
  evglm[pr.glm <= CUTOFF]<-1
  
  
  #SVM
  svmaa=tune(svm,train.x=dlearn[,-1],train.y=dlearn[,1],data=dlearn[,-1])
  svma=svm(dlearn[,1]~.,data=dlearn[,-1],cost=svmaa$best.model$cost,gamma=svmaa$best.model$gamma,probability=TRUE)
  pr.svm<- as.numeric(attributes(predict(svma,newdata=dtest,probability=TRUE))$probabilities[,"a"])# Probability for class 1
  evsvm<-numeric()
  evsvm[pr.svm > CUTOFF]<-1
  evsvm[pr.svm <= CUTOFF]<-2
  
  #Non lineal
  #CART
  arbol=cart(dlearn,dtest,st=F,ms=5,cpopt=0.000001)
  pr.cart<-arbol$probTest
  ecart<-numeric()
  ecart[pr.cart >  CUTOFF]<-1
  ecart[pr.cart <= CUTOFF]<-2
  
  #SAMME
  samme=lastsamme(dlearn,dtest,nbiter=300,st=F,ms=5,cpopt=0.00001,cutoff=CUTOFF) 
  evsamme=samme$prevtst
  
  #RANDOM FOREST SIMPLE
  rf=randomForest(y=dlearn[,1],x=dlearn[,-1],data=dlearn,ntree=1000)
  pr.rf= as.numeric(predict(rf,newdata=dtest,type="prob")[,"a"])
  evrf<-numeric()
  evrf[pr.rf >0.5]<-1
  evrf[pr.rf <= 0.5]<-2 # Aqui no puse el CUTOFF por que sino el metodo es equivalente a rf_cutoff que esta abajo (sirve como una especie de control)
  
  
  
  ### CALCULOS DE CALIDAD DE AJUSTE
  
  errores=cbind(LDA=sum(evdis!=yobstst)/nrow(dtest),
                SVM=sum(evsvm!=yobstst)/nrow(dtest), 
                LR=sum(evglm!=yobstst)/nrow(dtest), 
                CART=sum(ecart!=yobstst)/nrow(dtest),
                SAMME=sum(evsamme!=yobstst)/nrow(dtest),
                RF=sum(evrf!=yobstst)/nrow(dtest))#
  round(errores,2)
  
  ERRORES[k,]<-errores
  
  # AUC
  auc=c(auc(as.numeric(yobstst),evdis),
        auc(as.numeric(yobstst),evsvm),
        auc(as.numeric(yobstst),evglm),
        auc(as.numeric(yobstst),evsamme),
        auc(as.numeric(yobstst),ecart),
        auc(as.numeric(yobstst),evrf))
  AUCs[k,]<-auc
  
  # TRUE POSITIVE
  truePositive= cbind(LDA=TP_fun(yobstst,evdis)$TPR, SVM=TP_fun(yobstst,evsvm)$TPR, 
                      LR=TP_fun(yobstst,evglm)$TPR,CART=TP_fun(yobstst,ecart)$TPR, 
                      SAMME=TP_fun(yobstst,evsamme)$TPR,
                      RF=TP_fun(yobstst,evrf)$TPR)
  TP_MAT[k,]<-truePositive
  
  falsePositive= cbind(LDA=TP_fun(yobstst,evdis)$FPR, SVM=TP_fun(yobstst,evsvm)$FPR, LR=TP_fun(yobstst,evglm)$FPR,
                       CART=TP_fun(yobstst,ecart)$FPR, 
                       SAMME=TP_fun(yobstst,evsamme)$FPR, 
                       RF=TP_fun(yobstst,evrf)$FPR)
  FP_MAT[k,]<-falsePositive
  
  
}# END LOOP      


ModNames<-cbind("LDA", "SVM", "LR","CART", "SAMME", "RF")
#ModNames<-cbind("LDA", "SVM", "LR","CART", "RF")



MeanAcc<-apply(1-ERRORES, 2, mean)
MeanAcc=round(MeanAcc,4)
SDAcc<-round(apply(1-ERRORES, 2, sd),2)
Acctab<-t(data.frame(paste(MeanAcc, SDAcc, sep="Ypm")))
colnames(Acctab)=ModNames
rownames(Acctab)=c("ACC")

MeanTP_MAT<-apply(TP_MAT, 2, mean)
MeanTP_MAT=round(MeanTP_MAT,4)
SDTP_MAT<-round(apply(TP_MAT, 2, sd),2)
TPRtab<-t(data.frame(paste(MeanTP_MAT, SDTP_MAT, sep="Ypm")))
colnames(TPRtab)=ModNames
rownames(TPRtab)=c("TPR")

MeanFP_MAT<-apply(FP_MAT, 2, mean)
MeanFP_MAT=round(MeanFP_MAT,4)
SDFP_MAT<-round(apply(FP_MAT, 2, sd),2)
FPRtab<-t(data.frame(paste(MeanFP_MAT, SDFP_MAT, sep="Ypm")))
colnames(FPRtab)=ModNames
rownames(FPRtab)=c("FPR")


MeanAUC<-apply(AUCs, 2, mean)
MeanAUC=round(MeanAUC,4)
SDAUC<-round(apply(AUCs, 2, sd),2)
AUCtab<-t(data.frame(paste(MeanAUC, SDAUC, sep="Ypm")))
colnames(AUCtab)=ModNames
rownames(AUCtab)=c("AUC")

tabla_smote_nl=rbind(Acctab,AUCtab,TPRtab,FPRtab)


fic = paste("ressmotenl.txt",sep="")

dump(paste("ERRORES"),file=fic,append=T)
dump(paste("TP_MAT"),file=fic,append=T)
dump(paste("FP_MAT"),file=fic,append=T)
dump(paste("AUCs"),file=fic,append=T)
dump(paste("tabla_smote_nl"),file=fic,append=T)

xtable(tabla_smote_nl,caption = "Tabla Smote No Lineal")


