#############################################
#Datos SIMULADOS NO LINEALES Para umbalanced#
#############################################
#Tomado y modificado del supporting material del paper multiclass classigfication Bourel y Segura 2018


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



######################################################################################################################################################
## MODEL FITTING
######################################################################################################################################################

simul_data_nl<-data.frame(datay,datax) # ## REVISAR QUE ESTO ESTE CORRECTO

errores_nl=as.list(0)
nbiter=300

nsim=100 # numero de simulaciones para evaluar distribucion de errores

ERRORESnl<-matrix(NaN, ncol=9, nrow=nsim) # modelos x simulaciones
AUCsnl<-matrix(NaN, ncol=9, nrow=nsim) # Matrix con las Areas bajo la curva
TP_MATnl<-matrix(NaN, ncol=9, nrow=nsim) # Matriz con los true positive calculados por TP_fun
FP_MATnl<-matrix(NaN, ncol=9, nrow=nsim) # Matriz con los true positive calculados por FP_fun

CUTOFF<- 0.5 # If posterior probability is > CUTOFF then classifier will choose the class- typical majority vote ==0.5

## LOOP LINEAL
for(k in 1:nsim){
  
  print(k)
  n=nrow(simul_data_nl)
  s=ech.tst(simul_data_nl[,1],prop=1/3)
  dlearn=simul_data_nl[-s,]
  dlearn=dlearn[complete.cases(dlearn),]
  cl=dlearn[,1]
  dtest=simul_data_nl[s,]
  dtest=dtest[complete.cases(dtest),]
  yobstst=dtest[ ,1]
  yobstst=as.numeric(yobstst)
  
  #Lineal
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
  
  
  
  #Random Forest modificados para mejorar la prediccion
  # Ver https://stats.stackexchange.com/questions/168415/random-forest-in-r-using-unbalanced-data
  #RANDOM FOREST cutoff
  threshold=0.1
  rf_cutoff=randomForest(y=dlearn[,1],x=dlearn[,-1],data=dlearn,ntree=1000, cutoff=c(threshold, 1-threshold)) # Following the receipe from internet...
  evrfcf= as.numeric(predict(rf_cutoff,newdata=dtest,type="class")) # n
  
  #RANDOM FOREST Stratified sampling
  sampsize<-min(table(dlearn[,1])) # Add this because gives error because random selection of learn set
  if(sampsize>5){
    rf_stratsamp=randomForest(y=dlearn[,1],x=dlearn[,-1],data=dlearn,ntree=1000, sampsize=c(sampsize, sampsize)) # 
    pr.rf_stratsamp= as.numeric(predict(rf_stratsamp,newdata=dtest,type="class"))
  } else pr.rf_stratsamp=NaN
  
  pr.rf_ss= predict(rf_stratsamp,newdata=dtest,type="prob")[,"a"]
  erfst<-numeric()
  erfst[pr.rf_ss > CUTOFF]<-1
  erfst[pr.rf_ss <=CUTOFF]<-2
  #table(erfst,yobstst)
  
  #RANDOM FOREST con classwt
  rf_wt=randomForest(y=dlearn[,1],x=dlearn[,-1],data=dlearn,ntree=1000, classwt=c(10,0.001)) # 
  pr.rfwt= as.numeric(predict(rf_wt,newdata=dtest,type="prob")[,"a"])
  erfcw<-numeric()
  erfcw[pr.rfwt >CUTOFF]<-1
  erfcw[pr.rfwt <= CUTOFF]<-2
  
  ### CALCULOS DE CALIDAD DE AJUSTE
  
  errores_nl=cbind(LDA=sum(evdis!=yobstst)/nrow(dtest),
                  SVM=sum(evsvm!=yobstst)/nrow(dtest), 
                  LR=sum(evglm!=yobstst)/nrow(dtest), 
                  CART=sum(ecart !=yobstst)/nrow(dtest), 
                  SAMME=sum(evsamme!=yobstst)/nrow(dtest),
                  RF=sum(evrf !=yobstst)/nrow(dtest), 
                  RFcutoff=sum(evrfcf!=yobstst)/nrow(dtest), 
                  RF_strat=sum(erfst!=yobstst)/nrow(dtest), 
                  RF_classwt=sum(erfcw!=yobstst)/nrow(dtest))#
  round(errores_nl,2)
  
  ERRORESnl[k,]<-errores_nl
  
  # AUC
  auc_nl=c(auc(as.numeric(yobstst),evdis),
          auc(as.numeric(yobstst),evsvm),
          auc(as.numeric(yobstst),evglm),
          auc(as.numeric(yobstst),evsamme),
          auc(as.numeric(yobstst),ecart),
          auc(as.numeric(yobstst),evrf),
          auc(as.numeric(yobstst),evrfcf),
          auc(as.numeric(yobstst),erfst),
          auc(as.numeric(yobstst),erfcw))
  AUCsnl[k,]<-auc_nl
  
  # TRUE POSITIVE RATE
  truePositive_nl= cbind(LDA=TP_fun(yobstst,evdis)$TPR, SVM=TP_fun(yobstst,evsvm)$TPR, LR=TP_fun(yobstst,evglm)$TPR,
                        CART=TP_fun(yobstst,ecart)$TPR, SAMME=TP_fun(yobstst,evsamme)$TPR, 
                        RF=TP_fun(yobstst,evrf)$TPR,RFcutoff=TP_fun(yobstst,evrfcf)$TPR, 
                        RF_strat=TP_fun(yobstst,erfst)$TPR, RF_classwt=TP_fun(yobstst,erfcw)$TPR)
  TP_MATnl[k,]<-truePositive_nl
  
  falsePositive_nl= cbind(LDA=TP_fun(yobstst,evdis)$FPR, SVM=TP_fun(yobstst,evsvm)$FPR, LR=TP_fun(yobstst,evglm)$FPR,
                         CART=TP_fun(yobstst,ecart)$FPR, SAMME=TP_fun(yobstst,evsamme)$FPR, 
                         RF=TP_fun(yobstst,evrf)$FPR,RFcutoff=TP_fun(yobstst,evrfcf)$FPR, 
                         RF_strat=TP_fun(yobstst,erfst)$FPR, RF_classwt=TP_fun(yobstst,erfcw)$FPR)
  FP_MATnl[k,]<-falsePositive_nl
  
  
  
}# END LOOP      


ModNames<-cbind("LDA", "SVM", "LR","CART", "SAMME", "RF","RFcutoff", "RF_strat", "RF_classwt")

MeanAcc<-apply(1-ERRORESnl, 2, mean)
MeanAcc=round(MeanAcc,4)
SDAcc<-round(apply(1-ERRORESnl, 2, sd),2)
Acctab<-t(data.frame(paste(MeanAcc, SDAcc, sep="Ypm")))
colnames(Acctab)=ModNames
rownames(Acctab)=c("ACC")

MeanTP_MAT<-apply(TP_MATnl, 2, mean)
MeanTP_MAT=round(MeanTP_MAT,4)
SDTP_MAT<-round(apply(TP_MATnl, 2, sd),2)
TPRtab<-t(data.frame(paste(MeanTP_MAT, SDTP_MAT, sep="Ypm")))
colnames(TPRtab)=ModNames
rownames(TPRtab)=c("TPR")

MeanFP_MAT<-apply(FP_MATnl, 2, mean)
MeanFP_MAT=round(MeanFP_MAT,4)
SDFP_MAT<-round(apply(FP_MATnl, 2, sd),2)
FPRtab<-t(data.frame(paste(MeanFP_MAT, SDFP_MAT, sep="Ypm")))
colnames(FPRtab)=ModNames
rownames(FPRtab)=c("FPR")


MeanAUC<-apply(AUCsnl, 2, mean)
MeanAUC=round(MeanAUC,4)
SDAUC<-round(apply(AUCsnl, 2, sd),2)
AUCtab<-t(data.frame(paste(MeanAUC, SDAUC, sep="Ypm")))
colnames(AUCtab)=ModNames
rownames(AUCtab)=c("AUC")

tabla_no_lineal=rbind(Acctab,AUCtab,TPRtab,FPRtab)




fic = paste("res_no_lineal.txt",sep="")
dump(paste("ERRORESnl"),file=fic,append=T)
dump(paste("TP_MATnl"),file=fic,append=T)
dump(paste("FP_MATnl"),file=fic,append=T)
dump(paste("AUCsnl"),file=fic,append=T)
dump(paste("tabla_no_lineal"),file=fic,append=T)

xtable(tabla_no_lineal,caption = "Tabla No Lineal")






x11(height=7, width=8)
par(mfrow=c(4,1), mar=c(5,4,1,1))

boxplot(ERRORESnl, axes=FALSE, xlab="", ylab="Error rate", ylim=c(0,0.5)); axis(1, at=1:ncol(ERRORESnl), labels=ModNames) 
axis(2); box()
abline(h=0.05, lty=2)

boxplot(AUCsnl,axes=FALSE, xlab="", ylab="AUC" ,ylim=c(0.5,1)); axis(1, at=1:ncol(AUCsnl), labels=ModNames) 
axis(2); box()

boxplot(TP_MATnl,axes=FALSE, xlab="Models", ylab="TRUE POSITIVE", ylim=c(0,1) ); axis(1, at=1:ncol(TP_MATnl), labels=ModNames) 
axis(2); box()

boxplot(FP_MATnl,axes=FALSE, xlab="Models", ylab="FALSE POSITIVE", ylim=c(0,1) ); axis(1, at=1:ncol(FP_MATnl), labels=ModNames) 
axis(2); box()

dev.off()



