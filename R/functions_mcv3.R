###################################
#Funciones para articulo imbalance#
###################################

library(rpart)

########################
vif_calc<-function(Xmat){
  VIF<-numeric()
  for(i in 1:ncol(Xmat)){
    Xmat_Y<-Xmat[,i]
    dataMAT<-cbind(Xmat_Y, Xmat[,-i])
    R2<-summary(lm(Xmat_Y~.,data=dataMAT, na.action="na.exclude"))$r.squared
    
    VIF[i]<-1/(1-R2)
  }
  names(VIF)<-colnames(Xmat)
  print(VIF)
}

###########################
#Para estratificar muestra#
###########################
ech.tst = function(y,prop=1/3) {
  #echantillonnage avec stratification
  # renvoie positions des ?l?ments de l'?chantillon test
  # fonction pr?vue pour les variables y discr?tes	
  
  ll = split(1:length(y),y)
  res = NULL
  for(j in 1:length(ll))
    res = c(res,sample(ll[[j]],max(c(length(ll[[j]])*prop,1))))
  res
}


#########
#TPR/FPR#
#########

TP_fun<-function(obs,pred){ # 
  u<-sort(union(obs,pred)) # para que siempre sea 1,2 el orden- 1 es CONTAMINADO
  CM<-table(factor(pred,u),factor(obs,u)) # Confusion matrix- OBSERVADOS EN LAS COLUMNAS!!!!
  TPR<-CM[1,1]/ sum(CM[,1])  # True Positive Rate TPR= TP/(TP+TN)- Acc+ o Recall
  TNR<-CM[2,2]/ sum(CM[,2])  # True Negative Rate TNR= TN/(TN+FP)- Acc-
  PR<-CM[1,1]/sum(CM[1,])    # Presicion          PR= TP/(TP+FP)
  list(TPR=TPR,FPR=1-TNR,TNR=TNR,PR=PR)
}


#############
#Cart########
#############
cart=function(learn,test,st=F,ms=5,cpopt=0.000001){
  #	yobsbase = as.numeric(learn[,1])
  yobstst = as.numeric(test[,1])
  
  if(st)	{ar1=rpart(learn[,1]~.,data=learn[,-1],maxdepth=1)} else{
    ar1 <- rpart(learn[,1]~., data = learn[, -1],minsplit=ms,cp=cpopt)
    cp.opt = ar1$cptable[which.min(ar1$cptable[,"xerror"]),"CP"]
    ar1 = prune(ar1,cp=cp.opt)}
  prob=predict(ar1,newdata=test,type="prob")[,"a"]
  prevvc=as.numeric(predict(ar1,newdata=test,type="class"))
  ar1tfctst=sum(prevvc!=yobstst)/nrow(test)
  list(probTest=prob,errorcart=ar1tfctst) 
}




#########
#SAMME###
#########

"lastsamme" = function(dlearn, dtest, nbiter = 10,itermarge=NULL,verb=F,ms=5,cpopt=0.01,st=F,cutoff=0.5)
{
  ncl <- length(unique(dlearn[, 1]))
  n <- nrow(dlearn)
  n2 <- nrow(dtest)
  #cat("Sample sizes: ")
  #cat(c(dim(dlearn), dim(dtest),"\n"))
  #cat(c("Classes distribution in Learning sample : ",table(dlearn[,1]),"\n"))
  #cat(c("Classes distribution in Test sample : ",table(dtest[,1]),"\n"))
  tfcbase = tfctst = rep(NA,nbiter)
  proba <- rep(1, n)/n
  beta <- numeric(0)	#
  yobsbase = as.numeric(dlearn[,1])
  yobstst = as.numeric(dtest[,1])
  j = 1
  kk = 1
  nbfail = 0
  if(!is.null(itermarge)) {
    marginL = matrix(0,n,length(itermarge))
    marginT = matrix(0,n2,length(itermarge))	
  }
  auxbase = matrix(0,nrow=n,ncol=ncl)
  auxtst = matrix(0,nrow=n2,ncol=ncl)
  
  while(j<=nbiter)	{
    k <- 1
    while (k == 1) {
      bg=sample(1:n,prob=proba,replace=T)
      while(dim(table(dlearn[bg,1]))==1){bg=sample(1:n,prob=proba,replace=T)}
      
      if(st)	{ar1=rpart(dlearn[bg,1]~.,data=dlearn[bg,-1],maxdepth=1)} else{
        ar1 <- rpart(dlearn[bg,1]~., data = dlearn[bg, -1],minsplit=ms,cp=cpopt)
        cp.opt = ar1$cptable[which.min(ar1$cptable[,"xerror"]),"CP"]
        ar1 = prune(ar1,cp=cp.opt)}
      k <- length(ar1$frame$var)
    }
    
    
    
    prevctstproba=predict(ar1,newdata=dtest,type="prob")[,"a"]
    prevctst<-numeric()
    prevctst[prevctstproba >  CUTOFF]<-1
    prevctst[prevctstproba <= CUTOFF]<-2
    
    prevcbaseproba=predict(ar1,newdata=dlearn,type="prob")[,"a"]
    prevcbase<-numeric()
    prevcbase[prevcbaseproba >  CUTOFF]<-1
    prevcbase[prevcbaseproba <= CUTOFF]<-2
    
    # prevctst=as.numeric(predict(ar1,newdata=test,type="class"))
    #prevcbase=as.numeric(predict(ar1,newdata=learn,type="class"))
    # Update weights probabilites
    #
    ind = prevcbase != yobsbase
    eps = sum(proba[ind])
    #	if(verb) cat(c(sum(ind), "//", signif(eps,3), ">> "))
    betaval = ((1 - eps)/eps) *(ncl-1)
    beta[j] = betaval
    maj = ((betaval)^(ind)) * proba
    maj = maj/sum(maj)
    proba = maj
    if(j > 1) {
      beta0 = log(betaval)
      # I do not normalize the betas as suggested sometimes in the litterature
      auxbase[cbind(1:n,prevcbase)] = auxbase[cbind(1:n,prevcbase)]+ beta0
      auxtst[cbind(1:n2,prevctst)] = auxtst[cbind(1:n2,prevctst)]+ beta0
      prevbase2 = max.col(auxbase)
      prevtst2 = max.col(auxtst)	
      # Missclassification error after j iterations 
      tfcbase[j] = mean(prevbase2 != yobsbase) 
      tfctst[j] = mean(prevtst2 != yobstst)
      if (!is.null(itermarge)) {
        if( j == itermarge[kk]) {
          indlin = ((yobsbase - 1) * n ) + 1:n
          aux = auxbase 
          mrg.yn = aux[indlin]
          aux[indlin] = -Inf
          marginL[,kk] = (mrg.yn - apply(aux,1,max)) / sum(log(beta))
          indlin = ((yobstst - 1) * n2 ) + 1:n2
          aux = auxtst 
          mrg.yn = aux[indlin]
          aux[indlin] = -Inf
          marginT[,kk] = (mrg.yn - apply(aux,1,max)) / sum(log(beta))
          kk=kk+1
        }	
      }			
    }
    j = j + 1
  }
  cat(paste("Number of failure to get better error rate then 1/2 :",nbfail,"\n") )
  if(!is.null(itermarge)) list(tfctst = tfctst, tfcbase = tfcbase, nbfail=nbfail,testmargins=marginT,learnmargins=marginL)
  else list(tfctst = tfctst, tfcbase = tfcbase, nbfail=nbfail,prevtst=prevtst2,prevbase=prevbase2)
}








