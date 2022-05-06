rm(list=ls())

# generate a random graph
set.seed(1)
x=rnorm(100)
y=runif(100)
g.mat=p.mat=matrix(nrow=100,ncol=100,data=0)
is_edge=prob=c()
u=v=x_u=x_v=y_u=y_v=c()
for(i in 1:99) for(j in (i+1):100){
  p.mat[i,j]=p.mat[j,i]= y[i]*y[j]*exp(-abs(x[i]-x[j]))
  g.mat[i,j]=g.mat[j,i]=(runif(1)<p.mat[i,j])
  prob = c(prob, p.mat[i,j])
  is_edge = c(is_edge, g.mat[i,j])
  u = c(u, i)
  v = c(v, j)
  x_u = c(x_u, x[i])
  x_v = c(x_v, x[j])
  y_u = c(y_u, y[i])
  y_v = c(y_v, y[j])
}


## no. of edges
sum(g.mat)/2

## convert to a dataframe
g.df = data.frame(u,v,x_u,x_v,y_u,y_v,prob,is_edge)
g.df$is_edge = as.factor(g.df$is_edge)

## 10 node pairs with highest probability
head(g.df[order(-g.df$prob),], 10)



# build a simple logistic regression model - is_edge~x_u+x_v+y_u+y_v
library(glm2)
g.lm = glm(is_edge~x_u+x_v+y_u+y_v, data=g.df, family=binomial(link='logit'))
summary(g.lm)


## predict probability using the model
g.df$pred = predict(g.lm, new_data=g.df, type='response')


## 10 node pairs with highest predicted probability
head(g.df[order(-g.df$pred),], 10)


## 10 node pairs with highest actual probability
head(g.df[order(-g.df$prob),], 10)



# evaluation metrics
AvgProb <- function(pred,topn=10){
  return (mean(pred[order(-pred)][1:topn]))
}
AWPS <- function(truth,pred,topn=10){
  a = sum(truth[order(-pred)][1:topn])
  b = sum(truth[order(-truth)][1:topn])
  return (a/b)
}
AWRS <- function(truth,pred,topn=10){
  a = sum(pred[order(-truth)][1:topn])
  b = sum(pred[order(-pred)][1:topn])
  return (a/b)
}


## metrics for model is_edge~x_u+x_v+y_u+y_v
AvgProb(g.df$pred)
AWPS(g.df$prob,g.df$pred)
AWRS(g.df$prob,g.df$pred)



# data visualization and improvement on logistic regression model
library(ggplot2)
library(gridExtra)
plot1 = ggplot(g.df) + aes(x=is_edge, y=prob, fill=is_edge) + geom_boxplot() + ylim(0,1)
plot2 = ggplot(g.df) + aes(x=is_edge, y=pred, fill=is_edge) + geom_boxplot() + ylim(0,1)
grid.arrange(plot1,plot2,ncol=2,nrow=1)


## analysis on transformations of x_u,x_v
plot3 = ggplot(g.df) + aes(x=is_edge, y=x_u) + geom_boxplot()
plot4 = ggplot(g.df) + aes(x=is_edge, y=x_v) + geom_boxplot()
plot5 = ggplot(g.df) + aes(x=is_edge, y=abs(x_u)) + geom_boxplot()
plot6 = ggplot(g.df) + aes(x=is_edge, y=abs(x_v)) + geom_boxplot()
plot7 = ggplot(g.df) + aes(x=is_edge, y=exp(x_u)) + geom_boxplot()
plot8 = ggplot(g.df) + aes(x=is_edge, y=exp(x_v)) + geom_boxplot()
plot9 = ggplot(g.df) + aes(x=is_edge, y=x_u*x_v) + geom_boxplot()
plot10 = ggplot(g.df) + aes(x=is_edge, y=abs(x_u-x_v)) + geom_boxplot()
grid.arrange(plot3,plot4,plot5,plot6,plot7,plot8,plot9,plot10,ncol=4,nrow=2)



# build an improved logistic regression model - is_edge~abs(x_u-x_v)+y_u+y_v
g.df$"abs(x_u-x_v)" = abs(g.df$x_u-g.df$x_v)
g.lm2 = glm(is_edge~abs(x_u-x_v)+y_u+y_v, data=g.df, family=binomial(link='logit'))
summary(g.lm2)


## predict probability using the improved model
g.df$pred2 = predict(g.lm2, new_data=g.df, type='response')


## 10 node pairs with highest predicted probability
head(g.df[order(-g.df$pred2),], 10)


## 10 node pairs with highest actual probability
head(g.df[order(-g.df$prob),], 10)


## metrics for model is_edge~abs(x_u-x_v)+y_u+y_v
AvgProb(g.df$pred2)
AWPS(g.df$prob,g.df$pred2)
AWRS(g.df$prob,g.df$pred2)



# build an ergm model
library(ergm)
g.s=network::as.network(as.matrix(g.mat),directed=FALSE)
g.df2 = g.df[c("u","v","prob","is_edge")]
models=avg_pred=awps=awrs=c()


## construct a function for model generation and evaluation
MPLE <- function(formula,pred_name){
  set.seed(1)
  g.mple=ergmMPLE(formula,output="fit")
  
  g.predict = predict(g.mple)
  g.df2 <<- merge(x=g.df2, y=g.predict, by.x=c("u","v"), by.y=c("tail","head"), all.x=TRUE)
  names(g.df2)[names(g.df2)=="p"] <<- pred_name
  
  models <<- c(models, as.character(paste(formula[2],formula[1],formula[3])))
  avg_pred <<- c(avg_pred, AvgProb(g.df2[[pred_name]]))
  awps <<- c(awps, AWPS(g.df2$prob,g.df2[[pred_name]]))
  awrs <<- c(awrs, AWRS(g.df2$prob,g.df2[[pred_name]]))
  
  return (summary(g.mple))
}


## formula = g.s~edges+triangles+kstar(2)+degree(1)
g.mple = MPLE(g.s~edges+triangles+kstar(2)+degree(1), "simple_ergm")
g.mple$coefs

### 10 node pairs with highest predicted probability
head(g.df2[order(-g.df2$simple_ergm),], 10)

### metrics for model g.s~edges+triangles+kstar(2)+degree(1)
AvgProb(g.df2$simple_ergm)
AWPS(g.df2$prob,g.df2$simple_ergm)
AWRS(g.df2$prob,g.df2$simple_ergm)


## formula = g.s~edges+gwesp(0.1,fixed=TRUE)+kstar(2)+degree(1)
g.mple2 = MPLE(g.s~edges+gwesp(0.1,fixed=TRUE)+kstar(2)+degree(1), "gwesp_0.1")
g.mple2$coefs


## formula = g.s~edges+triangle+gwdegree(0.1,fixed=TRUE)+degree(1)
g.mple3 = MPLE(g.s~edges+triangle+gwdegree(0.1,fixed=TRUE)+degree(1), "gwd_0.1")
g.mple3$coefs



# prototype of optimal gamma searching
set.seed(1)
gamma=seq(0,5,length=11)
aic=c()
for(g in gamma){
  gamma.ergm=ergmMPLE(g.s~edges+gwesp(g,fixed=TRUE)+kstar(2)+degree(1),output="fit")
  gamma.aic = summary(gamma.ergm)$aic[1]
  aic = c(aic, gamma.aic)
}


## plot aic vs. gamma
qplot(x=gamma,y=aic)


## write a recursive function to find optimal gamma given search range and tolerance
OptimumSearch <- function(model,start=0,end=10,tol=0.01,length=10){
  set.seed(1)
  gamma=seq(start,end,length=length+1)
  aic=c()
  for(g in gamma){
    if (tolower(model)=='gwesp') gamma.ergm=ergmMPLE(g.s~edges+gwesp(g,fixed=TRUE)+kstar(2)+degree(1),output="fit")
    else if (tolower(model)=='gwd') gamma.ergm=ergmMPLE(g.s~edges+triangle+gwdegree(g,fixed=TRUE)+degree(1),output="fit")
    else stop("Wrong model type.")
    gamma.aic = summary(gamma.ergm)$aic[1]
    aic = c(aic, gamma.aic)
  }
  
  if((end-start)/length*2 > tol) {
    new.start = start + (order(aic)[1]-2)*((end-start)/length)
    new.end = start + order(aic)[1]*((end-start)/length)
    return (OptimumSearch(model,new.start,new.end,tol,length))
  }
  else {
    return (start + (order(aic)[1]-1)*((end-start)/length))
  }
}


## get optimal gamma for gwesp and gwd models
gamma.gwesp = OptimumSearch('gwesp',0,5,0.001,20)
gamma.gwd = OptimumSearch('gwd',0,5,0.001,20)


## formula = g.s~edges+gwesp(gamma.gwesp,fixed=TRUE)
g.mple4 = MPLE(g.s~edges+gwesp(gamma.gwesp,fixed=TRUE)+kstar(2)+degree(1), "gwesp_opt")
g.mple4$coefs


## formula = g.s~edges+gwdegree(gamma.gwd,fixed=TRUE)
g.mple5 = MPLE(g.s~edges+triangle+gwdegree(gamma.gwd,fixed=TRUE)+degree(1), "gwd_opt")
g.mple5$coefs



# get ergm metrics summary
ergm.metrics = data.frame(formula=as.character(models),avg_pred,awps,awrs)
ergm.metrics


## 10 node pairs with highest actual probability
head(g.df2[order(-g.df2$prob),], 10)



# predict without model fitting
ps=rwms=awps2=awrs2=c()


## revise metrics for prediction score evaluation
GetAvgWeight <- function(rmin,rmax){
  w = matrix(nrow=100,ncol=100,data=0)
  for(i in 1:99) for(j in (i+1):100) {
    if (rmin[i,j]>0){
      sum = 0
      for(k in rmin[i,j]:rmax[i,j]) sum = sum + 1/k
      w[i,j] = sum
      }
  }
  return (w)
}
RWMS <- function(pred,topn=10){
  rank.min = rank(-pred, ties.method="min")
  rank.max = rank(-pred, ties.method="max")
  select = rank.min<=topn
  rank.min = select*rank.min
  rank.max = select*rank.max
  rank.min = matrix(nrow=100,ncol=100,data=rank.min)
  rank.max = matrix(nrow=100,ncol=100,data=rank.max)
  weight = GetAvgWeight(rank.min, rank.max)
  a = sum(weight*g.mat)
  b = sum(weight)
  
  return (a/b)
}
AWPS2 <- function(pred,topn=10){
  select.thre = sort(pred,decreasing = TRUE)[topn]
  tie.count = sum(pred==select.thre)
  select=(pred>select.thre)+(pred==select.thre)*(topn-sum(pred>select.thre))/tie.count
  if (sum(select) != topn) stop("Wrong weightage assigned.")
  
  a = sum(select*p.mat)
  b = sum(p.mat[order(-p.mat)][1:topn])
  return (a/b)
}
AWRS2 <- function(pred,topn=10){
  a = sum(pred[order(-p.mat)][1:topn])
  b = sum(pred[order(-pred)][1:topn])
  return (a/b)
}


## common neighbours
a2=g.mat%*%g.mat

### reduce redundancy due to symmetry
for(i in 1:100) for(j in 1:i) a2[i,j]=0

### prediction evaluation
ps = c(ps, 'A^2')
rwms = c(rwms, RWMS(a2,10))
awps2 = c(awps2, AWPS2(a2,10))
awrs2 = c(awrs2, AWRS2(a2,10))


## k_u+k_v
k=g.mat%*%g.mat
for(i in 1:99) for(j in (i+1):100) k[i,j]=k[i,i]+k[j,j]
for(i in 1:100) for(j in 1:i) k[i,j]=0

### prediction evaluation
ps = c(ps, 'k_u+k_v')
rwms = c(rwms, RWMS(k,10))
awps2 = c(awps2, AWPS2(k,10))
awrs2 = c(awrs2, AWRS2(k,10))


## Jaccard measure
ja=g.mat%*%g.mat
for(i in 1:99) for(j in (i+1):100) ja[i,j]=ja[i,j]/(ja[i,i]+ja[j,j])
for(i in 1:100) for(j in 1:i) ja[i,j]=0
ja[is.na(ja)] = 0

### prediction evaluation
ps = c(ps, 'Jaccard')
rwms = c(rwms, RWMS(ja,10))
awps2 = c(awps2, AWPS2(ja,10))
awrs2 = c(awrs2, AWRS2(ja,10))


## k_u*k_v (preferential attachment)
pa=g.mat%*%g.mat
for(i in 1:99) for(j in (i+1):100) pa[i,j]=pa[i,i]*pa[j,j]
for(i in 1:100) for(j in 1:i) pa[i,j]=0

### prediction evaluation
ps = c(ps, 'k_u*k_v')
rwms = c(rwms, RWMS(pa,10))
awps2 = c(awps2, AWPS2(pa,10))
awrs2 = c(awrs2, AWRS2(pa,10))


## matrix exponential - alpha=0.1
alpha = 0.1

### use first 100 terms to approximate infinite sum
library(matrixcalc)
me_0.1 = matrix(0, nrow=100, ncol=100)
for(i in 0:100) me_0.1 = me_0.1 + alpha^i/factorial(i)*matrix.power(g.mat,i)
for(i in 1:100) for(j in 1:i) me_0.1[i,j]=0

### prediction evaluation
ps = c(ps, 'mat.exp_0.1')
rwms = c(rwms, RWMS(me_0.1,10))
awps2 = c(awps2, AWPS2(me_0.1,10))
awrs2 = c(awrs2, AWRS2(me_0.1,10))


## matrix exponential - alpha=0.5
alpha = 0.5

### use first 100 terms to approximate infinite sum
me_0.5 = matrix(0, nrow=100, ncol=100)
for(i in 0:100) me_0.5 = me_0.5 + alpha^i/factorial(i)*matrix.power(g.mat,i)
for(i in 1:100) for(j in 1:i) me_0.5[i,j]=0

### prediction evaluation
ps = c(ps, 'mat.exp_0.5')
rwms = c(rwms, RWMS(me_0.5,10))
awps2 = c(awps2, AWPS2(me_0.5,10))
awrs2 = c(awrs2, AWRS2(me_0.5,10))


## matrix exponential - alpha=2.5
alpha = 2.5

### use first 100 terms to approximate infinite sum
me_2.5 = matrix(0, nrow=100, ncol=100)
for(i in 0:100) me_2.5 = me_2.5 + alpha^i/factorial(i)*matrix.power(g.mat,i)
for(i in 1:100) for(j in 1:i) me_2.5[i,j]=0

### prediction evaluation
ps = c(ps, 'mat.exp_2.5')
rwms = c(rwms, RWMS(me_2.5,10))
awps2 = c(awps2, AWPS2(me_2.5,10))
awrs2 = c(awrs2, AWRS2(me_2.5,10))


## von Neumann kernel - alpha=0.05/lambda
eigen.max.inv = 1/max(eigen(g.mat)$values)
alpha = 0.05*eigen.max.inv

vnk_0.05 = solve(diag(100) - alpha*g.mat)
for(i in 1:100) for(j in 1:i) vnk_0.05[i,j]=0

### prediction evaluation
ps = c(ps, 'vnk_0.05/max.eigen')
rwms = c(rwms, RWMS(vnk_0.05,10))
awps2 = c(awps2, AWPS2(vnk_0.05,10))
awrs2 = c(awrs2, AWRS2(vnk_0.05,10))


## von Neumann kernel - alpha=0.50/lambda
eigen.max.inv = 1/max(eigen(g.mat)$values)
alpha = 0.50*eigen.max.inv

vnk_0.5 = solve(diag(100) - alpha*g.mat)
for(i in 1:100) for(j in 1:i) vnk_0.5[i,j]=0

### prediction evaluation
ps = c(ps, 'vnk_0.50/max.eigen')
rwms = c(rwms, RWMS(vnk_0.5,10))
awps2 = c(awps2, AWPS2(vnk_0.5,10))
awrs2 = c(awrs2, AWRS2(vnk_0.5,10))


## von Neumann kernel - alpha=0.95/lambda
eigen.max.inv = 1/max(eigen(g.mat)$values)
alpha = 0.95*eigen.max.inv

vnk_0.95 = solve(diag(100) - alpha*g.mat)
for(i in 1:100) for(j in 1:i) vnk_0.95[i,j]=0

### prediction evaluation
ps = c(ps, 'vnk_0.95/max.eigen')
rwms = c(rwms, RWMS(vnk_0.95,10))
awps2 = c(awps2, AWPS2(vnk_0.95,10))
awrs2 = c(awrs2, AWRS2(vnk_0.95,10))


## summarize the metrics for different prediction score
ps.metrics = data.frame("prediction_score"=ps,rwms,"awps"=awps2,"awrs"=awrs2)
ps.metrics
