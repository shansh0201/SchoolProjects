data = read.csv("C:\\Users\\Lenovo\\Desktop\\project_residential_price_data_optional.csv")


library(ggplot2)
library(GGally) 

y_col = "V.9"
x_col = setdiff(names(data),c("V.9","V.30"))
data = data[,setdiff(names(data),c("V.30"))]
data1 <- data


###Chi-square test for categorial values:V.1 V.10 V.20 

ggplot(data)+aes(x=as.factor(V.10),fill=factor(V.1))+
  geom_bar()
ggplot(data)+aes(x=as.factor(V.1),fill=factor(V.20))+
  geom_bar()
ggplot(data)+aes(x=as.factor(V.10),fill=factor(V.20))+
  geom_bar()

mychisq <- function(x,y){
  a<- table(x,y)
  print(a)
  print(prop.table(a,2))
  S = chisq.test(x,y,correct = TRUE)
  print(S)
  ##print(S$expected)
}

mychisq(data$V.1,data$V.10)
mychisq(data$V.1,data$V.20)
mychisq(data$V.10,data$V.20) 



####t test -2group 

#V.1~V.9
#STEP1-BOXPLOT

#set a function
data$V.1 <- as.factor(data$V.1) # convert to factor


myboxplot <- function(a){
  ggplot(data = data,aes(x = V.1,
                         y = a,
                         fill = V.1,
                         color = V.1))+
    geom_boxplot(alpha=0.7)+
    labs(x = "Project locality",
         #y = "Actual Sales Prices",
         #title ='Sales price boxplot by Project locality'
    )+
    theme(plot.title = element_text(face="plain",size=15,hjust=0.5))+
    scale_fill_brewer(palette='Accent')
  
}

##BOXPLOT OF THE OUTPUT BY v1
ggplot(data = data,aes(x = V.1,
                       y = V.9,
                       fill = V.1,
                       color = V.1))+
  geom_boxplot(alpha=0.7)+
  labs(x = "Project locality",
       y = "Actual Sales Prices",
       title ='Sales price boxplot by Project locality')+
  theme(plot.title = element_text(face="plain",size=15,hjust=0.5))+
  scale_fill_brewer(palette='Accent')

#STEP2-ttest

t.test(V.9~V.1,alternative = "two.sided", data=data)
t.test(V.9~V.1,alternative = "greater", data=data)

##others
##2->similar as v.9
##3-> almost no difference
##else has some difference
myboxplot(data$V.5)#2
myboxplot(data$V.6)#2
myboxplot(data$V.8)#2
#myboxplot(data$V.14)#3
#myboxplot(data$V.16)#3
#myboxplot(data$V.18)#3
myboxplot(data$V.21)#3
#myboxplot(data$V.22)#3
#myboxplot(data$V.24)#3
#myboxplot(data$V.28)#3


####anova test ->2 group
data$V.10 <- as.factor(data$V.10) # convert to factor


##BOXPLOT OF THE OUTPUT BY v10
ggplot(data = data,aes(x = V.10,
                       y = V.9,
                       fill = V.10,
                       color = V.10))+
  geom_boxplot(alpha=0.7)+
  labs(x = "Type of residential building",
       y = "Actual Sales Prices",
       title ='Sales price boxplot by residential building type')+
  theme(plot.title = element_text(face="plain",size=15,hjust=0.5))+
  scale_fill_brewer(palette='Accent')



aov_V10<-aov(V.9~V.10,data=data)
summary(aov_V10)
#P<2e-16<0.05 reject->related

#pair-wise anova
install.packages("gplots")
library(gplots)
plotmeans(data$V.9 ~ data$V.10,xlab = 'Type of residential building',ylab = 'values')#,mean = 'an plot\nwith 95% CI')
# difference between groups
#method1
TukeyHSD(aov_V10)
#method2
install.packages("agricolae")
library(agricolae)
result_O <- HSD.test(aov_V10, "V.10", group = T)
print(result_O)
#method3
install.packages("multcomp")
install.packages("survival")
install.packages("TH.data")
install.packages("MASS")
library(survival)
library(TH.data)
library(MASS)
library(multcomp)
tukey <- glht(aov_V10, linfct = mcp(V.10 = "Tukey"))
summary(tukey)
tukey.cld <- cld(tukey)
opar <- par(mai=c(1,1,1.6,1))
plot(tukey.cld)
par(opar)




###### V.20


data$V.20 <- as.factor(data$V.20) # convert to factor


##BOXPLOT OF THE OUTPUT BY v10
ggplot(data = data,aes(x = V.20,
                       y = V.9,
                       fill = V.20,
                       color = V.20))+
  geom_boxplot(alpha=0.7)+
  labs(x = "The interest rate for loan in a time resolution",
       y = "Actual Sales Prices",
       title ='Sales price boxplot by interest rate for loan')+
  theme(plot.title = element_text(face="plain",size=15,hjust=0.5))+
  scale_fill_brewer(palette='Accent')



aov_V20<-aov(V.9~V.20,data=data)
summary(aov_V20)
##p<2e-16
#V.1~V20
tukey <- glht(aov_V20, linfct = mcp(V.20 = "Tukey"))
summary(tukey)
tukey.cld <- cld(tukey)
opar <- par(mai=c(1,1,1.6,1))
plot(tukey.cld)
par(opar)






#####backward variable selection
lm_all = lm(V.9~., data=data1)
summary(lm_all)
lm_all.step <- step(lm_all, direction = "backward")
#V.1 + V.2 + V.3 + V.4 + V.5 + V.6 + V.7 + V.8 + V.12 + 
#V.13 + V.15 + V.17 + V.19 + V.20 + V.21 + V.25 + V.26 + V.27 + V.10
summary(lm_all.step)
anova(lm_all,lm_all.step)