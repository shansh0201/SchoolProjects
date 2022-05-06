setwd("G:\\Master\\DSA5101 Introduction to Big Data for Industry\\Assignment_Exercise\\Stats Project")

data = read.csv("project_residential_price_data_optional.csv")
data = data[,setdiff(names(data),c("V.30"))]

# descriptive analysis

summary(data$V.9)
qplot(V.9,data=data,geom="histogram",bins=20)

summary(data$V.8)
qplot(V.8,data=data,geom="histogram",bins=20)

cor(data$V.9,data$V.8)
ggplot(data) + aes(x=V.8,y=V.9) + geom_point() + geom_smooth(method = "lm")

summary(data$V.5)
qplot(V.5,data=data,geom="histogram",bins=20)
cor(data$V.9,data$V.5)
ggplot(data) + aes(x=V.5,y=V.9) + geom_point() + geom_smooth(method = "lm")

qplot(as.factor(V.1),data=data,geom="bar")

qplot(as.factor(V.10),data=data,geom="bar")

qplot(as.factor(V.20),data=data,geom="bar")


# linear basis model

lm_all = lm(V.9~., data=data)

lm_filtered = lm(V.9~V.1+V.2+V.3+V.4+V.5+V.6+V.7+V.8+V.10+V.12+V.13+V.15+V.17+V.19+V.20+V.21+V.25+V.26+V.27, data=data)

lm_trial = lm(V.9~V.8+(V.22>3800)+(V.29>1500000)+(V.29>2000000), data=data)

pred = predict(lm_all, newdata=data[,x_col])
pred2 = predict(lm_filtered, newdata=data[,x_col])
pred3 = predict(lm_trial, newdata=data[,x_col])

data_with_pred = cbind(data,pred,pred2,pred3)

ggplot(data_with_pred) + geom_abline(slope=0)  +  
  geom_smooth(aes(y=pred-V.9,x=pred), se=F, color="blue") + 
  geom_smooth(aes(y=pred2-V.9,x=pred2), se=F, color="red") + 
  geom_smooth(aes(y=pred3-V.9,x=pred3), se=F, color="green")
