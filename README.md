# ADAP
A Distributed Algorithm for fitting Penalized (ADAP) regression models 

## Outline

1. Description 
2. Opioid use disorder (OUD) analysis

## Description

Integrating data across institutions can improve learning efficiency. ADAP is proposed for integrating data efficiently while protecting privacy for fitting penalized regression models across multiple sites. ADAP utilizes patient-level data from a lead site and incorporates the first-order (ADAP1) and second-order gradients (ADAP2) of the objective function from collaborating sites to construct a surrogate objective function at the lead site, where model fitting is then completed with proper regularizations applied. 

The implementation of ADAP in a practical collaborative learning environment is assembled into the R package [PDA: Prevacy-preserving Distributed Algorithms](https://github.com/Penncil/pda) where multiple distributed learning algorithms are available to fulfill various analysis demands. Toy examples to illustrate the fitting process of ADAP are also available in [PDA](https://github.com/Penncil/pda). 

ADAP is used in a real-world application to study risk factors for opioid use disorder (OUD) using 15,000 patient data from the OneFlorida Clinical Research Consortium. This page is prepared for reproducing the OUD analysis with a synthetic data. Since the OneFlorida+ data is available only upon application, for the reader’s convenience, we provide an illustrative synthetic dataset to simulate the true OneFlorida+ data. Specifically, the covariates are drawn from the empirical distribution of the true covariates, and the estimated coefficients from the OUD analysis are used to generate the OUD status by using a sparse logistic regression model. With this synthetic data where the generated patient-level records from all sites are available, we directly apply the R function of implementing ADAP without installing the PDA package for convenience.    


## OUD analysis

The OneFlorida data repository integrates multiple data sources from its participating healthcare organizations and provides real-world data to support biomedical and clinical research. Here, data are generated to mimic EHRs extracted from the 5 participating sites for 15,000 patients who had chronic pain and an opioid prescription (including buprenorphine, codeine, fentanyl, hydromorphone, meperidine, methadone, morphine, oxycodone, tramadol, and hydrocodone) and no cancer or OUD diagnosis before their first opioid prescription. Among these patients who were exposed to an opioid, we define a case of OUD as having a first diagnosis of OUD after their first prescription and define a control as having no diagnosis of OUD during the entire time window. A list of risk factors was compiled from the literature and extracted from the database, including basic demographic features such as age, 
gender and race, and co-occurring diagnoses, e.g., depression and sleep disorder (see Supplementary Table 2 for all 42 covariates). A logistic lasso regression is then applied to locate truly influential risk factors. 

Below is a summary of the generated data that include 15,000 patient records over 42 covariates: 
```{r}
load("OUD_synthetic.rda")
dim(OUD_synthetic)
table(OUD_synthetic$OUD_status, OUD_synthetic$Site)

posBysite <- 
  cbind(apply(OUD_synthetic[OUD_synthetic$Site==1, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==2, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==3, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==4, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==5, -c(1,2)], 2, mean))

rownames(posBysite) <- colnames(OUD_synthetic)[-c(1:2)]
colnames(posBysite) <- c("site 1", "site 2", "site 3", "site 4", "site 5")
knitr::kable(round(posBysite*100, 2), booktabs = TRUE, caption = 'Site-specific Characteristics (%)')
```

Next, let us fit the model:
```{r}
site <- as.numeric(OUD_synthetic$Site)
Yall <- as.numeric(OUD_synthetic$OUD_status)
Xall <- as.matrix(OUD_synthetic[, -c(1,2)])

set.seed(100)
library(glmnet)
source("functions.r")
fit.res <- compare.methods(Xall, Yall, site, local_site = c(4,1), norder = NULL) 
res_mat <- cbind(fit.res$estimation_local, fit.res$estimation_ave, 
                 fit.res$estimation_odal1_5cv, fit.res$estimation_odal2_5cv,
                 fit.res$estimation_pooled)
colnames(res_mat) <- c("Local", "Average", "ADAP1", "ADAP2", "Pooled")
# display fitting results
knitr::kable(round(res_mat, 2), booktabs = TRUE, caption = 'Fitting Results')

# approximation performance of each method to the pooled estimator
dif_res <- res_mat[ ,1:4] - fit.res$estimation_pooled%*%t(rep(1,4))
round(apply(dif_res^2, 2, sum),2)
```

To visualize the results:
```{r}
rela.bias.local <- abs((fit.part1$estimation_local - fit.part1$estimation_pooled)/fit.part1$estimation_pooled)
rela.bias.ave <- abs((fit.part1$estimation_ave - fit.part1$estimation_pooled)/fit.part1$estimation_pooled)
rela.bias.odal1 <- abs((fit.part1$estimation_odal1_5cv - fit.part1$estimation_pooled)/fit.part1$estimation_pooled)
rela.bias.odal2 <- abs((fit.part1$estimation_odal2_5cv - fit.part1$estimation_pooled)/fit.part1$estimation_pooled)

summary(rela.bias.local[-which(fit.part1$estimation_pooled == 0)])

summary(rela.bias.ave[-which(fit.part1$estimation_pooled == 0)])
summary(rela.bias.odal1[-which(fit.part1$estimation_pooled == 0)])
summary(rela.bias.odal2[-which(fit.part1$estimation_pooled == 0)])

res_mat <- cbind(rela.bias.local, rela.bias.ave, rela.bias.odal1, rela.bias.odal2)
colnames(res_mat) <- c("Local", "Average", "ADAP1", "ADAP2")
boxplot(res_mat[-which(fit.part1$estimation_pooled == 0),]*(res_mat[-which(fit.part1$estimation_pooled == 0),]<2))


# all
tseq <- 1:40 #downsize3: 1:42
rec_all <- res_mat[-which(fit.part1$estimation_pooled == 0),]
rec_all0 <- melt(rec_all)
rec_all0$size <- rep(tseq, 4)
rec_all0$Method <- rec_all0$Var2

# no local
tseq <- 1:40 #downsize3: 1:42
rec_all <- res_mat[-which(fit.part1$estimation_pooled == 0),-1]
rec_all0 <- melt(rec_all)
rec_all0$size <- rep(tseq, 3)
rec_all0$Method <- rec_all0$Var2

# all + pooled
tseq <- 1:42
rec_all <- res_part1[-1,]
colnames(rec_all) <- c("Local", "Average", "ADAP1", "ADAP2", "Pooled")
rec_all <- rec_all[order(rec_all[,5], decreasing = T),]
rec_all <- melt(rec_all)
rec_all$size <- rep(tseq, 5)
rec_all$Method <- rec_all$Var2


plot_all <- ggplot(rec_all, aes(x=size, y=value)) + 
  geom_line(aes(colour=Method), size =0.5) +
  geom_point(aes(shape=Method, colour=Method)) + 
  scale_x_continuous("Covariates", breaks = tseq[c(5, 10, 15, 20, 25, 30, 35, 40)]) + 
  scale_colour_manual(values=c(Local="steelblue1", Average="turquoise", ADAP1="slateblue1",
                               ADAP2="rosybrown1", Pooled="tomato")) +
  ylab("Log Odds Ratio") +
  #scale_y_continuous("Absolute Relative Estimation Bias", breaks = c(0, 1, 5, 10, 15)) + 
  theme_bw() +
  theme(axis.title = element_text(size=14,face="bold"), axis.text = element_text(size=10)) +
  geom_hline(yintercept=0, linetype="dashed", color = "gray") #+ geom_hline(yintercept=1, linetype="dashed", color = "gray")  


plot_part <- ggplot(rec_all0, aes(x=size, y=value)) + 
  geom_line(aes(colour=Method), size =0.5) +
  geom_point(aes(shape=Method, colour=Method)) + 
  scale_y_continuous("Relative Estimation Bias", breaks = c(0, 0.2, 0.5, 1, 2, 3, 4)) + 
  scale_x_continuous("Covariates", breaks = tseq[c(5, 10, 15, 20, 25, 30, 35, 40)]) + 
  scale_colour_manual(values=c(Average="turquoise", ADAP1="slateblue1",
                               ADAP2="rosybrown1")) +
  theme_classic() + 
  theme(axis.title = element_text(size=14,face="bold"), axis.text = element_text(size=10)) +
  geom_hline(yintercept=0, linetype="dashed", color = "gray") + geom_hline(yintercept=0.2, linetype="dashed", color = "gray") +
  geom_hline(yintercept=0.5, linetype="dashed", color = "gray")
```



The random-splitting procedure to measure prediction preformance is:
```{r}
library(glmnet)
library(ROCR)
source("/home/xiaokang/functions.r")

n.rep <- 50
auc_all <- array(NA, dim = c(9, n.rep, 5))  
set.seed(100)
for (i in 1:9){
  for (j in 1:n.rep){
    train.valid.flag <- 0
    while (train.valid.flag < 1) {
      index_train <- NULL
      index_test <- NULL
      for (k in 1:5){
        sel0 <- sample(which(site == k & Yall == 0), i*200, replace = FALSE)
        sel1 <- sample(which(site == k & Yall == 1), i*100, replace = FALSE)
        index_test <- c(index_test, sel0, sel1)
      }
      index_train <- setdiff(1:length(site), index_test)
      
      site_train <- site[index_train]
      Yall_train <- Yall[index_train]
      Xall_train <- Xall[index_train,]
      
      posBysite <- matrix(NA, nrow = 42, ncol = 5)
      posBysite[,1] <- apply(OUD_analy[intersect(index_train, OUD_analy$site == "AVH"), -var.delete], 2, sum)
      posBysite[,2] <- apply(OUD_analy[intersect(index_train, OUD_analy$site == "ORL"), -var.delete], 2, sum)
      posBysite[,3] <- apply(OUD_analy[intersect(index_train, OUD_analy$site == "TMH"), -var.delete], 2, sum)
      posBysite[,4] <- apply(OUD_analy[intersect(index_train, OUD_analy$site == "UFH"), -var.delete], 2, sum)
      posBysite[,5] <- apply(OUD_analy[intersect(index_train, OUD_analy$site == "UMI"), -var.delete], 2, sum)
      train.valid.flag <- all(posBysite > 1)
      
      site_test <- site[index_test]
      Yall_test <- Yall[index_test]
      Xall_test <- Xall[index_test,]
    }
    
    tryCatch({
      fit.train <- compare.methods(as.matrix(OUD_analy[index_train, -var.delete]), Yall_train, site_train, local_site = c(4,1), norder = NULL) 
      
      pred.local <- as.matrix(cbind(1, OUD_analy[index_test, -var.delete]))%*%fit.train$estimation_local
      pred.ave <- as.matrix(cbind(1, OUD_analy[index_test, -var.delete]))%*%fit.train$estimation_ave
      pred.odal1 <- as.matrix(cbind(1, OUD_analy[index_test, -var.delete]))%*%fit.train$estimation_odal1_5cv
      pred.odal2 <- as.matrix(cbind(1, OUD_analy[index_test, -var.delete]))%*%fit.train$estimation_odal2_5cv
      pred.pooled <- as.matrix(cbind(1, OUD_analy[index_test, -var.delete]))%*%fit.train$estimation_pooled
      
      
      pred <- prediction(pred.local, Yall_test)
      auc.tmp <- performance(pred,"auc")
      auc_all[i, j, 1] <- as.numeric(auc.tmp@y.values)
      
      pred <- prediction(pred.ave, Yall_test)
      auc.tmp <- performance(pred,"auc")
      auc_all[i, j, 2] <- as.numeric(auc.tmp@y.values)
      
      pred <- prediction(pred.odal1, Yall_test)
      auc.tmp <- performance(pred,"auc")
      auc_all[i, j, 3] <- as.numeric(auc.tmp@y.values)
      
      pred <- prediction(pred.odal2, Yall_test)
      auc.tmp <- performance(pred,"auc")
      auc_all[i, j, 4] <- as.numeric(auc.tmp@y.values)
      
      pred <- prediction(pred.pooled, Yall_test)
      auc.tmp <- performance(pred,"auc")
      auc_all[i, j, 5] <- as.numeric(auc.tmp@y.values)
    }, error=function(e) NA)
  }
}

auc_mean <- matrix(NA, nrow = 9, ncol = 5)
for (i in 1:9){
  auc_mean[i,] <- apply(auc_all[i,,], 2, mean, na.rm=T)
}
colnames(auc_mean) <- c("Local", "Ave", "ADAP1", "ADAP2", "Global")

tseq <- 1:9
rec_all <- auc_mean
rec_all0 <- melt(rec_all)
rec_all0$size <- rep(tseq, 5)
rec_all0$Method <- rec_all0$Var2


plot_all <- ggplot(rec_all0, aes(x=size, y=value)) + 
  geom_line(aes(colour=Method), size =0.5) +
  geom_point(aes(shape=Method, colour=Method)) + 
  #scale_x_discrete("Number of Sites", breaks = c(5,10,20,30,40,50)) + 
  scale_x_continuous("Test Size Index", breaks=tseq) + 
  # scale_colour_manual(values=c(Local="#000066", Ave="#663399", ODALLA1="#339999",
  #                              ODALLA2="#CC0033", Global="#FF6600")) +
  scale_colour_manual(values=c(Local="grey60", Ave="grey40", ADAP1="#339999",
                               ADAP2="#CC0033", Global="#663399")) +
  ylab("AUC") +
  theme_bw() +
  theme(axis.title = element_text(size=14,face="bold"), axis.text = element_text(size=10))
```



