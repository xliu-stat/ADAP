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

posBysite <- 
  cbind(apply(OUD_synthetic[OUD_synthetic$Site==1, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==2, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==3, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==4, -c(1,2)], 2, mean),
        apply(OUD_synthetic[OUD_synthetic$Site==5, -c(1,2)], 2, mean))

rownames(posBysite) <- colnames(OUD_synthetic)[-c(1:2)]
colnames(posBysite) <- c("site 1", "site 2", "site 3", "site 4", "site 5")
knitr::kable(round(posBysite*100, 2), booktabs = TRUE, caption = 'Site-specific Characteristics (%)')
# Table: Site-specific Characteristics (%)
# 
#   |                                          | site 1| site 2| site 3| site 4| site 5|
#   |:-----------------------------------------|------:|------:|------:|------:|------:|
#   |alcohol_related_disorders                 |   1.77|   2.43|   2.67|   2.73|   2.23|
#   |depression                                |   7.97|   8.60|  11.97|  11.50|  10.93|
#   |anxiety                                   |  11.03|  13.10|  14.80|  14.30|  10.37|
#   |sleep_disorder                            |   4.47|   3.17|   7.23|   4.37|   6.17|
#   |rheumatoid_arthritis                      |   2.10|   1.47|   2.17|   1.47|   1.77|
#   |pain                                      |  14.17|  22.43|  11.50|  16.97|  13.60|
#   |cannabis_related_disorder                 |   1.20|   3.27|   2.30|   1.93|   1.83|
#   |sedative_related_disorder                 |   0.23|   0.50|   0.27|   0.07|   0.40|
#   |cocaine_related_disorder                  |   0.60|   2.90|   1.70|   1.17|   2.80|
#   |nicotine_related_disorder                 |  11.27|  17.77|  15.10|  22.13|  11.83|
#   |other_psychoactive_disorder               |   2.33|   1.90|   1.53|   2.47|   2.43|
#   |CCI_Myocardial_infarction                 |   2.70|   3.43|   3.00|   1.83|   2.33|
#   |CCI_Congestive_heart_failure              |   4.30|   3.73|   5.20|   4.87|   4.53|
#   |CCI_Peripheral_vascular_disease           |   3.77|   3.23|   5.13|   4.60|   5.47|
#   |CCI_Cerebrovascular_disease               |   2.90|   3.00|   4.67|   4.33|   4.70|
#   |CCI_Dementia                              |   0.97|   0.63|   1.00|   0.63|   1.17|
#   |CCI_Chronic_pulmonary_disease             |  14.97|  16.07|  16.17|  16.73|  13.53|
#   |CCI_Rheumatic_disease                     |   3.20|   2.07|   2.87|   2.27|   3.00|
#   |CCI_Peptic_ulcer_disease                  |   1.47|   1.00|   0.67|   0.97|   0.73|
#   |CCI_Mild_liver_disease                    |   4.57|   4.43|   2.90|   6.17|   5.37|
#   |CCI_Diabetes_without_chronic_complication |  15.27|  12.37|  13.87|  13.60|  15.83|
#   |CCI_Diabetes_with_chronic_complication    |   3.20|   2.43|   2.87|   2.97|   5.27|
#   |CCI_Hemiplegia_or_paraplegia              |   0.73|   0.97|   1.23|   1.63|   2.73|
#   |CCI_Renal_disease                         |   4.73|   3.73|   4.40|   3.30|   4.83|
#   |CCI_Any_malignancy                        |   0.63|   1.40|   0.40|   1.27|   0.13|
#   |CCI_Moderate_or_severe_liver_disease      |   0.63|   0.53|   0.07|   0.73|   1.07|
#   |CCI_AIDS_HIV                              |   0.30|   0.67|   0.83|   0.77|   3.00|
#   |insomia                                   |   0.50|   0.50|   1.37|   1.60|   1.57|
#   |sleep_apnea                               |   4.23|   2.90|   6.43|   3.93|   3.17|
#   |bmi_1                                     |   3.17|   3.33|   4.13|   4.10|   2.60|
#   |bmi_2                                     |  29.60|  26.23|  27.90|  29.27|  27.97|
#   |bmi_3                                     |  41.83|  45.53|  42.53|  39.37|  42.13|
#   |smoke_1                                   |   4.73|   9.93|  24.17|  29.33|   0.17|
#   |smoke_2                                   |  88.90|  89.30|  52.97|  41.63|  99.53|
#   |race_1                                    |  52.73|  50.17|  63.17|  66.77|  13.47|
#   |race_2                                    |   1.07|   1.50|   4.60|   1.00|   4.20|
#   |insu_1                                    |  24.20|  23.53|  34.10|  43.57|  45.30|
#   |insu_2                                    |  58.93|   1.90|   1.93|  14.10|  21.03|
#   |age_1                                     |   3.17|   2.80|   3.77|   6.30|   2.30|
#   |age_2                                     |  39.03|  50.83|  32.67|  32.97|  25.83|
#   |age_3                                     |  41.47|  35.10|  42.67|  47.63|  53.97|
#   |gender_1                                  |  64.10|  71.17|  63.87|  59.53|  58.03|
```

Next, let us fit the model:
```{r}
site <- as.numeric(OUD_synthetic$Site)
Yall <- as.numeric(OUD_synthetic$OUD_status)
Xall <- as.matrix(OUD_synthetic[, -c(1,2)])

set.seed(100)
library(glmnet)
source("functions.r")
fit.res <- compare.methods(Xall, Yall, site, local_site = c(1,2), norder = NULL) 
res_mat <- cbind(fit.res$estimation_local, fit.res$estimation_ave, 
                 fit.res$estimation_odal1_5cv, fit.res$estimation_odal2_5cv,
                 fit.res$estimation_pooled)
colnames(res_mat) <- c("Local", "Average", "ADAP1", "ADAP2", "Pooled")
# display fitting results
knitr::kable(round(res_mat, 2), booktabs = TRUE, caption = 'Fitting Results')
# Table: Fitting Results
# 
#   |                                          | Local| Average| ADAP1| ADAP2| Pooled|
#   |:-----------------------------------------|-----:|-------:|-----:|-----:|------:|
#   |(Intercept)                               | -3.12|   -2.53| -3.12| -2.84|  -2.96|
#   |alcohol_related_disorders                 |  0.05|   -0.11| -0.05| -0.05|  -0.12|
#   |depression                                |  0.00|    0.03|  0.07|  0.07|   0.08|
#   |anxiety                                   |  0.51|    0.41|  0.44|  0.45|   0.46|
#   |sleep_disorder                            |  0.00|   -0.20| -0.26| -0.24|  -0.27|
#   |rheumatoid_arthritis                      | -0.11|    0.06|  0.00|  0.00|   0.02|
#   |pain                                      |  0.31|    0.28|  0.30|  0.30|   0.31|
#   |cannabis_related_disorder                 |  0.46|    0.53|  0.60|  0.58|   0.66|
#   |sedative_related_disorder                 |  0.13|    0.56|  0.29|  0.41|   0.97|
#   |cocaine_related_disorder                  |  0.66|    0.62|  0.53|  0.61|   0.70|
#   |nicotine_related_disorder                 |  0.42|    0.37|  0.39|  0.40|   0.41|
#   |other_psychoactive_disorder               |  0.29|    0.71|  0.78|  0.76|   0.84|
#   |CCI_Myocardial_infarction                 | -0.14|   -0.03|  0.00| -0.01|  -0.08|
#   |CCI_Congestive_heart_failure              |  0.21|    0.23|  0.28|  0.29|   0.32|
#   |CCI_Peripheral_vascular_disease           |  0.00|   -0.17| -0.21| -0.20|  -0.24|
#   |CCI_Cerebrovascular_disease               | -0.14|   -0.18| -0.19| -0.19|  -0.24|
#   |CCI_Dementia                              | -1.77|   -1.31| -1.29| -1.26|  -1.69|
#   |CCI_Chronic_pulmonary_disease             |  0.01|    0.00|  0.00|  0.00|   0.01|
#   |CCI_Rheumatic_disease                     |  0.58|    0.22|  0.25|  0.25|   0.30|
#   |CCI_Peptic_ulcer_disease                  |  0.00|    0.28|  0.19|  0.11|   0.29|
#   |CCI_Mild_liver_disease                    |  0.62|    0.24|  0.21|  0.22|   0.25|
#   |CCI_Diabetes_without_chronic_complication |  0.00|    0.09|  0.11|  0.11|   0.12|
#   |CCI_Diabetes_with_chronic_complication    |  0.24|   -0.10|  0.00|  0.00|  -0.02|
#   |CCI_Hemiplegia_or_paraplegia              | -0.23|   -0.04|  0.00|  0.00|   0.08|
#   |CCI_Renal_disease                         |  0.44|    0.35|  0.40|  0.40|   0.44|
#   |CCI_Any_malignancy                        | -0.18|   -0.01| -0.02| -0.01|  -0.25|
#   |CCI_Moderate_or_severe_liver_disease      | -0.20|    0.15|  0.00|  0.00|   0.05|
#   |CCI_AIDS_HIV                              |  0.99|    0.09|  0.00|  0.00|  -0.03|
#   |insomia                                   |  0.88|   -0.11| -0.14| -0.10|  -0.28|
#   |sleep_apnea                               | -0.34|   -0.07| -0.03| -0.03|  -0.06|
#   |bmi_1                                     |  0.12|    0.18|  0.20|  0.19|   0.24|
#   |bmi_2                                     | -0.01|   -0.01| -0.03| -0.03|  -0.03|
#   |bmi_3                                     | -0.09|   -0.10| -0.14| -0.14|  -0.14|
#   |smoke_1                                   |  0.67|    0.34|  0.76|  0.56|   0.61|
#   |smoke_2                                   |  0.74|    0.40|  0.86|  0.64|   0.68|
#   |race_1                                    |  1.01|    0.94|  0.94|  0.96|   0.98|
#   |race_2                                    |  0.39|    0.03|  0.14|  0.09|   0.18|
#   |insu_1                                    |  1.14|    1.09|  1.22|  1.14|   1.17|
#   |insu_2                                    |  0.40|    0.36|  0.48|  0.37|   0.39|
#   |age_1                                     | -0.52|   -0.81| -0.95| -0.91|  -0.97|
#   |age_2                                     |  0.73|    0.66|  0.72|  0.73|   0.77|
#   |age_3                                     |  0.70|    0.68|  0.75|  0.76|   0.80|
#   |gender_1                                  | -0.25|   -0.28| -0.33| -0.33|  -0.32|


# approximation performance of each method to the pooled estimator
dif_res <- res_mat[ ,1:4] - fit.res$estimation_pooled%*%t(rep(1,4))
round(apply(dif_res^2, 2, sum),2)
# Local Average   ADAP1   ADAP2 
# 4.61    0.98    0.89    0.72
```

To visualize the results:
```{r}
rela.bias.local <- abs((fit.res$estimation_local - fit.res$estimation_pooled)/fit.res$estimation_pooled)
rela.bias.ave <- abs((fit.res$estimation_ave - fit.res$estimation_pooled)/fit.res$estimation_pooled)
rela.bias.odal1 <- abs((fit.res$estimation_odal1_5cv - fit.res$estimation_pooled)/fit.res$estimation_pooled)
rela.bias.odal2 <- abs((fit.res$estimation_odal2_5cv - fit.res$estimation_pooled)/fit.res$estimation_pooled)

res_mat_relative <- cbind(rela.bias.local, rela.bias.ave, rela.bias.odal1, rela.bias.odal2)
colnames(res_mat_relative) <- c("Local", "Average", "ADAP1", "ADAP2")

library(reshape2)
library(ggplot2)

# visualize the estimates
rec_all <- res_mat[-1,]
tseq <- 1:dim(rec_all)[1]
rec_all <- rec_all[order(rec_all[,5], decreasing = T),]
rec_all <- melt(rec_all)
rec_all$size <- rep(tseq, 5)
rec_all$Method <- rec_all$Var2


ggplot(rec_all, aes(x=size, y=value)) + 
  geom_line(aes(colour=Method), size =0.5) +
  geom_point(aes(shape=Method, colour=Method)) + 
  scale_x_continuous("Covariates", breaks = tseq[c(5, 10, 15, 20, 25, 30, 35, 40)]) + 
  scale_colour_manual(values=c(Local="steelblue1", Average="turquoise", ADAP1="slateblue1",
                               ADAP2="rosybrown1", Pooled="tomato")) +
  ylab("Log Odds Ratio") +
  theme_bw() +
  theme(axis.title = element_text(size=14,face="bold"), axis.text = element_text(size=10)) +
  geom_hline(yintercept=0, linetype="dashed", color = "gray") 


# visualize the relative estimation error (global estimator as the reference)
rec_all <- res_mat_relative[,-1]
tseq <- 1:dim(rec_all)[1]
rec_all0 <- melt(rec_all)
rec_all0$size <- rep(tseq, 3)
rec_all0$Method <- rec_all0$Var2



ggplot(rec_all0, aes(x=size, y=value)) + 
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
