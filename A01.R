# Acknowledgment 
# Some of the code taken from script written by Prof Bharat Verma
# github.com/dearbharat
# Code/Analysis majorly authored by Me Starts on Line 810
##


## Line 10 - 810 - Data Cleaning @author Prof Bharat Verma
## Line 810  - End - EDA @author Gabriel Adeyemo

# Library
library(dplyr)
library(ggplot2)
library(reshape2)
library(Information)
library(ggcorrplot)

# Reading dataset
dem <- read.csv("demogs.csv")
credit <- read.csv("Credit_Bureau.csv")
glimpse(dem)

# Summary
summary(dem)
summary(credit)

str(dem)
str(credit)

# No of rows
nrow(dem)
# 71295

nrow(credit)
# 71295


#===================#
#   Color palette   #
#===================#

cp_2 <- c("#FEA47F", "#F97F51")
cp_3 <- c("#2A363B", "#E84A5F", "#FF847C")
cp_5 <- c("#2A363B", "#E84A5F", "#FF847C", "#FECEAB", "#99B898")
cp_8 <- c("#FEA47F", "#F97F51", "#B33771", "#3B3B98", "#58B19F", "#BDC581", "#2C3A47", "#82589F")

#############################
# Exploratory Data Analysis #
#############################

#=================================#
#   Checking for duplicate data   #
#=================================#

# Demographic Dataset
length(unique(dem$Application.ID))

# 71292

# Credit Dataset
length(unique(credit$Application.ID))

# 71292


dem %>%
  group_by(Application.ID) %>%
  filter(n() > 1)

credit %>%
  group_by(Application.ID) %>%
  filter(n() > 1)

# We could see that few data with same Application ID is for different persons.
# So removing all the duplicate rows

dem <- dem %>%
  group_by(Application.ID) %>%
  filter(n() == 1)

credit <- credit %>%
  group_by(Application.ID) %>%
  filter(n() == 1)

# Total 71289


# Merging the datasets
merged_data <- merge(dem, credit, by=c("Application.ID", "Performance.Tag"))


#===========================#
#   Changing Column Names   #
#===========================#
# 5 , 11:25
names(merged_data)[c(1:2, 5:6, 10:29)] <- c("Application_ID", "Performance_Tag", "Marital_Status", "No_Of_Dependents", "Type_Of_Residence", "Months_In_Current_Residence", "Months_In_Current_Company", "No_Of_90_DPD_6_months", "No_Of_60_DPD_6_months", "No_Of_30_DPD_6_months", "No_Of_90_DPD_12_months","No_Of_60_DPD_12_months","No_Of_30_DPD_12_months", "Avg_CC_Utilization_12_months", "Trades_6_months", "Trades_12_months", "PL_Trades_6_months", "PL_Trades_12_months", "Inquiries_6_months", "Inquiries_12_months", "Open_Home_Loan", "Outstanding_Balance", "Total_No_of_trades", "Open_Auto_Loan")

#=====================#
#   Performance Tag   #
#=====================#

# NA count
merged_data$Performance_Tag %>%
  is.na() %>%
  sum()
# 1425

table(merged_data$Performance_Tag)
# Percentage of Default
non_default_count <- as.numeric(table(merged_data$Performance_Tag)[2])
default_count <- as.numeric(table(merged_data$Performance_Tag)[1])

default_percentage <- default_count / (default_count+non_default_count)
default_percentage*100
# 95.78 %

#=========#
#   Age   #
#=========#

# Check for Age variable rows with NA values
merged_data$Age %>%
  is.na() %>% sum()
# 0

# Checking for outliers
merged_data$Age %>%
  quantile(seq(0,1, 0.01))

merged_data$Age %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

## Min age is -3
## Some ages are 0
## Capping minimum age to 18
## Since 18 is the minimum age to avail a credit card

merged_data$Age <- merged_data$Age %>%
  as.numeric()

merged_data[(which(merged_data$Age < 18)), ]$Age <- 18

## Creating age bins
merged_data$Age %>%
  summary()

# Different Bins
# 1) 16-20
# 2) 21-25
# 3) 26-30
# 4) 31-35
# 5) 36-40
# 6) 41-45
# 7) 46-50
# 8) 51-55
# 9) 56-60
# 10) 61-65

# Age Bins function
age_bin <- function(age=3){
  if(age > 17 && age < 21)
    return ("18-20")
  else if(age > 20 && age < 26)
    return ("21-25")
  else if(age > 25 && age < 31)
    return ("26-30")
  else if(age > 30 && age < 36)
    return ("31-35")
  else if(age > 35 && age < 41)
    return ("36-40")
  else if(age > 40 && age < 46)
    return ("41-45")
  else if(age > 45 && age < 51)
    return ("46-50")
  else if(age > 50 && age < 56)
    return ("51-55")
  else if(age > 55 && age < 61)
    return ("56-59")
  else if(age > 60 && age < 66)
    return ("61-65")
  
}

# Creating Age Bin field
merged_data$Age_Bin <-  merged_data$Age %>%
  sapply(age_bin) %>%
  as.factor()

#=============#
#   Gender    #
#=============#
# Summary for Gender
merged_data$Gender %>%
  summary()

# 2 NA's

# Converting NA for Gender variable to "M"
levels(merged_data$Gender)[1] <- "M"

#=====================#
#   Marital Status   #
#=====================#

# Summary for Marital status at time of application
merged_data$Marital_Status %>%
  summary()
# 6 NA's

# Converting NA for Marital status at time of application variable to "Married"
levels(merged_data$Marital_Status)[1] <- "Married"


#=====================#
#   No of Dependents  #
#=====================#

# Checking for NA values
merged_data$No_Of_Dependents %>%
  is.na() %>%
  sum()

# 3 NA's

merged_data$No_Of_Dependents[which(is.na(merged_data$No_Of_Dependents))] <- 3

merged_data$No_Of_Dependents %>%
  as.factor() %>%
  summary()

# Checking for outliers
merged_data$No_Of_Dependents %>%
  quantile(seq(0,1,0.01), na.rm = T)

merged_data$No_Of_Dependents %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

#Converting the variable into factor type
merged_data$No_Of_Dependents <- merged_data$No_Of_Dependents %>% as.factor()

#=============#
#   Income    #
#=============#

# checking for NA values
merged_data$Income %>%
  is.na() %>%
  sum()
# 0

# Checking for outliers
merged_data$Income %>%
  quantile(seq(0,1,0.01), na.rm = T)

merged_data$Income %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data$Income %>%
  as.factor() %>%
  levels()

# Converting Income less than 1 to 1.0
merged_data[(which(merged_data$Income < 1)), ] $Income <- 1.0

# Creating Income Bracket
# Income Bracket Function

income_bin <- function(income = 1){
  if(income >= 1 && income <=10)
    return ("1-10")
  else if(income >= 11 && income <=20)
    return ("11-20")
  else if(income >= 21 && income <=30)
    return ("21-30")
  else if(income >= 31 && income <=40)
    return ("31-40")
  else if(income >= 41 && income <=50)
    return ("41-50")
  else
    return ("51-60")
}

merged_data$Income_Bin <-  merged_data$Income %>%
  sapply(income_bin) %>%
  as.factor()


#===============#
#   Education   #
#===============#

# checking for NA values
merged_data$Education %>%
  is.na() %>%
  sum()

# 0

# Checking for blank rows
merged_data$Education %>%
  summary()

levels(merged_data$Education)[1] <- "Professional"


#=================#
#   Profession    #
#=================#

# checking for NA values
merged_data$Profession %>%
  is.na() %>%
  sum()

# 0

# Checking for blank rows
merged_data$Profession %>%
  summary()

levels(merged_data$Profession)[1] <- "SAL"

#=======================#
#   Type of residence   #
#=======================#
# checking for NA values
merged_data$Type_Of_Residence %>%
  is.na() %>%
  sum()

# 0

# Checking for blank rows
merged_data$Type_Of_Residence %>%
  summary()

levels(merged_data$Type_Of_Residence)[1] <- "Rented"

#===========================================#
#   Number of months in current residence   #
#===========================================#

# Checking for NA values
merged_data$Months_In_Current_Residence %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$Months_In_Current_Residence %>%
  quantile(seq(0,1,0.01), na.rm = T)

merged_data$Months_In_Current_Residence %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")


# Resident Years Bin Function
res_yrs_bin <- function(nom=0){
  noy = nom/12
  if(noy > 0 && noy < 1)
    return("< 1 yr")
  else if(noy >= 1 && noy < 2)
    return("1 yr")
  else if(noy >= 2 && noy < 3)
    return("2 yrs")
  else if(noy >= 3 && noy < 4)
    return("3 yrs")
  else if(noy >= 4 && noy < 5)
    return("4 yrs")
  else if(noy >= 5 && noy < 6)
    return("5 yrs")
  else if(noy >= 6 && noy < 7)
    return("6 yrs")
  else if(noy >= 7 && noy < 8)
    return("7 yrs")
  else if(noy >= 8 && noy < 9)
    return("8 yrs")
  else if(noy >= 9 && noy < 10)
    return("9 yrs")
  else
    return("> 10 yrs")
}

# Creating No of years in current residence variable
merged_data$Yrs_Curr_Res <- merged_data$Months_In_Current_Residence %>%
  sapply(res_yrs_bin) %>%
  as.factor()

# Plot of frequency of No of years in current residence variable
ggplot(merged_data, aes(x=Yrs_Curr_Res, y=..count../1000, fill=Yrs_Curr_Res)) +
  geom_bar() +
  labs(x="No of Years in residence", y="Frequency in 1000s", fill="No of Years in residence", title="Frequency of Years in residence") +
  theme_minimal() +
  theme(axis.text.x=element_text(angle=40, hjust=1))


#=========================================#
#   Number of months in current company   #
#=========================================#

# Checking for NA values
merged_data$Months_In_Current_Company %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$Months_In_Current_Company %>%
  quantile(seq(0,1,0.01), na.rm = T)


merged_data$Months_In_Current_Company %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

# Capping No of months in current company to 74
merged_data[(which(merged_data$Months_In_Current_Company > 74)),] $Months_In_Current_Company <- 74

#   Current Company Years Bin Function
comp_yrs_bin <- function(nom=0){
  noy = nom/12
  if(noy > 0 && noy < 1)
    return("< 1 yr")
  else if(noy >= 1 && noy < 2)
    return("1 yr")
  else if(noy >= 2 && noy < 3)
    return("2 yrs")
  else if(noy >= 3 && noy < 4)
    return("3 yrs")
  else if(noy >= 4 && noy < 5)
    return("4 yrs")
  else if(noy >= 5 && noy < 6)
    return("5 yrs")
  else
    return("> 6 yrs")
}

# Crating variable No of years in curr comp
merged_data$Yrs_Curr_Comp <- merged_data$Months_In_Current_Company %>%
  sapply(comp_yrs_bin) %>%
  as.factor()


#===================================================#
#   No of times 90 DPD or worse in last 6 months    #
#===================================================#

# Checking for NA values
merged_data$No_Of_90_DPD_6_months %>%
  is.na() %>%
  sum()

# 0

merged_data$No_Of_90_DPD_6_months %>%
  as.factor() %>%
  summary()


#===================================================#
#   No of times 60 DPD or worse in last 6 months    #
#===================================================#

# Checking for NA values
merged_data$No_Of_60_DPD_6_months %>%
  is.na() %>%
  sum()

# 0

merged_data$No_Of_60_DPD_6_months %>%
  as.factor() %>%
  summary()



#===================================================#
#   No of times 30 DPD or worse in last 6 months    #
#===================================================#

# Checking for NA values
merged_data$No_Of_30_DPD_6_months %>%
  is.na() %>%
  sum()

# 0

merged_data$No_Of_30_DPD_6_months %>%
  as.factor() %>%
  summary()



#===================================================#
#   No of times 90 DPD or worse in last 12 months   #
#===================================================#

# Checking for NA values
merged_data$No_Of_90_DPD_12_months %>%
  is.na() %>%
  sum()

# 0

merged_data$No_Of_90_DPD_12_months %>%
  as.factor() %>%
  summary()



#===================================================#
#   No of times 60 DPD or worse in last 12 months   #
#===================================================#

# Checking for NA values
merged_data$No_Of_60_DPD_12_months %>%
  is.na() %>%
  sum()

# 0

merged_data$No_Of_60_DPD_12_months %>%
  as.factor() %>%
  summary()



#===================================================#
#   No of times 30 DPD or worse in last 12 months   #
#===================================================#

# Checking for NA values
merged_data$No_Of_30_DPD_12_months %>%
  is.na() %>%
  sum()
# 0

merged_data$No_Of_30_DPD_12_months %>%
  as.factor() %>%
  summary()

#===================================#
#   Correlation of DPD Variables    #
#===================================#

DPD_data_6 <- merged_data[, c(13:15)]
DPD_data_12 <- merged_data[, c(16:18)]

cor_DPD_6 <- round(cor(DPD_data_6), 2)
cor_DPD_6
melted_cor_DPD_6 <- melt(cor_DPD_6)

cor_DPD_12 <- round(cor(DPD_data_12), 2)
melted_cor_DPD_12 <- melt(cor_DPD_12)
#=======================================================#
#   Average Credit Card utilisation in last 12 months   #
#=======================================================#

# Checking for NA values
merged_data$Avg_CC_Utilization_12_months %>%
  is.na() %>%
  sum()
# 1058

merged_data$Avg_CC_Utilization_12_months %>%
  summary()

# Replacing the NA value with the median
merged_data$Avg_CC_Utilization_12_months[which(is.na(merged_data$Avg_CC_Utilization_12_months))] <- 15


# Checking for outliers
merged_data$Avg_CC_Utilization_12_months %>%
  quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Avg_CC_Utilization_12_months %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$Avg_CC_Utilization_12_months > 103)),] $Avg_CC_Utilization_12_months <- 103


#==========================================#
#   No of trades opened in last 6 months   #
#==========================================#

# Checking for NA values
merged_data$Trades_6_months %>%
  is.na() %>%
  sum()

# 1

merged_data$Trades_6_months %>%
  summary()

# Replacing the NA value with the median
merged_data$Trades_6_months[which(is.na(merged_data$Trades_6_months))] <- 2

# Checking for outliers
merged_data$Trades_6_months %>% quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Trades_6_months %>% boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$Trades_6_months > 6)),] $Trades_6_months <- 6



#===========================================#
#   No of trades opened in last 12 months   #
#===========================================#

# Checking for NA values
merged_data$Trades_12_months %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$Trades_12_months %>% quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Trades_12_months %>% boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$Trades_12_months > 19)),] $Trades_12_months <- 19

#===================================#
#   Correlation of trades opened    #
#===================================#

trades_opened <- merged_data[, c(20, 21)]

cor_trades_opened <- round(cor(trades_opened), 2)
melted_cor_trades_opened <- melt(cor_trades_opened)

#==============================================#
#   No of PL trades opened in last 6 months    #
#==============================================#

# Checking for NA values
merged_data$PL_Trades_6_months  %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$PL_Trades_6_months  %>% quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$PL_Trades_6_months  %>% boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$PL_Trades_6_months  > 5)),] $PL_Trades_6_months  <- 5


#===============================================#
#   No of PL trades opened in last 12 months    #
#===============================================#

# Checking for NA values
merged_data$PL_Trades_12_months  %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$PL_Trades_12_months  %>% quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$PL_Trades_12_months  %>% boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$PL_Trades_12_months  > 10)),] $PL_Trades_12_months  <- 10

#===================================#
#   Correlation of PL trades opened    #
#===================================#

pl_trades_opened <- merged_data[, c(22, 23)]

cor_pl_trades_opened <- round(cor(pl_trades_opened), 2)
melted_cor_pl_trades_opened <- melt(cor_pl_trades_opened)


#===============================================================#
#   No if inquiries in last 6 months excluding home auto loan   #
#===============================================================#

# Checking for NA values
merged_data$Inquiries_6_months %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$Inquiries_6_months %>%
  quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Inquiries_6_months %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$Inquiries_6_months > 7)),] $Inquiries_6_months <- 7



#=================================================================#
#   No if inquiries in last 12 months excluding home auto loan    #
#=================================================================#

# Checking for NA values
merged_data$Inquiries_12_months %>%
  is.na() %>%
  sum()

# 0

# Checking for outliers
merged_data$Inquiries_12_months %>%
  quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Inquiries_12_months %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$Inquiries_12_months > 12)),] $Inquiries_12_months <- 12


#=================================#
#   Presence of open home loan    #
#=================================#

# Checking for NA values
merged_data$Open_Home_Loan %>%
  is.na() %>%
  sum()

# 272

merged_data$Open_Home_Loan %>%
  as.factor() %>%
  summary()

merged_data$Open_Home_Loan[which(is.na(merged_data$Open_Home_Loan))] <- 0

# Converting to factor type
merged_data$Open_Home_Loan <- merged_data$Open_Home_Loan %>%
  as.factor()

#=================================#
#   Presence of open auto loan    #
#=================================#

# Checking for NA values
merged_data$Open_Auto_Loan %>%
  is.na() %>%
  sum()

# 0

merged_data$Open_Auto_Loan %>%
  as.factor() %>%
  summary()

# Converting to factor type
merged_data$Open_Auto_Loan <- merged_data$Open_Auto_Loan %>%
  as.factor()

#=========================#
#   Outstanding Balance   #
#=========================#

# Checking for NA values
merged_data$Outstanding_Balance %>%
  is.na() %>%
  sum()
# 272

merged_data$Outstanding_Balance %>%
  summary()
# Median = 774985

merged_data$Outstanding_Balance[which(is.na(merged_data$Outstanding_Balance))] <- 774985


# Checking for outliers
merged_data$Outstanding_Balance %>%
  quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Outstanding_Balance %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")


#=========================#
#   Total no of trades    #
#=========================#

# Checking for NA values
merged_data$Total_No_of_trades %>%
  is.na() %>%
  sum()

# Checking for outliers
merged_data$Total_No_of_trades %>%
  quantile(seq(0, 1, 0.01), na.rm = T)

merged_data$Total_No_of_trades %>%
  boxplot(border = "#6fa058", outcol = "#ee853f")

merged_data[(which(merged_data$Total_No_of_trades > 20)),] $Total_No_of_trades <- 20



names(merged_data)
#=====================================#
# Remove Rest of Missing Data 
#=====================================#

merged_data <- na.omit(merged_data)
merged_data %>% is.na %>% sum()

#==================================#
#  Data Visual.
#==================================#

# Correlation Matrix

# Income vs Performance Tag
merged_data %>%
  filter(!is.na(Performance_Tag)) %>%
  ggplot(aes(x=Income_Bin, y=..count../1000, fill=as.factor(Performance_Tag))) +
  geom_bar(position = "dodge") +
  theme_minimal()+
  scale_fill_manual(values = cp_2) +
  labs(x="Income Buckets", y="Frequency in 1000s", fill="Performance Tag", title="Income Bucket wise Performance Tag Frequency")
# The Lower Income classes the more chances of default. Lower Incomes have the higher default frequency

# Income vs Credit Usage
ggplot(merged_data, aes(Income_Bin, Avg_CC_Utilization_12_months, fill = Income_Bin)) + 
  geom_boxplot()
# Lower incomes use more Credit.

# Income vs Performance
ggplot(merged_data, aes(Performance_Tag, Income, fill = as.factor(Performance_Tag))) + 
  geom_boxplot()

# Income with Age
ggplot(merged_data, aes(x = Age_Bin, y = Income, fill = Age_Bin)) + geom_boxplot()
ggplot(merged_data, aes(x = Age, y = Income)) + geom_smooth()
# Income Changes with different stages in Life
# The median income is lowest between Age 26 and 40 which represent the stages of life for growing, marriage, buying a house, raising dependents etc
# Income also drops towards old age at retirement

# Plot for Marital status at time of application frquency
ggplot(merged_data, aes(x=Marital_Status, y=..count../1000, fill=No_Of_Dependents)) +
  geom_bar(position = "dodge")+
  scale_fill_manual(values = cp_8)+
  labs(x="Marital Status at time of application", y="Frequency in 1000s", fill="No of Dependents", title="Frequency of different Marital Status") +
  theme_minimal()

# Marital Status wise Performance Tag Frequency
merged_data %>%
  filter(!is.na(Performance_Tag)) %>%
  ggplot(aes(x=Marital_Status, y=..count../1000, fill=as.factor(Performance_Tag))) +
  geom_bar(position = "dodge") +
  theme_minimal()+
  scale_fill_manual(values = cp_2) +
  labs(x="Marital Status", y="Frequency in 1000s", fill="Performance Tag", title="Marital Status wise Performance Tag Frequency")

# Credit Card Use
ggplot(merged_data, aes(x = as.factor(Performance_Tag), y = Avg_CC_Utilization_12_months, fill = as.factor(Performance_Tag))) + 
  geom_boxplot() +
  labs(x="Performance", y="Average Use of Credit Card in 12 Months", fill="Performance Tag", title="Use of Credit Card wise Performance Tag Frequency")
# The rate the use of Credit Card affect the chances of loan.
# median use of credit card in 12 months is less for non-default loan
# There is a correlation between Performance and Credit Card use

# Age
ggplot(merged_data, aes(x=Age_Bin, y=..count../1000, fill=Age_Bin)) +
  geom_bar() +
  labs(x="Age Bin", y="Frequency in 1000s", fill="Age Bin", title="Frequency of different Age Bins") +
  theme_minimal()

# Age Bucket wise Performance Tag Frequency
merged_data %>%
  filter(!is.na(Performance_Tag)) %>%
  ggplot(aes(x=Age_Bin, y=..count../1000, fill=as.factor(Performance_Tag))) +
  geom_bar(position = "dodge") +
  theme_minimal()+
  scale_fill_manual(values = cp_2) +
  labs(x="Age Buckets", y="Frequency in 1000s", fill="Performance Tag", title="Age Bucket wise Performance Tag")

head(merged_data$No_Of_90_DPD_12_months)

#Effect of Past Late Payments vs Performance

ggplot(merged_data, aes(Performance_Tag, No_Of_30_DPD_12_months, fill = as.factor(Performance_Tag))) + geom_boxplot() +
  labs(x="Performance", y="No Of 90 DPD in 12_months", fill="Performance Tag", title="Late Payment vs Performance Tag")
ggplot(merged_data, aes(Performance_Tag, No_Of_60_DPD_12_months, fill = as.factor(Performance_Tag))) + geom_boxplot() +
  labs(x="Performance", y="No Of 60 DPD in 12_months", fill="Performance Tag", title="Late Payment vs Performance Tag")
# The Higher the number of Late Payments the more the chances of default

#Effect of Credit check
ggplot(merged_data, aes(as.factor(Performance_Tag), Inquiries_12_months, fill = as.factor(Performance_Tag))) + geom_boxplot() + 
  labs(x="Performance", y="Inquiries for Other Loans", fill="Performance Tag", title="Credit Check or Loan enquiry vs Performance Tag")



# Correlation Matrix of Financial Information
ggcorrplot(cor(merged_data[, c(2,seq(13,29,1))]), lab = T)

ggplot(merged_data, aes(as.factor(Performance_Tag), Outstanding_Balance/1000, fill = as.factor(Performance_Tag))) + geom_boxplot() + 
  labs(x="Performance", y="Outstanding Balance", fill="Performance Tag", title="Other Debt vs Performance Tag")