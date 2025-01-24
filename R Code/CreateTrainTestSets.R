# Code for makign train test sets
allAnno = read.csv('C:/Users/kaity/Documents/GitHub/DCLDE2026/Annotations.csv')
allAnno = allAnno[,c(2:16)]
allAnno = rbind(allAnno, JASCO_malahat)
colnames(JASCO_malahat)[2]<-'Dataset'

# double check all audio files are there
allAnno$FileOk  = file.exists(allAnno$FilePath)

# Fun fun, some of the annotations have negative durations... cull them
allAnno$Duration = allAnno$FileEndSec-allAnno$FileBeginSec
allAnno$CenterTime = allAnno$FileBeginSec+allAnno$Duration/2
allAnno = subset(allAnno, Duration>0.1)


# overall annotations
table(allAnno$ClassSpecies)

# Pull out the killer whale data
KW_data = subset(allAnno, KW ==1)

table(KW_data$Ecotype)



#############################################################################
#  Train test set
#############################################################################


library(dplyr)
library(lubridate)
library(stringr)
library(caret)

# Example input data
# Replace this with your actual `allAnnoEcotype` data
set.seed(123)


# Fun fun, some of the annotations have negative durations... cull them
allAnno$Duration = allAnno$FileEndSec-allAnno$FileBeginSec
allAnno$CenterTime = allAnno$FileBeginSec+allAnno$Duration/2
allAnno = subset(allAnno, Duration>0.1)



# Only KW certain and other classes
allAnnoEcotype = subset(allAnno, KW_certain %in% c(1, NA))

# Create the proper labels
# Use the ecotype where available
allAnnoEcotype$Labels = as.character(allAnnoEcotype$Ecotype)

# Use ClassSpecies where not available
allAnnoEcotype$Labels[is.na(allAnnoEcotype$Ecotype)] =
  allAnnoEcotype$ClassSpecies[is.na(allAnnoEcotype$Ecotype)] 
allAnnoEcotype = subset(allAnnoEcotype, Labels != 'KW')
allAnnoEcotype$Labels = as.factor(allAnnoEcotype$Labels)

# Change levels to AB, HW, RKW, TKW, OKW, UndBio
levels(allAnnoEcotype$Labels)<-c('AB', 'HW', 'RKW', 'OKW', 'RKW', 'RKW', 
                                 'TKW', 'UndBio')

# Numeric labels for training
allAnnoEcotype$label = as.numeric(as.factor(allAnnoEcotype$Labels))-1


# Add a `Date` column to group by UTC date
allAnnoEcotype <- allAnnoEcotype %>% mutate(Date = as.Date(UTC))

# Separate Malahat data
malahat_data <- allAnnoEcotype %>% filter(Provider == "JASCO_Malahat")

# Filter out Malahat from the main dataset
allAnnoEcotype <- allAnnoEcotype %>% filter(Provider != "JASCO_Malahat")


# Balance Data via Augmentation
augment_data <- function(data, target_label, ref_label) {
  label_counts <- table(data$Labels)
  to_augment <- label_counts[ref_label] - label_counts[target_label]
  
  if (to_augment > 0) {
    augment <- data %>% filter(Labels == target_label) %>%
      sample_n(to_augment, replace = TRUE) %>%
      mutate(
        random_offset = Duration * runif(n(), 0.10, 0.25) * ifelse(runif(n()) >
                                                                     0.5, 1, -1),
        FileBeginSec = FileBeginSec + random_offset,
        FileEndSec = FileBeginSec + Duration
      )
    return(bind_rows(data, augment))
  }
  return(data)
}

# Balance the ecotypes
allAnnoEcotype <- augment_data(allAnnoEcotype, "TKW", "RKW")
allAnnoEcotype <- augment_data(allAnnoEcotype, "OKW", "RKW")

# Train/Test Split

# Group data by Deployment and Date
allAnnoEcotype <- allAnnoEcotype %>%
  group_by(Dep, Date) %>%
  mutate(GroupID = cur_group_id()) %>%
  ungroup()

# Stratify Train/Test Split by Labels
set.seed(123)

# Create a stratified partition of 80% for training
train_indices <- createDataPartition(allAnnoEcotype$Labels, p = 0.8, list = FALSE)
train <- allAnnoEcotype[train_indices, ] %>%
  mutate(traintest = "Train")
test <- allAnnoEcotype[-train_indices, ] %>%
  mutate(traintest = "Test")

# Verify Split
table(train$Labels)  # Check label balance in train
table(test$Labels)   # Check label balance in test

# Export train test sets
write.csv(train, row.names = FALSE,
          file = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\RTO_train.csv')

write.csv(test,  row.names = FALSE,
          file = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\RTO_test.csv')

write.csv(malahat_data, row.names = FALSE, 
          file = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\RTO_malahat.csv')

###############################################################################
# Create data set that has nixed some of the humpback detentions to balance
#################################################################################

# Code for makign train test sets
allAnno = read.csv('C:/Users/kaity/Documents/GitHub/DCLDE2026/Annotations.csv')
colnames(allAnno)[2]<-'Dep'

# double check all audio files are there
allAnno$FileOk  = file.exists(allAnno$FilePath)

# Fun fun, some of the annotations have negative durations... cull them
allAnno$Duration = allAnno$FileEndSec-allAnno$FileBeginSec
allAnno$CenterTime = allAnno$FileBeginSec+allAnno$Duration/2
allAnno = subset(allAnno, Duration>0.1)


# overall annotations
table(allAnno$ClassSpecies)

# Pull out the killer whale data
KW_data = subset(allAnno, KW ==1)

table(KW_data$Ecotype)


#############################################################################
#  Train test set
#############################################################################


library(dplyr)
library(lubridate)
library(stringr)
library(caret)

# Example input data
# Replace this with your actual `allAnnoEcotype` data
set.seed(123)


# Fun fun, some of the annotations have negative durations... cull them
allAnno$Duration = allAnno$FileEndSec-allAnno$FileBeginSec
allAnno$CenterTime = allAnno$FileBeginSec+allAnno$Duration/2
allAnno = subset(allAnno, Duration>0.1)



# Only KW certain and other classes
allAnnoEcotype = subset(allAnno, KW_certain %in% c(1, NA))

# Create the proper labels
# Use the ecotype where available
allAnnoEcotype$Labels = as.character(allAnnoEcotype$Ecotype)

# Use ClassSpecies where not available
allAnnoEcotype$Labels[is.na(allAnnoEcotype$Ecotype)] =
  allAnnoEcotype$ClassSpecies[is.na(allAnnoEcotype$Ecotype)] 
allAnnoEcotype = subset(allAnnoEcotype, Labels != 'KW')
allAnnoEcotype$Labels = as.factor(allAnnoEcotype$Labels)

# Change levels to AB, HW, RKW, TKW, OKW, UndBio
levels(allAnnoEcotype$Labels)<-c('AB', 'HW', 'RKW', 'OKW', 'RKW', 'RKW', 
                                 'TKW', 'UndBio')

# Numeric labels for training
allAnnoEcotype$label = as.numeric(as.factor(allAnnoEcotype$Labels))-1


# Add a `Date` column to group by UTC date
allAnnoEcotype <- allAnnoEcotype %>% mutate(Date = as.Date(UTC))

# Separate Malahat data
malahat_data <- allAnnoEcotype %>% filter(Provider == "JASCO_Malahat")

# Filter out Malahat from the main dataset
allAnnoEcotype <- allAnnoEcotype %>% filter(Provider != "JASCO_Malahat")


# Balance ecotypes via Augmentation
augment_data <- function(data, target_label, ref_label) {
  label_counts <- table(data$Labels)
  to_augment <- label_counts[ref_label] - label_counts[target_label]
  
  if (to_augment > 0) {
    augment <- data %>% filter(Labels == target_label) %>%
      sample_n(to_augment, replace = TRUE) %>%
      mutate(
        random_offset = Duration * runif(n(), 0.10, 0.25) * ifelse(runif(n()) >
                                                                     0.5, 1, -1),
        FileBeginSec = FileBeginSec + random_offset,
        FileEndSec = FileBeginSec + Duration
      )
    return(bind_rows(data, augment))
  }
  return(data)
}

# Balance the ecotypes
allAnnoEcotype <- augment_data(allAnnoEcotype, "TKW", "RKW")
allAnnoEcotype <- augment_data(allAnnoEcotype, "OKW", "RKW")

# Train/Test Split

# Group data by Deployment and Date
allAnnoEcotype <- allAnnoEcotype %>%
  group_by(Dep, Date) %>%
  mutate(GroupID = cur_group_id()) %>%
  ungroup()

# Class counts- way too many humpbacks
table(allAnnoEcotype$Labels)

# West vancouver Island hugely over represented in the data
table(allAnnoEcotype$Dep[allAnnoEcotype$ClassSpecies== 'HW'])
nRemove =127630-36792  
hwidx = which(allAnnoEcotype$ClassSpecies== 'HW')
idxOut = sample(hwidx, nRemove)

# Now nix those files
allAnnoEcotype <- allAnnoEcotype[-c(idxOut), ]
table(allAnnoEcotype$ClassSpecies)


# Stratify Train/Test Split by Labels
set.seed(123)

# Create a stratified partition of 80% for training
train_indices <- createDataPartition(allAnnoEcotype$Labels, p = 0.8, list = FALSE)
train <- allAnnoEcotype[train_indices, ] %>%
  mutate(traintest = "Train")
test <- allAnnoEcotype[-train_indices, ] %>%
  mutate(traintest = "Test")

# Verify Split
table(train$Labels)  # Check label balance in train
table(test$Labels)   # Check label balance in test

# Export train test sets
write.csv(train, row.names = FALSE,
          file = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\RTO_train_HumpBalanced.csv')

write.csv(test,  row.names = FALSE,
          file = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\RTO_test_HumpBalanced.csv')

write.csv(malahat_data,  row.names = FALSE,
          file = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\RTO_malahat.csv')

