# Look at the distribution of KW acoustic encounters

# Load data
data = read.csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Train.csv')
data = rbind(data, read.csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Test.csv')
             
# Lets first look at the inter detection interval of SRKW calls across different
# deployments

SRKW_data = subset(data, Ecotype== 'SRKW' & !is.nan('UTC'))

library(ggplot2)
library(tidyr)
# Load necessary libraries
library(tidyr)
library(lubridate)  # For handling date-time data
library(dplyr)

SRKW_data$UTC = as.POSIXct(SRKW_data$UTC)

# Assuming your data frame is named 'data'

# Step 1: Identify acoustic encounters within each deployment
# Define a function to calculate inter-detection intervals
calculate_inter_detection_intervals <- function(UTC) {
  intervals <- c(NA, round(diff(UTC),2))
  return(intervals)
}


encounters <- SRKW_data %>%
  group_by(Dep) %>%
  arrange(UTC) %>%
  mutate(
    inter_detection_time_seconds = calculate_inter_detection_intervals(UTC),
    no_detection_period = ifelse(is.na(inter_detection_time_seconds) | inter_detection_time_seconds >= 60, 1, 0),
    encounter_id = cumsum(no_detection_period)
  ) %>%
  ungroup()

# Figure out the number of encounters per deployment
aa = aggregate(data = encounters, FileOk ~ Dep+ encounter_id, FUN = length)


ggplot(data = aa, aes(x= Dep, y=FileOk, color = Dep))+
  geom_point()+
  geom_jitter()+
  coord_trans( y="log2")

# Load the malahat data 

malahatPreds = read.csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\NN_scores_ReBalanced_melSpec_5class_8khz_300hz.csv')

ggplot(malahatPreds)+
  geom_point(aes(x = SNR, y= Pred_value, color = Correct))+
  facet_wrap(~LabelsShort)

# Which calls are confidently wrong
bad_data = subset(malahatPreds, Correct==FALSE)

malahatPreds$UTC = as.POSIXct(malahatPreds$UTC)
# Not a strong relationship break into encounters
encounters <- malahatPreds %>%
  group_by(Dep) %>%
  arrange(UTC) %>%
  mutate(
    inter_detection_time_seconds = calculate_inter_detection_intervals(UTC),
    no_detection_period = ifelse(is.na(inter_detection_time_seconds) | inter_detection_time_seconds >= 60, 1, 0),
    encounter_id = cumsum(no_detection_period)
  ) %>%
  ungroup()

encounters$UnqueEncounters = paste0(encounters$Dep, encounters$encounter_id)

# now figure out the new encounter predictions
uniqueEncounters  = unique(encounters$UnqueEncounters)

for (ii in 1:length(uniqueEncounters)){
  
  idxs = which(encounters$UnqueEncounters== uniqueEncounters[ii])
  
  # get the predictions
  preds = encounters[idxs, c("AB", "BKW",  "SRKW", "NRKW" , "OFFSHORE")]
  
  # get the new prediction
  if(nrow(preds)>1){}else{
    NewgGuess = colnames(preds)[which.max(preds)]}
  
}


