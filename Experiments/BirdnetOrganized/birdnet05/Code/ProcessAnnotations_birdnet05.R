# Same as first birdnet save 6k detections

# make it repeatable
rm(list =ls())
set.seed(5)
DCLDE_train = read.csv( 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv')


labels = unique(DCLDE_train$Labels)
trainIDX = c()


for(label in labels){
  labelIdx = which(DCLDE_train$Labels == label) 
  
  if (length(labelIdx)<6000)
  {idx = sample(labelIdx, 6000, replace = TRUE)}
  else{idx = sample(labelIdx, 6000, replace = FALSE)}
  trainIDX= c(trainIDX, idx)
  
  
}

# Write the database
DCLDE_train_birdnet05 = DCLDE_train[trainIDX,]

# Exclude alaska and offshore data
DCLDE_train_birdnet05= DCLDE_train_birdnet05[DCLDE_train_birdnet05$Labels %in% 
                                               c('Background', 'HW', 'SRKW','TKW'),]

table(DCLDE_train_birdnet05$Labels)
write.csv(DCLDE_train_birdnet05, 'DCLDE_train_birdnet05.csv')


# Sanity check
print(table(DCLDE_train_birdnet05$Labels))
library(ggplot2)
library(dplyr)
# Plot label distribution by dataset
label_dataset_counts <- DCLDE_train_birdnet05 %>%
  group_by(Labels, Dataset) %>%
  summarise(Count = n(), .groups = 'drop')

ggplot(label_dataset_counts, aes(x = Labels, y = Count, fill = Dataset)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Balanced Dataset Composition by Label and Dataset",
       x = "Label",
       y = "Number of Annotations",
       fill = "Dataset") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
