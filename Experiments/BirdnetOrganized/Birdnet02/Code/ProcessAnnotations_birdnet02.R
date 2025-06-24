#Birdnet 02
# In this model we will have a total of 2k annotations for each class, however
# We will balance across call types with at least 100 in each calltype.


rm(list = ls())
# make it repeatable
set.seed(5)
DCLDE_train = read.csv( 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv')


# Exclude alaska and offshore data
DCLDE_train<-DCLDE_train[DCLDE_train$Labels %in% 
                                               c('Background', 'HW', 'SRKW','TKW'),]

DCLDE_train<-DCLDE_train[!DCLDE_train$CalltypeCategory %in% 
                           c('N09', 'OFF07'),]


table(DCLDE_train$CalltypeCategory)

library(dplyr)

# Start with an empty list to store class-wise samples
balanced_list <- list()

# Get all unique labels
all_labels <- unique(DCLDE_train$Labels)

# Iterate over labels
for (lab in all_labels) {
  
  df_lab <- DCLDE_train %>% filter(Labels == lab)
  
  if (lab %in% c("SRKW", "TKW")) {
    
    calltypes <- unique(na.omit(df_lab$CalltypeCategory))
    
    calltype_samples <- lapply(calltypes, function(ct) {
      ct_rows <- df_lab %>% filter(CalltypeCategory == ct)
      slice_sample(ct_rows, n = 100, replace = (nrow(ct_rows) < 100))
    })
    
    ct_block <- bind_rows(calltype_samples)
    ct_block_dedup <- ct_block %>% distinct()
    
    n_remaining <- 2000 - nrow(ct_block_dedup)
    
    remaining_pool <- df_lab %>%
      anti_join(ct_block_dedup, by = colnames(df_lab))
    
    extra_samples <- remaining_pool %>%
      group_by(Dataset) %>%
      slice_sample(prop = 1) %>%
      ungroup() %>%
      slice_sample(n = n_remaining, replace = TRUE)
    
    final_df <- bind_rows(ct_block_dedup, extra_samples) %>%
      slice_sample(n = 2000)
  } else {
    # For non-killer-whale labels
    final_df <- df_lab %>%
      group_by(Dataset) %>%
      slice_sample(prop = 1) %>%  # shuffle within datasets
      ungroup() %>%
      slice_sample(n = 2000)
  }
  
  balanced_list[[lab]] <- final_df
}

# Combine all label-based samples into final dataframe
balanced_data <- bind_rows(balanced_list)

write.csv(balanced_data, 'DCLDE_train_birdnet02.csv')

# Sanity check
print(table(balanced_data$Labels))

library(dplyr)
library(ggplot2)


# Plot label distribution by dataset
label_dataset_counts <- balanced_data %>%
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





