# Birdnet 7 — Full ecotype coverage with augmentation for rare call types
rm(list = ls())
set.seed(5)

library(dplyr)
library(ggplot2)

# Load data
DCLDE_train <- read.csv(
  'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv'
)

# Filter to only valid ecotype labels
DCLDE_train <- DCLDE_train %>%
  filter(Labels %in% c('Background', 'HW', 'SRKW', 'TKW', 'OKW', 'NRKW')) %>%
  filter(!CalltypeCategory %in% c('N09', 'OFF07'))

augment_calltype <- function(df, label, target_per_type = 100) {
  calls_augmented <- df %>% filter(Labels == label)
  calltypes <- unique(na.omit(calls_augmented$CalltypeCategory))
  
  for (ct in calltypes) {
    ct_rows <- calls_augmented %>% filter(CalltypeCategory == ct)
    n_ct <- nrow(ct_rows)
    
    if (n_ct < target_per_type) {
      n_needed <- target_per_type - n_ct
      add_rows <- ct_rows %>%
        slice_sample(n = n_needed, replace = TRUE)
      
      # Shift begin/end times by small % of duration
      shifts <- runif(n_needed, min = -0.5, max = 0.5)
      durations <- add_rows$Duration
      sec_shifts <- floor(shifts * durations)
      
      add_rows$FileBeginSec <- add_rows$FileBeginSec + sec_shifts
      add_rows$FileEndSec <- add_rows$FileEndSec + sec_shifts
      
      # Append augmented rows
      calls_augmented <- bind_rows(calls_augmented, add_rows)
    }
  }
  
  return(calls_augmented)
}

ecotype_classes <- c("SRKW", "TKW", "OKW", "NRKW")
balanced_list <- list()

for (ecotype in ecotype_classes) {
  df_label <- DCLDE_train %>% filter(Labels == ecotype)
  df_augmented <- augment_calltype(df_label, ecotype)
  
  # Pad to 4600 total
  n_current <- nrow(df_augmented)
  if (n_current < 4600) {
    extra <- df_label %>%
      slice_sample(n = 4600 - n_current, replace = TRUE)
    df_augmented <- bind_rows(df_augmented, extra)
  }
  
  df_augmented <- df_augmented %>% slice_sample(n = 4600)
  balanced_list[[ecotype]] <- df_augmented
}

# Add HW and Background
balanced_list[['HW']] <- DCLDE_train %>%
  filter(Labels == "HW") %>%
  slice_sample(n = 4600)

balanced_list[['Background']] <- DCLDE_train %>%
  filter(Labels == "Background") %>%
  slice_sample(n = 4600)

# Combine into final dataframe
balanced_data <- bind_rows(balanced_list)
write.csv(balanced_data, 'DCLDE_train_birdnet07.csv', row.names = FALSE)

# Sanity check
print(table(balanced_data$Labels))

# Plot label distribution by dataset
label_dataset_counts <- balanced_data %>%
  group_by(Labels, Dataset) %>%
  summarise(Count = n(), .groups = 'drop')

ggplot(label_dataset_counts, aes(x = Labels, y = Count, fill = Dataset)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Birdnet 07: Dataset Composition by Label and Dataset",
       x = "Label",
       y = "Number of Annotations",
       fill = "Dataset") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
