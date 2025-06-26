# Birdnet 03 — Balanced per class with calltype control for killer whales
rm(list = ls())
set.seed(5)

library(dplyr)
library(ggplot2)

# Load data
DCLDE_train <- read.csv(
  'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv'
)

# Filter labels of interest
DCLDE_train <- DCLDE_train %>%
  filter(Labels %in% c('Background', 'HW', 'SRKW', 'TKW')) %>%
  filter(!CalltypeCategory %in% c('N09', 'OFF07'))

# Start with an empty list to store class-wise samples
balanced_list <- list()
all_labels <- unique(DCLDE_train$Labels)

for (lab in all_labels) {
  
  df_lab <- DCLDE_train %>% filter(Labels == lab)
  
  if (lab %in% c("SRKW", "TKW")) {
    
    # Ensure at least 100 per calltype
    calltypes <- unique(na.omit(df_lab$CalltypeCategory))
    
    calltype_samples <- lapply(calltypes, function(ct) {
      ct_rows <- df_lab %>% filter(CalltypeCategory == ct)
      slice_sample(ct_rows, n = 100, replace = (nrow(ct_rows) < 100))
    })
    
    ct_block <- bind_rows(calltype_samples)
    ct_block_dedup <- ct_block %>% distinct()
    
    # Fill up to 2000, excluding already-sampled rows
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
    
  } else if (lab == "Background") {
    
    # Balanced 1000 Abiotic + 1000 UndBio
    abiotic_rows <- df_lab %>%
      filter(ClassSpecies == "AB") %>%
      group_by(Dataset) %>%
      slice_sample(prop = 1) %>%
      ungroup() %>%
      slice_sample(n = 1000)
    
    undbio_rows <- df_lab %>%
      filter(ClassSpecies == "UndBio") %>%
      group_by(Dataset) %>%
      slice_sample(prop = 1) %>%
      ungroup() %>%
      slice_sample(n = 1000)
    
    final_df <- bind_rows(abiotic_rows, undbio_rows)
    
  } else {
    # All other labels (e.g., HW)
    final_df <- df_lab %>%
      group_by(Dataset) %>%
      slice_sample(prop = 1) %>%
      ungroup() %>%
      slice_sample(n = 2000)
  }
  
  balanced_list[[lab]] <- final_df
}

# Combine into final dataframe
balanced_data <- bind_rows(balanced_list)

write.csv(balanced_data, 'DCLDE_train_birdnet03.csv')


# Sanity check
print(table(balanced_data$Labels))

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
