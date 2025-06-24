# Birdnet 7: Ecotype-balanced with augmentation and stratified background
rm(list = ls())
set.seed(5)

library(dplyr)
library(readr)

# Load data
DCLDE_train <- read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv')

# Preprocess
DCLDE_train <- DCLDE_train %>%
  mutate(ID = row_number(),
         CenterTime = (FileBeginSec + FileEndSec) / 2,
         Duration = FileEndSec - FileBeginSec,
         CalltypeCategory = ifelse(is.na(CallType), "None", CallType),
         HasQ = grepl("\\?", CallType))

# Filter valid killer whale annotations
kw_labels <- c("NRKW", "OKW", "SRKW", "TKW")
DCLDE_kw <- DCLDE_train %>%
  filter(Ecotype %in% kw_labels,
         HasQ == FALSE,
         !(CallType %in% c("Buzz", "buzz", "EL", "Multiple", "Rasp", "Unk", "Whistle", 
                           "whistle/tone", "W", "whistle", "rasp", "Ck", "")),
         !is.na(CallType))

# Assign calltype categories manually
kw_cats <- list(
  NRKW = c("N01", "N02", "N03", "N04", "N05", "N07", "N08", "N09", "N011", "N016", "N018", "N20", "N23", "N24", "N25", "N28", "N29", "N30", "N32", "N33", "N39", "N40", "N41", "N44", "N45", "N48"),
  OKW  = c("OFF02", "OFF07", "OFF17", "OFF19", "OFF30"),
  TKW  = c("T01", "T02", "T03", "T04", "T07", "T08", "T11", "T12", "T13"),
  SRKW = c("S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S10", "S12", "S13", "S14", "S16", "S17", "S18", "S19", "S22", "S31", "S32", "S33", "S36", "S37", "S40", "S41", "S42", "S44")
)

assign_calltype <- function(df, call_list, label) {
  rows <- lapply(call_list, function(ct) {
    match <- df %>% filter(grepl(ct, CallType))
    if (nrow(match) > 0) match$CalltypeCategory <- ct
    return(match)
  })
  bind_rows(rows)
}

# Assign calltype categories
balanced_ecotypes <- list()
for (label in kw_labels) {
  calls_labeled <- assign_calltype(
    df = DCLDE_kw %>% filter(Ecotype == label),
    call_list = kw_cats[[label]],
    label = label
  )
  
  # Balance call types to ≥100 with augmentation
  calltypes <- unique(calls_labeled$CalltypeCategory)
  augmented <- calls_labeled
  for (ct in calltypes) {
    subdf <- calls_labeled %>% filter(CalltypeCategory == ct)
    if (nrow(subdf) < 100) {
      to_add <- 100 - nrow(subdf)
      sampled <- subdf %>% slice_sample(n = to_add, replace = TRUE)
      shifts <- floor(runif(to_add, -50, 50)) / 100 * sampled$Duration
      sampled$FileBeginSec <- sampled$FileBeginSec + shifts
      sampled$FileEndSec <- sampled$FileEndSec + shifts
      augmented <- bind_rows(augmented, sampled)
    }
  }
  
  # Pad to 4600 using unannotated calls from same ecotype
  annotated_ids <- augmented$ID
  untyped_pool <- DCLDE_train %>%
    filter(Ecotype == label, !(ID %in% annotated_ids))
  
  if (nrow(augmented) < 4600) {
    pad_needed <- 4600 - nrow(augmented)
    extra <- untyped_pool %>% slice_sample(n = pad_needed, replace = TRUE)
    augmented <- bind_rows(augmented, extra)
  }
  
  balanced_ecotypes[[label]] <- augmented %>% mutate(Labels = label) %>% slice_sample(n = 4600)
}

# Add HW with stratified sampling across Provider
HW_df <- DCLDE_train %>%
  filter(ClassSpecies == "HW") %>%
  group_by(Provider) %>%
  slice_sample(prop = 1) %>%
  ungroup() %>%
  slice_sample(n = 4600) %>%
  mutate(Labels = "HW")

# Add Background (AB + UndBio), stratified across Provider
BG_df <- DCLDE_train %>%
  filter(ClassSpecies %in% c("AB", "UndBio")) %>%
  group_by(Provider) %>%
  slice_sample(prop = 1) %>%
  ungroup() %>%
  slice_sample(n = 4600) %>%
  mutate(Labels = "Background")

# Combine everything
birdnet08 <- bind_rows(
  balanced_ecotypes$NRKW,
  balanced_ecotypes$OKW,
  balanced_ecotypes$SRKW,
  balanced_ecotypes$TKW,
  HW_df,
  BG_df
)

# Sanity check
print(table(birdnet08$Labels))

# Save final dataset
write_csv(birdnet08, 'DCLDE_train_birdnet07_balanced_4600perclass.csv')

library(ggplot2)

# Plot label distribution by dataset
label_dataset_counts <- birdnet08 %>%
  group_by(Labels, Dataset) %>%
  summarise(Count = n(), .groups = 'drop')

ggplot(label_dataset_counts, aes(x = Labels, y = Count, fill = Dataset)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Birdnet 8: Dataset Composition by Label and Dataset",
       x = "Label",
       y = "Number of Annotations",
       fill = "Dataset") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))