# Birdnet 9: exactly birdnet 08 (as below) but excludes Alaska and NRKW data

# Filters Ecotype directly and excludes ambiguous call types manually
# Explicit mapping of call types (N01, S03, etc.) to each ecotype using a custom assign_calltype() function
# Augmentation Applied per ecotype using ecotype-specific calltype lists, then pads with unannotated ecotype data
# Stratified across Provider before sampling 4600 examples each
# Filters out a detailed list of call types (Buzz, Whistle, Unk, etc.) and excludes annotations with a ?
# Augments every mapped calltype to ensure representation per ecotype


rm(list = ls())
set.seed(5)

library(dplyr)
library(readr)

# Load data
DCLDE_train <- read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\DCLDE_train_parent.csv')

DCLDE_train= DCLDE_train[DCLDE_train$Labels %in% 
                                               c('Background', 'HW', 'SRKW','TKW'),]

DCLDE_train= DCLDE_train[DCLDE_train$Provider != 'UAF',]


# Preprocess
DCLDE_train <- DCLDE_train %>%
  mutate(ID = row_number(),
         CenterTime = (FileBeginSec + FileEndSec) / 2,
         Duration = FileEndSec - FileBeginSec,
         CalltypeCategory = ifelse(is.na(CallType), "None", CallType),
         HasQ = grepl("\\?", CallType))

# Filter valid killer whale annotations
kw_labels <- c("SRKW", "TKW")
DCLDE_kw <- DCLDE_train %>%
  filter(Ecotype %in% kw_labels,
         HasQ == FALSE,
         !(CallType %in% c("Buzz", "buzz", "EL", "Multiple", "Rasp", "Unk", "Whistle", 
                           "whistle/tone", "W", "whistle", "rasp", "Ck", "","Whup")),
         !is.na(CallType))

# Assign calltype categories manually
kw_cats <- list(
  TKW  = c("T01", "T02", "T03", "T04", "T07", "T08", "T11", "T12", "T13"),
  SRKW = c("S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S10", "S12", "S13", "S14", "S16", "S17", "S18", "S19", "S22", "S31", "S32", "S33", "S36", "S37", "S40", "S41", "S42", "S44")
)


nExamples = 3000

assign_calltype <- function(df, call_list, label) {
  rows <- lapply(call_list, function(ct) {
    match <- df %>% filter(grepl(ct, CallType))
    if (nrow(match) > 0) match$CalltypeCategory <- ct
    return(match)
  })
  bind_rows(rows)
}

nCalltypeReplicates = 200

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
    if (nrow(subdf) < nCalltypeReplicates) {
      to_add <- nCalltypeReplicates - nrow(subdf)
      sampled <- subdf %>% slice_sample(n = to_add, replace = TRUE)
      shifts <- floor(runif(to_add, -50, 50)) / 100 * sampled$Duration
      sampled$FileBeginSec <- sampled$FileBeginSec + shifts
      sampled$FileEndSec <- sampled$FileEndSec + shifts
      augmented <- bind_rows(augmented, sampled)
    }
  }
  
  # Pad to nExamples using unannotated calls from same ecotype
  annotated_ids <- augmented$ID
  untyped_pool <- DCLDE_train %>%
    filter(Ecotype == label, !(ID %in% annotated_ids))
  
  if (nrow(augmented) < nExamples) {
    pad_needed <- nExamples - nrow(augmented)
    extra <- untyped_pool %>% slice_sample(n = pad_needed, replace = TRUE)
    augmented <- bind_rows(augmented, extra)
  }
  
  balanced_ecotypes[[label]] <- augmented %>% mutate(Labels = label) %>% slice_sample(n = nExamples)
}

# Add HW with stratified sampling across Provider
HW_df <- DCLDE_train %>%
  filter(ClassSpecies == "HW") %>%
  group_by(Provider) %>%
  slice_sample(prop = 1) %>%
  ungroup() %>%
  slice_sample(n = nExamples) %>%
  mutate(Labels = "HW")

# Add Background (AB + UndBio), stratified across Provider
BG_df <- DCLDE_train %>%
  filter(ClassSpecies %in% c("AB", "UndBio")) %>%
  group_by(Provider) %>%
  slice_sample(prop = 1) %>%
  ungroup() %>%
  slice_sample(n = nExamples) %>%
  mutate(Labels = "Background")

# Combine everything
birdnet10 <- bind_rows(
  balanced_ecotypes$NRKW,
  balanced_ecotypes$OKW,
  balanced_ecotypes$SRKW,
  balanced_ecotypes$TKW,
  HW_df,
  BG_df
)

# Sanity check
print(table(birdnet10$Labels))


# Save final dataset
birdnet10 = write_csv(birdnet10, 'DCLDE_train_birdnet10.csv')

library(ggplot2)

# Plot label distribution by dataset
label_dataset_counts <- birdnet10 %>%
  group_by(Labels, Dataset) %>%
  summarise(Count = n(), .groups = 'drop')

ggplot(label_dataset_counts, aes(x = Labels, y = Count, fill = Dataset)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Birdnet 9: Dataset Composition by Label and Dataset",
       x = "Label",
       y = "Number of Annotations",
       fill = "Dataset") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
