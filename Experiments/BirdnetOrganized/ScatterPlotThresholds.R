# Load libraries
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

# -------------------------------
# 1. Load and combine all model outputs
# -------------------------------
folder <- "C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/BirdnetOrganized"
model_ids <- sprintf("%02d", 1:10)
model_names <- paste0("Birdnet", model_ids)
file_paths <- file.path(folder, paste0(model_names, "_Malahat_eval.csv"))

all_preds <- lapply(seq_along(file_paths), function(i) {
  df <- read_csv(file_paths[i], show_col_types = FALSE)
  df$Model <- model_names[i]
  return(df)
}) %>%
  bind_rows()

# -------------------------------
# 2. Manually defined thresholds (based on prior logistic fit failures)
# -------------------------------
thresh_values <- c(
  0.28985, 0.5415,  0.508,
  0.2898,  0.54157, 0.5080,
  0.3821,  0.554,   0.5268,
  0.3600,  0.5530,  0.4905,
  0.3600,  0.553,   0.4905,
  0.3600,  0.553,   0.4905,
  0.2047,  0.5167,  0.485,
  0.2047,  0.516,   0.4855,
  0.4653,  0.565,   0.4743,
  0.4653,  0.565,   0.4743 # wrong
)
threshold_table <- expand.grid(
  Model = model_names,
  Class = c("SRKW", "TKW", "HW")
) %>%
  arrange(Model, Class) %>%
  mutate(Threshold = thresh_values)

# Pivot wider: one row per model, columns for each class threshold
threshold_wide <- threshold_table %>%
  pivot_wider(names_from = Class, values_from = Threshold, names_prefix = "Thresh_")

# Join thresholds into main predictions table
all_preds <- all_preds %>%
  left_join(threshold_wide, by = "Model")

# -------------------------------
# 3. Classify each row based on per-class threshold
# -------------------------------
ecotype_classes <- c("SRKW", "TKW", "HW")  # Set class list
all_preds <- all_preds %>%
  rowwise() %>%
  mutate(
    BestClass = ecotype_classes[which.max(c_across(all_of(ecotype_classes)))],
    BestScore = max(c_across(all_of(ecotype_classes))),
    BestThreshold = case_when(
      BestClass == "SRKW" ~ Thresh_SRKW,
      BestClass == "TKW" ~ Thresh_TKW,
      BestClass == "HW" ~ Thresh_HW,
      TRUE ~ NA_real_
    ),
    Predicted = ifelse(BestScore >= BestThreshold, BestClass, "Background")
  ) %>%
  ungroup()

# -------------------------------
# 4. Build Confusion Matrices per Model
# -------------------------------
cm_list <- all_preds %>%
  filter(Truth %in% ecotype_classes, Predicted %in% ecotype_classes) %>%
  group_by(Model) %>%
  {
    split_dfs <- group_split(., .keep = TRUE)
    model_keys <- group_keys(.)$Model
    setNames(split_dfs, model_keys)
  } %>%
  lapply(function(df) {
    table(Predicted = df$Predicted, Truth = df$Truth)
  })

# -------------------------------
# 5. Extract Recall and Confusion Metrics
# -------------------------------
extract_performance_metrics <- function(cm, model_name) {
  get_safe <- function(mat, row, col) {
    if (row %in% rownames(mat) && col %in% colnames(mat)) mat[row, col] else 0
  }
  
  total_SRKW <- sum(cm[, "SRKW"]);
  total_TKW  <- sum(cm[, "TKW"]);
  
  TP_SRKW <- get_safe(cm, "SRKW", "SRKW")
  TP_TKW  <- get_safe(cm, "TKW",  "TKW")
  
  Conf_SRKW_TKW <- get_safe(cm, "TKW", "SRKW") / total_SRKW
  Conf_TKW_SRKW <- get_safe(cm, "SRKW", "TKW") / total_TKW
  
  tibble(
    Model = model_name,
    Recall_SRKW = TP_SRKW / total_SRKW,
    Recall_TKW  = TP_TKW  / total_TKW,
    Confusion = mean(c(Conf_SRKW_TKW, Conf_TKW_SRKW))
  )
}

performance_df <- bind_rows(
  lapply(names(cm_list), function(m) extract_performance_metrics(cm_list[[m]], m))
)

# -------------------------------
# 6. Plot Two-Axis Performance Map
# -------------------------------
ggplot(performance_df, aes(x = Recall_SRKW, y = Recall_TKW)) +
  geom_point(aes(size = Confusion, color = Model), alpha = 0.75) +
  geom_text(aes(label = Model), nudge_y = 0.01, size = 3) +
  scale_size_continuous(range = c(2, 10), name = "SRKW↔TKW Confusion") +
  labs(title = "SRKW vs TKW Recall and Confusion",
       x = "Recall (SRKW)", y = "Recall (TKW") +
  theme_minimal()
