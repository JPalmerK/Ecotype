# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:31:46 2025

@author: kaity
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:04:22 2025

@author: kaity
"""


from keras.models import load_model
import EcotypeDefs as Eco
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_one_vs_others_roc(fp_df, all_df, relevant_classes=None, class_colors=None,
                           titleStr="One-vs-Others ROC Curve"):
    """
    Generates a One-vs-Others ROC curve for each class in the dataset.
    
    Parameters:
    - fp_df: DataFrame containing false positives (typically a subset of all_df)
    - all_df: Full dataset with 'Truth', 'Class', and 'Score' columns
    - relevant_classes: Optional list of class names to include in the comparison.
                        If None, all classes in the dataset will be used.
    - class_colors: Optional dictionary mapping class names to specific colors.
                    If None, uses Matplotlib's default 'tab10' colormap.
    - titleStr: Title for the ROC curve plot.

    Returns:
    - roc_data: Dictionary containing thresholds, false positive rate (FPR), and true positive rate (TPR) for each relevant class.
    """

    if relevant_classes is None:
        relevant_classes = np.unique(all_df['Class'])  # Use all classes if none are specified

    # Default colormap if no colors provided
    default_colors = {cls: plt.cm.get_cmap("tab10").colors[i % 10] for i, cls in enumerate(relevant_classes)}
    
    # Use provided colors or fall back to default
    color_map = default_colors if class_colors is None else {**default_colors, **class_colors}

    thresholds = np.linspace(0.35, 1, 100)  # Define the range of thresholds
    roc_data = {}  # Store ROC data for each relevant class

    plt.figure(figsize=(8, 6))

    for cls in relevant_classes:
        df_filtered = all_df[all_df['Truth'].isin(relevant_classes)].copy()
        df_class = fp_df[fp_df['Class'] == cls].copy()  # False positives subset
        tpData = df_filtered[df_filtered['Truth'] == cls].copy()  # True positives subset

        fpr, tpr = [], []

        for threshold in thresholds:
            # Apply threshold to determine predictions
            df_class['Predicted'] = df_class['Score'] >= threshold
            tpData['Predicted'] = (tpData['Score'] >= threshold) & (tpData['Class'] == cls)

            # Compute counts
            true_positive_count = tpData['Predicted'].sum()
            false_positive_count = df_class['Predicted'].sum()
            false_negative_count = len(tpData) - true_positive_count
            true_negative_count = len(df_filtered) - (true_positive_count + false_positive_count + false_negative_count)

            # Calculate FPR and TPR
            fpr_value = false_positive_count / (false_positive_count + true_negative_count) if (false_positive_count + true_negative_count) > 0 else 0
            tpr_value = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0

            fpr.append(fpr_value)
            tpr.append(tpr_value)

        # Store results for this class
        roc_data[cls] = {'thresholds': thresholds, 'fpr': np.array(fpr), 'tpr': np.array(tpr)}

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=cls, color=color_map.get(cls, "black"))  # Use class color or black fallback

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.show()

    return roc_data

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    data,
    threshold=0.5,
    class_column="Class",
    score_column="Score",
    truth_column="Truth"):
    """
    Plots a confusion matrix (rows = predicted, columns = true) based on a 
    given threshold (float or dict), annotating the diagonal cells with recall.

    Parameters:
    - data: DataFrame containing:
        * class_column (str): The model's predicted label.
        * score_column (float): The model's confidence score.
        * truth_column (str): The actual (ground truth) label.
    - threshold: Either:
        * A float (e.g. 0.5): single global threshold, or
        * A dict mapping class_name (str) -> threshold (float).
          Classes not in the dict fall back to 0.5.
    - class_column: Column name for the predicted class (default "Class").
    - score_column: Column name for the confidence score (default "Score").
    - truth_column: Column name for the ground truth (default "Truth").

    Returns:
    - cm_df: The unnormalized confusion matrix DataFrame (with predicted classes
             as rows and true classes as columns). The plotted heatmap is 
             row-normalized and includes recall on the diagonal.
    """

    df = data.copy()

    # ------------------------------------------------
    # 1) Assign threshold per row (if dict) or global
    # ------------------------------------------------
    if isinstance(threshold, dict):
        def get_threshold_for_row(row):
            return threshold.get(row[class_column], 0.5)
        df["_ThresholdToUse"] = df.apply(get_threshold_for_row, axis=1)
    else:
        df["_ThresholdToUse"] = threshold

    # ------------------------------------------------
    # 2) Binarize predictions at the threshold
    # ------------------------------------------------
    df["Predicted"] = np.where(
        df[score_column] >= df["_ThresholdToUse"],
        df[class_column],
        "Background"
    )

    # ------------------------------------------------
    # 3) Compute confusion matrix in the usual (truth, predicted) order
    #    and then transpose it to flip axes.
    # ------------------------------------------------
    # By default, confusion_matrix has shape:
    #   (len(labels), len(labels)) => (row = truth, col = predicted).
    # We transpose it afterward so that row = predicted, col = truth.
    labels = df[class_column].unique()  # or union of all truth/predict
    cm_normal = confusion_matrix(
        df[truth_column],
        df["Predicted"],
        labels=labels
    )

    # Flip (row=>predicted, col=>truth) by transposing
    cm_flipped = cm_normal.T

    # Create a DataFrame for unnormalized confusion matrix
    # index = predicted classes, columns = true classes
    cm_df = pd.DataFrame(
        cm_flipped,
        index=labels,
        columns=labels
    )

    # ------------------------------------------------
    # 4) Row-normalize this flipped matrix for display
    #    (so each row sums to 1).
    # ------------------------------------------------
    cm_df_norm = cm_df.div(cm_df.sum(axis=0), axis=1)

    # Remove empty rows (if any) from both unnormalized & normalized
    cm_df.dropna(axis=1, how='all', inplace=True)
    cm_df_norm.dropna(axis=1, how='all', inplace=True)

    # ------------------------------------------------
    # 5) Calculate recall for each class:
    #    Recall = TP / (total times class appears in ground truth).
    # ------------------------------------------------
    # Diagonal = predicted == c, truth == c. 
    # For recall, we need how many times c truly appears in `truth_column`.
    RecallArray = pd.DataFrame()
    RecallArray["Class"] = cm_df.index  # predicted classes

    # (A) True Positives on the diagonal (in flipped matrix)
    RecallArray["TP"] = np.diag(cm_df.values)

    # (B) Count how many times each class is in the truth
    truth_counts = df[truth_column].value_counts()

    # Map each predicted class to how often it appears in truth
    # (If a predicted class never appears as truth, recall is NaN)
    RecallArray["TruthCount"] = RecallArray["Class"].map(truth_counts).fillna(0).astype(int)
    RecallArray["Recall"] = RecallArray["TP"] / RecallArray["TruthCount"]
    RecallArray.loc[RecallArray["TruthCount"] == 0, "Recall"] = np.nan

    # ------------------------------------------------
    # 6) Plot the flipped & row-normalized matrix
    #    With recall on the diagonal cells
    # ------------------------------------------------
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_df_norm,
        annot=False,  # We'll place text manually
        cmap="Blues",
        linewidths=0.5
    )

    n_rows, n_cols = cm_df_norm.shape

    for i in range(n_rows):
        for j in range(n_cols):
            val = cm_df_norm.iloc[i, j]
            x_coord = j + 0.5
            y_coord = i + 0.5

            if i == j:
                # On the diagonal, also show recall
                cls_name = cm_df_norm.index[i]
                row_info = RecallArray[RecallArray["Class"] == cls_name]
                if not row_info.empty:
                    rec_val = row_info["Recall"].values[0]
                    if pd.notnull(rec_val):
                        text_str = f"{val:.2f}\nRecall: {rec_val:.2f}"
                    else:
                        text_str = f"{val:.2f}\nRecall: N/A"
                else:
                    text_str = f"{val:.2f}"
                text_color = "white"
            else:
                text_str = f"{val:.2f}"
                text_color = "black"

            ax.text(
                x_coord,
                y_coord,
                text_str,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12
            )

    plt.ylabel("Predicted Class")
    plt.xlabel("True Class")
    plt.title("Flipped Confusion Matrix (Predicted = Rows, Truth = Columns)")
    plt.show()

    # Clean up
    df.drop(columns=["_ThresholdToUse"], inplace=True, errors="ignore")
    return cm_df




def plot_confusion_matrix_multilabel(
    data,
    score_cols,
    truth_column="Truth",
    threshold=0.5):
    """
    Plots a confusion matrix for a multi-label classifier that outputs scores
    for multiple classes in each row. One row can surpass thresholds for more
    than one class, thus predicting multiple classes simultaneously.

    This version:
      - Removes 'Background' rows/columns from the final confusion matrix,
      - Removes any rows that are all NaN after normalization,
      - Forces the row and column order to match the order in `score_cols`.

    Parameters:
    - data: A pandas DataFrame with:
        * One column for the ground truth (truth_column).
        * Multiple columns (score_cols) for each class's prediction score.
    - score_cols: List of columns corresponding to each possible class's score,
                  e.g. ["SRKW", "TKW", "OKW", "HW"].
    - truth_column: Name of the ground-truth column in `data`.
    - threshold: Either:
        * A float (applies the same threshold to all classes), OR
        * A dict mapping class_name -> float threshold (others default to 0.5).

    Returns:
    - cm_df: The unnormalized multi-label confusion matrix (rows = true classes,
      columns = predicted classes), with 'Background' removed, row/column
      order forced to match `score_cols`, and any all-NaN rows dropped from
      the normalized heatmap.
    """

    # ---------------------------
    # 1) Determine the classes & thresholds
    # ---------------------------
    # We'll add "Background" for predictions if no class surpasses threshold.
    classes = list(score_cols)            # e.g. ["SRKW", "TKW", "OKW", "HW"]
    classes_with_bg = classes + ["Background"]

    def get_threshold_for_class(c):
        if isinstance(threshold, dict):
            return threshold.get(c, 0.5)  # default to 0.5 if missing
        else:
            return threshold  # single float

    # ---------------------------
    # 2) Build predicted sets for each row
    # ---------------------------
    predicted_sets = []
    for _, row in data.iterrows():
        row_predicted = []
        for c in classes:
            score_val = row[c]
            thr_val = get_threshold_for_class(c)
            if pd.notnull(score_val) and score_val >= thr_val:
                row_predicted.append(c)
        if not row_predicted:
            row_predicted = ["Background"]
        predicted_sets.append(row_predicted)

    # ---------------------------
    # 3) Identify possible truth labels
    # ---------------------------
    all_true_labels = sorted(data[truth_column].unique().tolist())
    # If ground truth can be "Background", handle it, otherwise add it
    if "Background" not in all_true_labels:
        all_possible_truth = all_true_labels + ["Background"]
    else:
        all_possible_truth = all_true_labels

    # Prepare array [num_truth_labels x num_pred_labels]
    cm_array = np.zeros((len(all_possible_truth), len(classes_with_bg)), dtype=int)

    # ---------------------------
    # 4) Fill the unnormalized confusion matrix
    # ---------------------------
    for i, row in enumerate(data.itertuples(index=False)):
        true_label = getattr(row, truth_column)
        if true_label not in all_possible_truth:
            true_label = "Background"
        row_idx = all_possible_truth.index(true_label)

        preds = predicted_sets[i]
        for p in preds:
            if p not in classes_with_bg:
                p = "Background"
            col_idx = classes_with_bg.index(p)
            cm_array[row_idx, col_idx] += 1

    # Convert to DataFrame
    cm_df = pd.DataFrame(cm_array, index=all_possible_truth, columns=classes_with_bg)

    # ---------------------------
    # 5) Remove "Background"
    # ---------------------------
    if "Background" in cm_df.index:
        cm_df.drop("Background", axis=0, inplace=True)
    if "Background" in cm_df.columns:
        cm_df.drop("Background", axis=1, inplace=True)

    # ---------------------------
    # 6) Force row/column order to match `score_cols`
    # ---------------------------
    # We'll only keep the classes that actually appear in `score_cols`,
    # but forcibly reindex in exactly that order. Missing classes get zero counts.
    cm_df = cm_df.reindex(index=score_cols, columns=score_cols, fill_value=0)

    # ---------------------------
    # 7) Create a row-normalized version and drop rows that become all NaN
    # ---------------------------
    cm_df_norm = cm_df.div(cm_df.sum(axis=1), axis=0)
    cm_df_norm.dropna(axis=0, how='all', inplace=True)

    # ---------------------------
    # 8) Compute recall: TP / total times class appears in ground truth
    # ---------------------------
    truth_counts = data[truth_column].value_counts()
    recall_vals = {}
    for cls_name in cm_df.index:
        if cls_name not in cm_df.columns:
            tp = 0
        else:
            tp = cm_df.loc[cls_name, cls_name]
        tot_truth = truth_counts.get(cls_name, 0)
        recall_vals[cls_name] = tp / tot_truth if tot_truth > 0 else np.nan

    # ---------------------------
    # 9) Plot the row-normalized matrix
    # ---------------------------
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_df_norm,
        annot=False,
        cmap="Blues",
        linewidths=0.5
    )

    # Manually annotate each cell
    n_rows = cm_df_norm.shape[0]
    n_cols = cm_df_norm.shape[1]

    for i in range(n_rows):
        for j in range(n_cols):
            val = cm_df_norm.iloc[i, j]
            x_coord = j + 0.5
            y_coord = i + 0.5

            if i == j:
                # Diagonal => add recall
                cls_name = cm_df_norm.index[i]
                rec_val = recall_vals.get(cls_name, np.nan)
                if pd.notnull(rec_val):
                    text_str = f"{val:.2f}\nRecall: {rec_val:.2f}"
                else:
                    text_str = f"{val:.2f}\nRecall: N/A"
                text_color = "white"
            else:
                text_str = f"{val:.2f}"
                text_color = "black"

            ax.text(
                x_coord,
                y_coord,
                text_str,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12
            )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Multi-Label Confusion Matrix (Row/Col Order = score_cols)")
    plt.show()

    return cm_df




import statsmodels.api as sm

def plot_logistic_fit_with_cutoffs(data, score_column="Score", 
                                   class_column="Class", truth_column="Truth"):
    """
    Fits logistic regression models to estimate the probability of correct detection 
    as a function of BirdNET confidence score, following Wood & Kahl (2024).
    
    Plots the logistic curve and thresholds for 90%, 95%, and 99% probability.
    
    Parameters:
    - data: DataFrame containing BirdNET evaluation results.
    - score_column: Column with BirdNET confidence scores (0-1).
    - class_column: Column with the predicted class.
    - truth_column: Column with the true class.

    Returns:
    - Logistic regression results for confidence and logit-transformed scores.
    """

    # Copy data to avoid modifying original
    data = data.copy()

    # Create binary column for correct detection
    data["Correct"] = (data[class_column] == data[truth_column]).astype(int)

    # Logit transformation of score (avoiding log(0) by clipping values)
    eps = 1e-9  # Small value to prevent log(0) errors
    data["Logit_Score"] = np.log(np.clip(data[score_column], eps, 1 - eps) / np.clip(1 - data[score_column], eps, 1 - eps))

    # Fit logistic regression models
    conf_model = sm.Logit(data["Correct"],  sm.add_constant(data[score_column])).fit(disp=False)
    logit_model = sm.Logit(data["Correct"], sm.add_constant(data["Logit_Score"])).fit(disp=False)

    # Generate prediction ranges
    conf_range = np.linspace(0.01, 0.99, 1000)  # Confidence score range
    logit_range = np.linspace(data["Logit_Score"].min(), data["Logit_Score"].max(), 1000)

    # Predict probabilities
    conf_pred = conf_model.predict(sm.add_constant(conf_range))
    logit_pred = logit_model.predict(sm.add_constant(logit_range))

    # Compute score cutoffs for 90%, 95%, and 99% probability thresholds
    def find_cutoff(model, coef_index):
        return (np.log([0.90 / 0.10, 0.95 / 0.05, 0.99 / 0.01]) - model.params[0]) / model.params[coef_index]

    conf_cutoffs = find_cutoff(conf_model, 1)
    logit_cutoffs = find_cutoff(logit_model, 1)

    # Plot Confidence Score Model
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=data[score_column], y=data["Correct"], alpha=0.3, label="Observations")
    plt.plot(conf_range, conf_pred, color="red", label="Logistic Fit (Confidence Score)")
    for i, cutoff in enumerate(conf_cutoffs):
        plt.axvline(cutoff, linestyle="--", color=["orange", "red", "magenta"][i], label=f"p={0.9 + i*0.05:.2f}")
    plt.xlabel("BirdNET Confidence Score")
    plt.ylabel("Pr(Correct Detection)")
    plt.title("Logistic Fit: Confidence Score")
    plt.legend()
    plt.grid()

    # Plot Logit Score Model
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=data["Logit_Score"], y=data["Correct"], alpha=0.3, label="Observations")
    plt.plot(logit_range, logit_pred, color="blue", label="Logistic Fit (Logit Score)")
    for i, cutoff in enumerate(logit_cutoffs):
        plt.axvline(cutoff, linestyle="--", color=["orange", "red", "magenta"][i], label=f"p={0.9 + i*0.05:.2f}")
    plt.xlabel("Logit of BirdNET Confidence Score")
    plt.ylabel("Pr(Correct Detection)")
    plt.title("Logistic Fit: Logit Score")
    plt.legend()
    plt.grid()

    plt.show()

    return conf_model, logit_model


def plot_one_vs_others_pr(all_df, relevant_classes=None, class_colors=None,
                          titleStr="One-vs-Others Precision–Recall Curve"):
    """
    Generates a One-vs.-Others Precision–Recall curve for each class in the dataset,
    deriving false positives and true positives internally (no need for fp_df).

    Parameters:
    - all_df: Full dataset with 'Truth', 'Class', and 'Score' columns, where
        * 'Class' is the model's predicted label
        * 'Truth' is the ground-truth label
        * 'Score' is the confidence score for the predicted label
    - relevant_classes: Optional list of class names to include in the comparison.
                       If None, uses all classes in 'all_df'.
    - class_colors: Optional dictionary mapping class names to specific colors.
                    If None, uses Matplotlib's default 'tab10' colormap.
    - titleStr: Title for the Precision–Recall curve plot.

    Returns:
    - pr_data: Dictionary containing thresholds, precision, and recall arrays for each relevant class.
               For example: pr_data[cls] = {
                   'thresholds': [...],
                   'precision': [...],
                   'recall': [...]
               }
    """

    # If no classes specified, use all unique ones from 'Class'
    if relevant_classes is None:
        relevant_classes = np.unique(all_df['Class'])

    # Assign a color to each class if not provided
    default_colors = {
        cls: plt.cm.get_cmap("tab10").colors[i % 10]
        for i, cls in enumerate(relevant_classes)
    }
    color_map = default_colors if class_colors is None else {**default_colors, **class_colors}

    # Define thresholds to sweep over
    #thresholds =np.sort(np.unique(all_df['Score']))
    thresholds =np.linspace(0,1,200)

    # Storage for precision–recall data
    pr_data = {}

    plt.figure(figsize=(8, 6))

    for cls in relevant_classes:
        # Filter out only rows for relevant_classes in the ground truth
        #df_filtered = all_df[all_df['Truth'].isin(relevant_classes)].copy()

        # Define "false-positives" subset: predicted label == cls, but truth != cls
        fpData = all_df[(all_df['Class'] == cls) & 
                             (all_df['Truth'] != cls)].copy()

        # Define "true-positives" subset: truth == cls (we'll check predictions below)
        tpData = all_df[all_df['Truth'] == cls].copy()

        precision_list, recall_list = [], []

        for threshold in thresholds:
            # Mark each row as "Predicted" if Score >= threshold
            # For TP data, require that the model predicted exactly cls
            tpData['Predicted'] = (tpData[cls] >= threshold) 
            # For FP data, the row is already known to have Class=cls but is not truly cls;
            # we just check if Score >= threshold
            fpData['Predicted'] = fpData[cls] >= threshold

            # Counts for this threshold
            true_positive_count = tpData['Predicted'].sum()
            false_positive_count = fpData['Predicted'].sum()
            false_negative_count = len(tpData) - true_positive_count

            # Precision = TP / (TP + FP)
            tp_fp_sum = true_positive_count + false_positive_count
            precision_val = (true_positive_count / tp_fp_sum) if tp_fp_sum > 0 else 1.0

            # Recall = TP / (TP + FN)
            tp_fn_sum = true_positive_count + false_negative_count
            recall_val = (true_positive_count / tp_fn_sum) if tp_fn_sum > 0 else 0.0

            precision_list.append(precision_val)
            recall_list.append(recall_val)

        # Store results
        pr_data[cls] = {
            'thresholds': thresholds,
            'precision': np.array(precision_list),
            'recall': np.array(recall_list)
        }

        # Plot Precision–Recall curve
        plt.plot(recall_list, precision_list, label=cls, color=color_map.get(cls, "black"))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.show()

    return pr_data

def find_cutoff(model, coef_index):
    return (np.log([0.90 / 0.10, 0.95 / 0.05, 0.99 / 0.01]) - model.params[0]) / model.params[coef_index]

#%% Birdnet 6 birdnet Class
########################################################################
# Run birnet trained nn smaller more balanced dataset normalized data

# Similar to birdnet 5 but birdnet five was trained with 15khz and DFO crp
# data were included which are limited to 16khz. So this has been run with 
# a 8khz limit. Data from the background and humback were also better split
# across the providers.
###########################################################################

# Malahat TKW model Loc
malahatTKW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\TKW\\'
# Model and labels
model_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\fix_mn_srkw_offshore_tkw_balanced_4k\\Output\\CustomClassifier_8khz.tflite"
label_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\fix_mn_srkw_offshore_tkw_balanced_4k\\Output\\CustomClassifier_8khz_Labels.txt"


output_csv = "predictions_output.csv"

# Batch process audio files in folder and export to CSV

processor = Eco.BirdNetPredictor(model_path, label_path, malahatTKW_folder)
TKW_testData_birdnet6, Malahat_tkw_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_tkw_birdnet06['Class'] = Malahat_tkw_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_tkw_birdnet06['Truth']= 'TKW'

TKW_testData_birdnet6['FP'] = TKW_testData_birdnet6['Class'] != 'TKW'
TKW_testData_birdnet6['Class'].value_counts()
TKW_testData_birdnet6['Truth'] = 'TKW'


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'UndBio':'#e377c2'}   



#%%
# To show recall

Malahat_tkw_birdnet06['Score'] = Malahat_tkw_birdnet06['TKW']

aa, bb = plot_logistic_fit_with_cutoffs(Malahat_tkw_birdnet06, score_column="TKW", 
                                   class_column="Class", truth_column="Truth")
TKW_90_cutoff = find_cutoff(aa, 1)[0] # 95th percentile 


# Using a dictionary threshold:
custom_thresholds = {
    "TKW": TKW_90_cutoff}


EvalDat  =  Malahat_tkw_birdnet06


plot_one_vs_others_pr(EvalDat,
                      relevant_classes=['SRKW', 'TKW', 'HW'], class_colors=None,
                          titleStr="One-vs-Others Precision–Recall Curve")

cm_df = plot_confusion_matrix(EvalDat, threshold=0)

cm_df = plot_confusion_matrix(EvalDat, threshold=custom_thresholds)

cm_df = plot_confusion_matrix_multilabel(
    data = EvalDat,
    score_cols=["SRKW", "TKW", "HW"],
    truth_column="Truth",
    threshold=custom_thresholds)


EvalDat['FP'] =  EvalDat['Truth'] != EvalDat['Class']
FPData_EvalDat = EvalDat[EvalDat['FP'] == True]
plot_one_vs_others_roc(FPData_EvalDat,  
                                     EvalDat,
                                     titleStr= "One-vs-Others ROC Birdnet 6", 
                                     class_colors= class_colors)


cm_df = plot_confusion_matrix(EvalDat, threshold=custom_thresholds)




