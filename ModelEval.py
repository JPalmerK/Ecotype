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


#%%

# Models and parameters
mod_1 = load_model('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\HumpbackBalanced_8khz\\Output\\Resnet18\\output_BalancedMn_8khz_MnBalanced_8khz_Resnet18_RTO_batchNorm.keras')

AudioParms_mod1 = {
           'clipDur': 3,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':256,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None, 
            'PCEN': True,
            'fmin': 0,
            'min_freq': None,       # default minimum frequency to retain
            'spec_power':2,
            'returnDB':True,         # return spectrogram in linear or convert to db 
            'PCEN_power':31,
            'time_constant':.8,
            'eps':1e-6,
            'gain':0.08,
            'power':.25,
            'bias':10,
            'fmax':16000,
            'Scale Spectrogram': False} # scale the spectrogram between 0 and 1
# Create the batch loader instances for streaming data from GCS


# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 0.25,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 0.25,  # Example threshold for class 5
}


class_names={0: 'AB', 1: 'HW', 2: 'RKW', 3: 'OKW', 4: 'TKW', 5: 'UndBio'}


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
RKW_testData = fasterProcesser.get_detections()


folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\TKW'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
TKW_testData = fasterProcesser.get_detections()




# Figure out the false positives
MN_testData['FP'] = MN_testData['Class'] != 'HW'
MN_testData['Class'].value_counts()
MN_testData['Truth'] = 'HW' 


AB_testData['FP'] = AB_testData['Class'] != 'AB'
AB_testData['Truth'] = 'AB' 

UndBio_testData['FP'] = UndBio_testData['Class'] != 'UndBio'
UndBio_testData['Class'].value_counts()
UndBio_testData['Truth'] = 'UndBio' 

RKW_testData['FP'] = RKW_testData['Class'] != 'RKW'
RKW_testData['Class'].value_counts()
RKW_testData['Truth'] = 'RKW' 


TKW_testData['FP'] = TKW_testData['Class'] != 'TKW'
TKW_testData['Class'].value_counts()
TKW_testData['Truth'] = 'TKW' 

ALLData = pd.concat([MN_testData, AB_testData,UndBio_testData, RKW_testData,
                     TKW_testData])
FPData = ALLData[ALLData['FP'] == True]



import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData['Class'].unique()}

# For each class and threshold, calculate the number of false positives
for cls in FPData['Class'].unique():
    df_class = FPData[FPData['Class'] == cls]
    false_positives = []
    
    for threshold in thresholds:
        # Apply the threshold to classify predictions
        df_class['Predicted'] = df_class['Score'] >= threshold
        # Calculate false positives (Predicted = True but True_Label = Negative)
        false_positives_count = len(df_class[(df_class['Predicted'] == True)])
        false_positives.append(false_positives_count)
    
    false_positives_by_class[cls] = false_positives

# Plotting the number of false positives for each class as a function of the threshold
plt.figure(figsize=(10, 6))
for cls, false_positives in false_positives_by_class.items():
    sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# Add labels and title
plt.xlabel("Threshold Score")
plt.ylabel("Number of False Positives")
plt.title("False Positives as a Function of Threshold Score by Class")
plt.legend(title="Class")
plt.grid(True)
plt.show()



roc_results = plot_one_vs_others_roc(FPData, ALLData, 
                                     titleStr= "One-vs-Others ROC Model 1")



class_colors = {
    'RKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   

    
relevant_classes = ['RKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData, ALLData, 
                                     relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Model 1", 
                                     class_colors= class_colors)


#pr_results = plot_one_vs_relevant_others_pr(FPData, ALLData, relevant_classes)



#%%
##############################################################################
# Resnet 50
##############################################################################


# Models and parameters
mod_2 = load_model('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\HumpbackBalanced_16khz\\output\\Resnet50\\output_BalancedMn_16khz_Resnet50_MnBalanced_16khz_Resnet50_20241228_042445.keras')
                   
                   
AudioParms_mod2 = {
    'clipDur': 3,
    'outSR': 16000,
    'nfft': 1024,
    'hop_length': 102,
    'spec_type': 'mel',
    'spec_power': 2,
    'rowNorm': False,
    'colNorm': False,
    'rmDCoffset': False,
    'inSR': None,
    'PCEN': True,
    'fmin': 0,
    'min_freq': None,
    'returnDB': True,
    'PCEN_power': 31,
    'time_constant': 0.8,
    'eps': 1e-6,
    'gain': 0.08,
    'power': 0.25,
    'bias': 10,
    'fmax': 16000,
    'Scale Spectrogram': False}

# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 2,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 2,  # Example threshold for class 5
}


class_names={0: 'AB', 1: 'HW', 2: 'RKW', 3: 'OKW', 4: 'TKW', 5: 'UndBio'}


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData_mod2 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData_mod2 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData_mod2 = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
RKW_testData_mod2 = fasterProcesser.get_detections()


folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\TKW'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
TKW_testData_mod2 = fasterProcesser.get_detections()



# Figure out the false positives
MN_testData_mod2['FP'] = MN_testData_mod2['Class'] != 'HW'
MN_testData_mod2['Class'].value_counts()
MN_testData_mod2['Truth'] = 'HW' 

AB_testData_mod2['FP'] = AB_testData_mod2['Class'] != 'AB'
AB_testData_mod2['Truth'] = 'AB' 

UndBio_testData_mod2['FP'] = UndBio_testData_mod2['Class'] != 'UndBio'
UndBio_testData_mod2['Class'].value_counts()
UndBio_testData_mod2['Truth'] = 'UndBio' 

RKW_testData_mod2['FP'] = RKW_testData_mod2['Class'] != 'SRKW'
RKW_testData_mod2['Class'].value_counts()
RKW_testData_mod2['Truth'] = 'SRKW' 

TKW_testData_mod2['FP'] = TKW_testData_mod2['Class'] != 'TKW'
TKW_testData_mod2['Class'].value_counts()
TKW_testData_mod2['Truth'] = 'TKW' 



ALLData_mod2 = pd.concat([MN_testData_mod2, AB_testData_mod2,
                     UndBio_testData_mod2, RKW_testData_mod2, 
                     TKW_testData_mod2])
FPData_mod2 = ALLData_mod2[ALLData_mod2['FP'] == True]



import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_mod2['Class'].unique()}

# For each class and threshold, calculate the number of false positives
for cls in FPData_mod2['Class'].unique():
    df_class = FPData_mod2[FPData_mod2['Class'] == cls]
    false_positives = []
    
    for threshold in thresholds:
        # Apply the threshold to classify predictions
        df_class['Predicted'] = df_class['Score'] >= threshold
        # Calculate false positives (Predicted = True but True_Label = Negative)
        false_positives_count = len(df_class[(df_class['Predicted'] == True)])
        false_positives.append(false_positives_count)
    
    false_positives_by_class[cls] = false_positives

# Plotting the number of false positives for each class as a function of the threshold
plt.figure(figsize=(10, 6))
for cls, false_positives in false_positives_by_class.items():
    sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# Add labels and title
plt.xlabel("Threshold Score")
plt.ylabel("Number of False Positives")
plt.title("False Positives as a Function of Threshold Score by Class")
plt.legend(title="Class")
plt.grid(True)
plt.show()





class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   

    
relevant_classes = ['SRKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData_mod2, ALLData_mod2, 
                                     relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Model 2", 
                                     class_colors= class_colors)


#%%
##############################################################################
# Resnet 50 8 khz
##############################################################################


# Models and parameters
mod_3 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced_8khz\\Output\\Resnet50\\MnBalanced_8khz__10fmin_1024fft_PCEN_RTW_batchNormResnet50.keras')
                   
                   
AudioParms_mod3 = {
            'clipDur': 3,
            'outSR': 8000,
            'nfft': 1024,
            'hop_length':102,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': True,
            'inSR': None, 
            'PCEN': True,
            'fmin': 10,
            'min_freq': None,       # default minimum frequency to retain
            'spec_power':2,
            'returnDB':True,         # return spectrogram in linear or convert to db 
            'PCEN_power':31,
            'time_constant':.8,
            'eps':1e-6,
            'gain':0.08,
            'power':.25,
            'bias':10,
            'fmax':8000,
            'Scale Spectrogram': False,
            'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
            'Excluding UAF data'} # scale the spectrogram between 0 and 1


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod3, 
    model=mod_3, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData_mod3 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod3, 
    model=mod_3, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData_mod3 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod3, 
    model=mod_3, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData_mod3 = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod3, 
    model=mod_3, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
RKW_testData_mod3 = fasterProcesser.get_detections()


folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\TKW'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod3, 
    model=mod_3, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
TKW_testData_mod3 = fasterProcesser.get_detections()



# Figure out the false positives
MN_testData_mod3['FP'] = MN_testData_mod3['Class'] != 'HW'
MN_testData_mod3['Class'].value_counts()
MN_testData_mod3['Truth'] = 'HW' 


AB_testData_mod3['FP'] = AB_testData_mod3['Class'] != 'AB'
AB_testData_mod3['Truth'] = 'AB' 

UndBio_testData_mod3['FP'] = UndBio_testData_mod3['Class'] != 'UndBio'
UndBio_testData_mod3['Class'].value_counts()
UndBio_testData_mod3['Truth'] = 'UndBio' 

RKW_testData_mod3['FP'] = RKW_testData_mod3['Class'] != 'SRKW'
RKW_testData_mod3['Class'].value_counts()
RKW_testData_mod3['Truth'] = 'SRKW' 

TKW_testData_mod3['FP'] = TKW_testData_mod3['Class'] != 'TKW'
TKW_testData_mod3['Class'].value_counts()
TKW_testData_mod3['Truth'] = 'TKW' 



ALLData_mod3 = pd.concat([MN_testData_mod3, AB_testData_mod3,
                     UndBio_testData_mod3, RKW_testData_mod3, 
                     TKW_testData_mod3])
FPData_mod3 = ALLData_mod3[ALLData_mod3['FP'] == True]




# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_mod3['Class'].unique()}

# For each class and threshold, calculate the number of false positives
for cls in FPData_mod3['Class'].unique():
    df_class = FPData_mod3[FPData_mod3['Class'] == cls]
    false_positives = []
    
    for threshold in thresholds:
        # Apply the threshold to classify predictions
        df_class['Predicted'] = df_class['Score'] >= threshold
        # Calculate false positives (Predicted = True but True_Label = Negative)
        false_positives_count = len(df_class[(df_class['Predicted'] == True)])
        false_positives.append(false_positives_count)
    
    false_positives_by_class[cls] = false_positives

# Plotting the number of false positives for each class as a function of the threshold
plt.figure(figsize=(10, 6))
for cls, false_positives in false_positives_by_class.items():
    sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# Add labels and title
plt.xlabel("Threshold Score")
plt.ylabel("Number of False Positives")
plt.title("False Positives as a Function of Threshold Score by Class")
plt.legend(title="Class")
plt.grid(True)
plt.show()



class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   

    
relevant_classes = ['SRKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData_mod3, ALLData_mod3,
                                     relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Model 3", 
                                     class_colors= class_colors)


#%% Model 4, trained with 1500 example subset of data. Class augmetntation 
##############################################################################
# Resnet 50 15 khz, smaller dataset
##############################################################################


# Models and parameters
mod_4 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\SmallerSetONCfixed_15khz\\MnBalanced_15khz_512fft_PCEN_RTW_batchNormResnet50.keras')
                   
                   
AudioParms_mod4 = {
            'clipDur': 3,
            'outSR': 15000,
            'nfft': 512,
            'hop_length':51,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': True,
            'inSR': None, 
            'PCEN': True,
            'fmin': 0,
            'min_freq': None,       # default minimum frequency to retain
            'spec_power':2,
            'returnDB':True,         # return spectrogram in linear or convert to db 
            'PCEN_power':31,
            'time_constant':.8,
            'eps':1e-6,
            'gain':0.08,
            'power':.25,
            'bias':10,
            'fmax':15000,
            'Scale Spectrogram': True} # scale the spectrogram between 0 and 1



class_names={0: 'NRKW',  1: 'OKW', 2: 'SRKW', 3: 'TKW', 4: 'HW', 5: 'AB',6: 'UndBio'}

# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 0.25,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 0.25,  # Example threshold for class 5
    6: 0.25,  # Example threshold for class 6

}


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData_mod4 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData_mod4 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData_mod4 = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="MOD04_SRKW_Test.txt")
fasterProcesser.process_all_files()
RKW_testData_mod4 = fasterProcesser.get_detections()


folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\TKW'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="MOD04_TKW_Test.txt")
fasterProcesser.process_all_files()
TKW_testData_mod4 = fasterProcesser.get_detections()

folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\RKW\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="MOD04_SRKW_Test.txt")
fasterProcesser.process_all_files()
RKW_testData_MalahatSub_mod4 = fasterProcesser.get_detections()




# Figure out the false positives
MN_testData_mod4['FP'] = MN_testData_mod4['Class'] != 'HW'
MN_testData_mod4['Class'].value_counts()
MN_testData_mod4['Truth'] = 'HW' 

AB_testData_mod4['FP'] = AB_testData_mod4['Class'] != 'AB'
AB_testData_mod4['Truth'] = 'AB' 
AB_testData_mod4['Truth'] = 'AB' 

UndBio_testData_mod4['FP'] = UndBio_testData_mod4['Class'] != 'UndBio'
UndBio_testData_mod4['Class'].value_counts()
UndBio_testData_mod4['Truth'] = 'UndBio' 


RKW_testData_mod4['FP'] = RKW_testData_mod4['Class'] != 'SRKW'
RKW_testData_mod4['Class'].value_counts()
RKW_testData_mod4['Truth'] = 'SRKW' 


TKW_testData_mod4['FP'] = TKW_testData_mod4['Class'] != 'TKW'
TKW_testData_mod4['Class'].value_counts()
TKW_testData_mod4['Truth'] = 'TKW' 

RKW_testData_MalahatSub_mod4['FP'] = RKW_testData_MalahatSub_mod4['Class'] != 'SRKW'
RKW_testData_MalahatSub_mod4['Class'].value_counts()
RKW_testData_MalahatSub_mod4['Truth'] = 'SRKW' 



ALLData_mod4 = pd.concat([MN_testData_mod4, AB_testData_mod4,
                     UndBio_testData_mod4, RKW_testData_mod4, 
                     RKW_testData_MalahatSub_mod4,
                     TKW_testData_mod4])
FPData_mod4 = ALLData_mod4[ALLData_mod4['FP'] == True]



import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# # Prepare a list to hold false positive counts for each class
# false_positives_by_class = {cls: [] for cls in FPData_mod2['Class'].unique()}

# # For each class and threshold, calculate the number of false positives
# for cls in FPData_mod4['Class'].unique():
#     df_class = FPData_mod4[FPData_mod4['Class'] == cls]
#     false_positives = []
    
#     for threshold in thresholds:
#         # Apply the threshold to classify predictions
#         df_class['Predicted'] = df_class['Score'] >= threshold
#         # Calculate false positives (Predicted = True but True_Label = Negative)
#         false_positives_count = len(df_class[(df_class['Predicted'] == True)])
#         false_positives.append(false_positives_count)
    
#     false_positives_by_class[cls] = false_positives

# # Plotting the number of false positives for each class as a function of the threshold
# plt.figure(figsize=(10, 6))
# for cls, false_positives in false_positives_by_class.items():
#     sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# # Add labels and title
# plt.xlabel("Threshold Score")
# plt.ylabel("Number of False Positives")
# plt.title("False Positives as a Function of Threshold Score by Class")
# plt.legend(title="Class")
# plt.grid(True)
# plt.show()





class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   

    
relevant_classes = ['SRKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData_mod4, ALLData_mod4,
                                     relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Resnet Model 4", 
                                     class_colors= class_colors)


#%% Model 5, trained with 1500 example subset of data not Mel scaled. Class augmetntation 
##############################################################################
# Resnet 50 15 khz, smaller dataset
##############################################################################


# Models and parameters
mod_4 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\SmallerSetONCfixed_15khz\\MnBalanced_15khz_512fft_PCEN_RTW_batchNormResnet50.keras')
                   
                   
AudioParms_mod4 = {
            'clipDur': 3,
            'outSR': 15000,
            'nfft': 512,
            'hop_length':51,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': True,
            'inSR': None, 
            'PCEN': True,
            'fmin': 0,
            'min_freq': None,       # default minimum frequency to retain
            'spec_power':2,
            'returnDB':True,         # return spectrogram in linear or convert to db 
            'PCEN_power':31,
            'time_constant':.8,
            'eps':1e-6,
            'gain':0.08,
            'power':.25,
            'bias':10,
            'fmax':15000,
            'Scale Spectrogram': True} # scale the spectrogram between 0 and 1



class_names={0: 'NRKW',  1: 'OKW', 2: 'SRKW', 3: 'TKW', 4: 'HW', 5: 'AB',6: 'UndBio'}

# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 0.25,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 0.25,  # Example threshold for class 5
    6: 0.25,  # Example threshold for class 6

}


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData_mod4 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData_mod4 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData_mod4 = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="MOD04_SRKW_Test.txt")
fasterProcesser.process_all_files()
RKW_testData_mod4 = fasterProcesser.get_detections()


folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\TKW'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="MOD04_TKW_Test.txt")
fasterProcesser.process_all_files()
TKW_testData_mod4 = fasterProcesser.get_detections()

folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\RKW\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod4['clipDur'], 
    overlap=0, 
    params=AudioParms_mod4, 
    model=mod_4, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="MOD04_SRKW_Test.txt")
fasterProcesser.process_all_files()
RKW_testData_MalahatSub_mod4 = fasterProcesser.get_detections()




# Figure out the false positives
MN_testData_mod4['FP'] = MN_testData_mod4['Class'] != 'HW'
MN_testData_mod4['Class'].value_counts()
MN_testData_mod4['Truth'] = 'HW' 

AB_testData_mod4['FP'] = AB_testData_mod4['Class'] != 'AB'
AB_testData_mod4['Truth'] = 'AB' 
AB_testData_mod4['Truth'] = 'AB' 

UndBio_testData_mod4['FP'] = UndBio_testData_mod4['Class'] != 'UndBio'
UndBio_testData_mod4['Class'].value_counts()
UndBio_testData_mod4['Truth'] = 'UndBio' 


RKW_testData_mod4['FP'] = RKW_testData_mod4['Class'] != 'SRKW'
RKW_testData_mod4['Class'].value_counts()
RKW_testData_mod4['Truth'] = 'SRKW' 


TKW_testData_mod4['FP'] = TKW_testData_mod4['Class'] != 'TKW'
TKW_testData_mod4['Class'].value_counts()
TKW_testData_mod4['Truth'] = 'TKW' 

RKW_testData_MalahatSub_mod4['FP'] = RKW_testData_MalahatSub_mod4['Class'] != 'SRKW'
RKW_testData_MalahatSub_mod4['Class'].value_counts()
RKW_testData_MalahatSub_mod4['Truth'] = 'SRKW' 



ALLData_mod4 = pd.concat([MN_testData_mod4, AB_testData_mod4,
                     UndBio_testData_mod4, RKW_testData_mod4, 
                     RKW_testData_MalahatSub_mod4,
                     TKW_testData_mod4])
FPData_mod4 = ALLData_mod4[ALLData_mod4['FP'] == True]



import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# # Prepare a list to hold false positive counts for each class
# false_positives_by_class = {cls: [] for cls in FPData_mod2['Class'].unique()}

# # For each class and threshold, calculate the number of false positives
# for cls in FPData_mod4['Class'].unique():
#     df_class = FPData_mod4[FPData_mod4['Class'] == cls]
#     false_positives = []
    
#     for threshold in thresholds:
#         # Apply the threshold to classify predictions
#         df_class['Predicted'] = df_class['Score'] >= threshold
#         # Calculate false positives (Predicted = True but True_Label = Negative)
#         false_positives_count = len(df_class[(df_class['Predicted'] == True)])
#         false_positives.append(false_positives_count)
    
#     false_positives_by_class[cls] = false_positives

# # Plotting the number of false positives for each class as a function of the threshold
# plt.figure(figsize=(10, 6))
# for cls, false_positives in false_positives_by_class.items():
#     sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# # Add labels and title
# plt.xlabel("Threshold Score")
# plt.ylabel("Number of False Positives")
# plt.title("False Positives as a Function of Threshold Score by Class")
# plt.legend(title="Class")
# plt.grid(True)
# plt.show()





class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   

    
relevant_classes = ['SRKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData_mod4, ALLData_mod4,
                                     relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Resnet Model 4", 
                                     class_colors= class_colors)
#%% Birdnet 1
#########################################################################
# Run birnet trained nn
###########################################################################


# Humpback data 
MN_testData_birdNet1 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\KWs_nonKW\\FPAnalysis\\Humpback_BirdNET_CombinedTable.csv')
AB_testData_birdNet1 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\KWs_nonKW\\FPAnalysis\\AbioticBirdNET_CombinedTable.csv')
TKW_testData_birdNet1 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\KWs_nonKW\\FPAnalysis\\TKW_BirdNET_CombinedTable.csv')
RKW_testData_birdNet1 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\KWs_nonKW\\FPAnalysis\\RKW_BirdNET_CombinedTable.csv')
UNDBIO_testData_birdNet1 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\KWs_nonKW\\FPAnalysis\\UnkBio_BirdNET_CombinedTable.csv')
RKW_testData_Malahat_birdNet1 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\UnkBio_BirdNET_CombinedTable\\FPAnalysis\\ONCFix_RKWmalahatSubset_BirdNET_CombinedTable.csv')



# Check previous colname
FPData_mod2.columns
MN_testData_birdNet1.columns



MN_testData_birdNet1.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
AB_testData_birdNet1.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
TKW_testData_birdNet1.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
RKW_testData_birdNet1.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)

RKW_testData_birdNet1.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)

UNDBIO_testData_birdNet1.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)



# Figure out the false positives
MN_testData_birdNet1['FP'] = MN_testData_birdNet1['Class'] != 'Negative'
MN_testData_birdNet1['Class'].value_counts()
MN_testData_birdNet1['Truth'] = 'HW' 

AB_testData_birdNet1['FP'] = AB_testData_birdNet1['Class'] != 'Negative'
AB_testData_birdNet1['Class'].value_counts()
AB_testData_birdNet1['Truth'] = 'Negative' 


TKW_testData_birdNet1['FP'] = TKW_testData_birdNet1['Class'] != 'TKW'
TKW_testData_birdNet1['Class'].value_counts()
TKW_testData_birdNet1['Truth'] = 'TKW' 

RKW_testData_birdNet1['FP'] = RKW_testData_birdNet1['Class'] != 'RKW'
RKW_testData_birdNet1['Class'].value_counts()
RKW_testData_birdNet1['Truth'] = 'RKW' 

UNDBIO_testData_birdNet1['FP'] = UNDBIO_testData_birdNet1['Class'] != 'Negative'
UNDBIO_testData_birdNet1['Class'].value_counts()
UNDBIO_testData_birdNet1['Truth'] = 'Negative' 


ALLData_birdnet01 = pd.concat([MN_testData_birdNet1, AB_testData_mod2,
                     UNDBIO_testData_birdNet1, RKW_testData_birdNet1, 
                     TKW_testData_birdNet1])
FPData_birdnet01 = ALLData_birdnet01[ALLData_birdnet01['FP'] == True]


import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_birdnet01['Class'].unique()}

# For each class and threshold, calculate the number of false positives
for cls in FPData_birdnet01['Class'].unique():
    df_class = FPData_birdnet01[FPData_birdnet01['Class'] == cls]
    false_positives = []
    
    for threshold in thresholds:
        # Apply the threshold to classify predictions
        df_class['Predicted'] = df_class['Score'] >= threshold
        # Calculate false positives (Predicted = True but True_Label = Negative)
        false_positives_count = len(df_class[(df_class['Predicted'] == True)])
        false_positives.append(false_positives_count)
    
    false_positives_by_class[cls] = false_positives

# Plotting the number of false positives for each class as a function of the threshold
plt.figure(figsize=(10, 6))
for cls, false_positives in false_positives_by_class.items():
    sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# Add labels and title
plt.xlabel("Threshold Score")
plt.ylabel("Number of False Positives")
plt.title("False Positives as a Function of Threshold Score by Class")
plt.legend(title="Class")
plt.grid(True)
plt.show()

roc_results = plot_one_vs_others_roc(FPData_birdnet01, ALLData_birdnet01)


relevant_classes = ['RKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData_birdnet01, ALLData_birdnet01,
                                     relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 1", 
                                     class_colors= class_colors)


#%% Birdnet 2
########################################################################
# Run birnet trained nn smaller more balanced dataset normalized data
###########################################################################

class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   

# Humpback data 
MN_testData_birdNet2 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet5Class_ONC_updates\\FPAnalysis\\ONCFix_HW_BirdNET_CombinedTable.csv')
AB_testData_birdNet2 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet5Class_ONC_updates\\FPAnalysis\\ONCFix_Abiotic_BirdNET_CombinedTable.csv')
TKW_testData_birdNet2 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet5Class_ONC_updates\\FPAnalysis\\ONCFix_TKWmalahat_BirdNET_CombinedTable.csv')
RKW_testData_birdNet2 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet5Class_ONC_updates\\FPAnalysis\\ONCFix_RKW_BirdNET_CombinedTable.csv')
UNDBIO_testData_birdNet2 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet5Class_ONC_updates\\FPAnalysis\\ONCFix_Unkbio_BirdNET_CombinedTable.csv')
RKW_testData_Malahat_birdNet2 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet5Class_ONC_updates\\FPAnalysis\\ONCFix_RKWmalahatSubset_BirdNET_CombinedTable.csv')


MN_testData_birdNet2.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
AB_testData_birdNet2.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
TKW_testData_birdNet2.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
RKW_testData_birdNet2.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)

RKW_testData_Malahat_birdNet2.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)

UNDBIO_testData_birdNet2.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)



# Figure out the false positives
MN_testData_birdNet2['FP'] = MN_testData_birdNet2['Class'] != 'HW'
MN_testData_birdNet2['Class'].value_counts()
MN_testData_birdNet2['Truth'] = 'HW' 

AB_testData_birdNet2['FP'] = AB_testData_birdNet2['Class'] != 'AB'
AB_testData_birdNet2['Class'].value_counts()
AB_testData_birdNet2['Truth'] = 'AB'

TKW_testData_birdNet2['FP'] = TKW_testData_birdNet2['Class'] != 'TKW'
TKW_testData_birdNet2['Class'].value_counts()
TKW_testData_birdNet2['Truth'] = 'TKW'

RKW_testData_birdNet2['FP'] = RKW_testData_birdNet2['Class'] != 'SRKW'
RKW_testData_birdNet2['Class'].value_counts()
RKW_testData_birdNet2['Truth'] = 'SRKW'


RKW_testData_Malahat_birdNet2['FP'] = RKW_testData_Malahat_birdNet2['Class'] != 'SRKW'
RKW_testData_Malahat_birdNet2['Class'].value_counts()
RKW_testData_Malahat_birdNet2['Truth'] = 'SRKW'



UNDBIO_testData_birdNet2['FP'] = UNDBIO_testData_birdNet2['Class'] != 'UndBio'
UNDBIO_testData_birdNet2['Class'].value_counts()
UNDBIO_testData_birdNet2['Truth'] = 'UndBio'

ALLData_birdnet02 = pd.concat([MN_testData_birdNet2, AB_testData_birdNet2,
                     UNDBIO_testData_birdNet2, 
                     RKW_testData_Malahat_birdNet2, 
                     RKW_testData_birdNet2,
                     TKW_testData_birdNet2])

FPData_birdnet02 = ALLData_birdnet02[ALLData_birdnet02['FP'] == True]


import matplotlib.pyplot as plt
import seaborn as sns


# # Set a range of threshold scores
# thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# # Prepare a list to hold false positive counts for each class
# false_positives_by_class = {cls: [] for cls in FPData_birdnet02['Class'].unique()}

# # For each class and threshold, calculate the number of false positives
# for cls in ALLData_birdnet02['Class'].unique():
#     df_class = FPData_birdnet02[FPData_birdnet02['Class'] == cls]
#     false_positives = []
    
#     for threshold in thresholds:
#         # Apply the threshold to classify predictions
#         df_class['Predicted'] = df_class['Score'] >= threshold
#         # Calculate false positives (Predicted = True but True_Label = Negative)
#         false_positives_count = len(df_class[(df_class['Predicted'] == True)])
#         false_positives.append(false_positives_count)
    
#     false_positives_by_class[cls] = false_positives

# # Plotting the number of false positives for each class as a function of the threshold
# plt.figure(figsize=(10, 6))
# for cls, false_positives in false_positives_by_class.items():
#     sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# # Add labels and title
# plt.xlabel("Threshold Score")
# plt.ylabel("Number of False Positives")
# plt.title("False Positives as a Function of Threshold Score by Class")
# plt.legend(title="Class")
# plt.grid(True)
# plt.show()


relevant_classes = ['SRKW', 'TKW']
roc_results = plot_one_vs_others_roc(FPData_birdnet02, ALLData_birdnet02,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 2", 
                                     class_colors= class_colors)
#%% Birdnet 3
########################################################################
# Run birnet trained nn smaller more balanced dataset normalized data
###########################################################################


# Humpback data 
MN_testData_birdNet3 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet7Class_ONC_Larger\\FPAnalysis\\ONCFix_HW_BirdNET_CombinedTable.csv')
AB_testData_birdNet3 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet7Class_ONC_Larger\\FPAnalysis\\ONCFix_Abiotic_BirdNET_CombinedTable.csv')
TKW_testData_birdNet3 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet7Class_ONC_Larger\\FPAnalysis\\ONCFix_TKWmalahat_BirdNET_CombinedTable.csv')
RKW_testData_birdNet3 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet7Class_ONC_Larger\\FPAnalysis\\ONCFix_RKW_BirdNET_CombinedTable.csv')
UNDBIO_testData_birdNet3 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet7Class_ONC_Larger\\FPAnalysis\\ONCFix_Unkbio_BirdNET_CombinedTable.csv')
RKW_testData_Malahat_birdNet3 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Birdnet7Class_ONC_Larger\\FPAnalysis\\ONCFix_RKWmalahatSubset_BirdNET_CombinedTable.csv')




MN_testData_birdNet3.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
AB_testData_birdNet3.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
TKW_testData_birdNet3.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
RKW_testData_birdNet3.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
RKW_testData_Malahat_birdNet3.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
UNDBIO_testData_birdNet3.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)



# Figure out the false positives

MN_testData_birdNet3['FP'] = MN_testData_birdNet3['Class'] != 'HW'
MN_testData_birdNet3['Class'].value_counts()
MN_testData_birdNet3['Truth'] = 'HW' 

AB_testData_birdNet3['FP'] = AB_testData_birdNet3['Class'] != 'AB'
AB_testData_birdNet3['Class'].value_counts()
AB_testData_birdNet3['Truth'] = 'AB'


TKW_testData_birdNet3['FP'] = TKW_testData_birdNet3['Class'] != 'TKW'
TKW_testData_birdNet3['Class'].value_counts()
TKW_testData_birdNet3['Truth'] = 'TKW'


RKW_testData_birdNet3['FP'] = RKW_testData_birdNet3['Class'] != 'SRKW'
RKW_testData_birdNet3['Class'].value_counts()
RKW_testData_birdNet3['Truth'] = 'SRKW'


RKW_testData_Malahat_birdNet3['FP'] = RKW_testData_Malahat_birdNet3['Class'] != 'SRKW'
RKW_testData_Malahat_birdNet3['Class'].value_counts()
RKW_testData_Malahat_birdNet3['Truth'] = 'SRKW'

UNDBIO_testData_birdNet3['FP'] = UNDBIO_testData_birdNet3['Class'] != 'UndBio'
UNDBIO_testData_birdNet3['Class'].value_counts()
UNDBIO_testData_birdNet3['Truth'] = 'UndBio'


ALLData_birdnet03 = pd.concat([MN_testData_birdNet3, AB_testData_birdNet3,
                     UNDBIO_testData_birdNet3, RKW_testData_Malahat_birdNet3, 
                     RKW_testData_birdNet3, TKW_testData_birdNet3])

FPData_birdnet03 = ALLData_birdnet03[ALLData_birdnet03['FP'] == True]


import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_birdnet03['Class'].unique()}


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'AB': '#d62728',# Red
    'OKW':'#9467bd', 
    'UndBio':'#e377c2'}   


relevant_classes = ['SRKW', 'TKW']
roc_results_03 = plot_one_vs_others_roc(FPData_birdnet03,  
                                     ALLData_birdnet03,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 3", 
                                     class_colors= class_colors)

# Example usage: Plot confusion matrix for ALLData_birdnet04
cm_df = plot_confusion_matrix(ALLData_birdnet03, threshold=0.5)


#%% Birdnet 4
########################################################################
# Run birnet trained nn smaller more balanced dataset normalized data

# 6k annotations in SRKW, OKW, TKW, MN and backgaround. Audio clips were
# als 48k, not 16k/ 
###########################################################################


# Humpback data 
MN_testData_birdNet4 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Backgrd_mn_srkw_tkw_okw_6k\\\\FPAnalysis\\ONCFix_HW_BirdNET_CombinedTable.csv')
AB_testData_birdNet4 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Backgrd_mn_srkw_tkw_okw_6k\\FPAnalysis\\ONCFix_Abiotic_BirdNET_CombinedTable.csv')
TKW_testData_birdNet4 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Backgrd_mn_srkw_tkw_okw_6k\\FPAnalysis\\ONCFix_TKWmalahat_BirdNET_CombinedTable.csv')
RKW_testData_birdNet4 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Backgrd_mn_srkw_tkw_okw_6k\\FPAnalysis\\ONCFix_RKW_BirdNET_CombinedTable.csv')
UNDBIO_testData_birdNet4 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Backgrd_mn_srkw_tkw_okw_6k\\FPAnalysis\\ONCFix_Unkbio_BirdNET_CombinedTable.csv')
RKW_testData_Malahat_birdNet4 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\Backgrd_mn_srkw_tkw_okw_6k\\FPAnalysis\\ONCFix_RKWmalahatSubset_BirdNET_CombinedTable.csv')




MN_testData_birdNet4.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
AB_testData_birdNet4.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
TKW_testData_birdNet4.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
RKW_testData_birdNet4.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
RKW_testData_Malahat_birdNet4.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)
UNDBIO_testData_birdNet4.rename(columns={
    'Start (s)':'Begin Time (S)',
    'End (s)': 'End Time (S)',
    'Scientific name': 'Class',
    'Common name':'Common name',
    'Confidence': 'Score'}, inplace=True)



# Figure out the false positives

MN_testData_birdNet4['FP'] = MN_testData_birdNet4['Class'] != 'HW'
MN_testData_birdNet4['Class'].value_counts()
MN_testData_birdNet4['Truth'] = 'HW' 

AB_testData_birdNet4['FP'] = AB_testData_birdNet4['Class'] != 'Background'
AB_testData_birdNet4['Class'].value_counts()
AB_testData_birdNet4['Truth'] = 'Background'


TKW_testData_birdNet4['FP'] = TKW_testData_birdNet4['Class'] != 'TKW'
TKW_testData_birdNet4['Class'].value_counts()
TKW_testData_birdNet4['Truth'] = 'TKW'


RKW_testData_birdNet4['FP'] = RKW_testData_birdNet4['Class'] != 'SRKW'
RKW_testData_birdNet4['Class'].value_counts()
RKW_testData_birdNet4['Truth'] = 'SRKW'


RKW_testData_Malahat_birdNet4['FP'] = RKW_testData_Malahat_birdNet4['Class'] != 'SRKW'
RKW_testData_Malahat_birdNet4['Class'].value_counts()
RKW_testData_Malahat_birdNet4['Truth'] = 'SRKW'

UNDBIO_testData_birdNet4['FP'] = UNDBIO_testData_birdNet4['Class'] != 'Background'
UNDBIO_testData_birdNet4['Class'].value_counts()
UNDBIO_testData_birdNet4['Truth'] = 'Background'


ALLData_birdnet04 = pd.concat([MN_testData_birdNet4, AB_testData_birdNet4,
                     UNDBIO_testData_birdNet4, RKW_testData_Malahat_birdNet4, 
                     RKW_testData_birdNet4, TKW_testData_birdNet4])

FPData_birdnet04 = ALLData_birdnet04[ALLData_birdnet04['FP'] == True]


import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_birdnet04['Class'].unique()}


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'UndBio':'#e377c2'}   


relevant_classes = ['SRKW', 'TKW']
roc_results_04 = plot_one_vs_others_roc(FPData_birdnet04,  
                                     ALLData_birdnet04,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 4", 
                                     class_colors= class_colors)

# Example usage: Plot confusion matrix for ALLData_birdnet04
cm_df = plot_confusion_matrix(ALLData_birdnet04, threshold=0.5)


#%% Birdnet 5 birdnet Class
########################################################################
# Run birnet trained nn smaller more balanced dataset normalized data

# 4.5k annotations in SRKW, OKW, TKW, MN and backgaround. Audio clips were
# als 48k, not 16k/ At least 100 of each call types incldued random sampling to 
# boost to 4.5k
###########################################################################

# Malahat TKW model Loc
malahatTKW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\TKW\\'
malahatSRKW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\SRKW\\'

# SMRU SRKW folder
SMRUSRKW_folder = 'C:/TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'

# SanctSound locs
UNK_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
MN_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\' 
AB_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\' 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model and labels
model_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\Bckrnd_mn_srkw_tkw_offshore_TKW_balanced_4k\\CustomClassifier_100_calls_Balanced_calltypes.tflite"
label_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\Bckrnd_mn_srkw_tkw_offshore_TKW_balanced_4k\\CustomClassifier_100_calls_Balanced_calltypes_Labels.txt"

# Example usage
audio_folder = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'


output_csv = "predictions_output.csv"

# Batch process audio files in folder and export to CSV
processor = Eco.BirdNetPredictor(model_path, label_path, malahatTKW_folder)
TKW_testData_birdnet5 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, malahatSRKW_folder)
RKW_testData_Malahat_birdnet5 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, SMRUSRKW_folder)
RKW_testData_birdnet5 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, UNK_bio_folder)
UNDBIO_testData_birdnet5 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, MN_bio_folder)
MN_testData_birdnet5 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, AB_bio_folder)
AB_testData_birdnet5 = processor.batch_process_audio_folder(output_csv)






# Figure out the false positives

MN_testData_birdnet5['FP'] = MN_testData_birdnet5['Class'] != 'HW'
MN_testData_birdnet5['Class'].value_counts()
MN_testData_birdnet5['Truth'] = 'HW' 

AB_testData_birdnet5['FP'] = AB_testData_birdnet5['Class'] != 'Background'
AB_testData_birdnet5['Class'].value_counts()
AB_testData_birdnet5['Truth'] = 'Background'


TKW_testData_birdnet5['FP'] = TKW_testData_birdnet5['Class'] != 'TKW'
TKW_testData_birdnet5['Class'].value_counts()
TKW_testData_birdnet5['Truth'] = 'TKW'


RKW_testData_birdnet5['FP'] = RKW_testData_birdnet5['Class'] != 'SRKW'
RKW_testData_birdnet5['Class'].value_counts()
RKW_testData_birdnet5['Truth'] = 'SRKW'


RKW_testData_Malahat_birdnet5['FP'] = RKW_testData_Malahat_birdnet5['Class'] != 'SRKW'
RKW_testData_Malahat_birdnet5['Class'].value_counts()
RKW_testData_Malahat_birdnet5['Truth'] = 'SRKW'

UNDBIO_testData_birdnet5['FP'] = UNDBIO_testData_birdnet5['Class'] != 'Background'
UNDBIO_testData_birdnet5['Class'].value_counts()
UNDBIO_testData_birdnet5['Truth'] = 'Background'


ALLData_birdnet05 = pd.concat([MN_testData_birdnet5, AB_testData_birdnet5,
                     UNDBIO_testData_birdnet5, RKW_testData_Malahat_birdnet5, 
                     RKW_testData_birdnet5, TKW_testData_birdnet5])

FPData_birdnet05 = ALLData_birdnet05[ALLData_birdnet05['FP'] == True]


import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_birdnet05['Class'].unique()}


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'UndBio':'#e377c2'}   


relevant_classes = ['SRKW', 'TKW']
roc_results_05 = plot_one_vs_others_roc(FPData_birdnet05,  
                                     ALLData_birdnet05,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 4", 
                                     class_colors= class_colors)

# Example usage: Plot confusion matrix for ALLData_birdnet05
cm_df = plot_confusion_matrix(ALLData_birdnet05, threshold=0.5)




###########################################################################
# One vs rest roc curves
###########################################################################




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
malahatSRKW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\SRKW\\'
malahatHW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\HW\\'

# SMRU SRKW folder
SMRUSRKW_folder = 'C:/TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'

# SanctSound locs
UNK_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
MN_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\' 
AB_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\' 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model and labels
model_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\fix_mn_srkw_offshore_tkw_balanced_4k\\Output\\CustomClassifier_8khz.tflite"
label_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\fix_mn_srkw_offshore_tkw_balanced_4k\\Output\\CustomClassifier_8khz_Labels.txt"

# Example usage
audio_folder = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'


output_csv = "predictions_output.csv"

# Batch process audio files in folder and export to CSV
processor = Eco.BirdNetPredictor(model_path, label_path, AB_bio_folder)
AB_testData_birdnet6 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, malahatTKW_folder)
TKW_testData_birdnet6, Malahat_tkw_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_tkw_birdnet06['Class'] = Malahat_tkw_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_tkw_birdnet06['Truth']= 'TKW'

processor = Eco.BirdNetPredictor(model_path, label_path, malahatSRKW_folder)
RKW_testData_Malahat_birdnet6, Malahat_srkw_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_srkw_birdnet06['Class'] = Malahat_srkw_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_srkw_birdnet06['Truth']= 'SRKW'

processor = Eco.BirdNetPredictor(model_path, label_path, malahatHW_folder)
HW_testData_Malahat_birdnet6, Malahat_HW_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_HW_birdnet06['Class'] = Malahat_HW_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_HW_birdnet06['Truth']= 'HW'




processor = Eco.BirdNetPredictor(model_path, label_path, SMRUSRKW_folder)
RKW_testData_birdnet6 = processor.batch_process_audio_folder(output_csv)


processor = Eco.BirdNetPredictor(model_path, label_path, UNK_bio_folder)
UNDBIO_testData_birdnet6 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, MN_bio_folder)
MN_testData_birdnet6 = processor.batch_process_audio_folder(output_csv)



#%%
# Figure out the false positives

MN_testData_birdnet6['FP'] = MN_testData_birdnet6['Class'] != 'HW'
MN_testData_birdnet6['Class'].value_counts()
MN_testData_birdnet6['Truth'] = 'HW' 

AB_testData_birdnet6['FP'] = AB_testData_birdnet6['Class'] != 'Background'
AB_testData_birdnet6['Class'].value_counts()
AB_testData_birdnet6['Truth'] = 'Background'


TKW_testData_birdnet6['FP'] = TKW_testData_birdnet6['Class'] != 'TKW'
TKW_testData_birdnet6['Class'].value_counts()
TKW_testData_birdnet6['Truth'] = 'TKW'


RKW_testData_birdnet6['FP'] = RKW_testData_birdnet6['Class'] != 'SRKW'
RKW_testData_birdnet6['Class'].value_counts()
RKW_testData_birdnet6['Truth'] = 'SRKW'


RKW_testData_Malahat_birdnet6['FP'] = RKW_testData_Malahat_birdnet6['Class'] != 'SRKW'
RKW_testData_Malahat_birdnet6['Class'].value_counts()
RKW_testData_Malahat_birdnet6['Truth'] = 'SRKW'

UNDBIO_testData_birdnet6['FP'] = UNDBIO_testData_birdnet6['Class'] != 'Background'
UNDBIO_testData_birdnet6['Class'].value_counts()
UNDBIO_testData_birdnet6['Truth'] = 'Background'


ALLData_birdnet06 = pd.concat([MN_testData_birdnet6, AB_testData_birdnet6,
                     UNDBIO_testData_birdnet6, RKW_testData_Malahat_birdnet6, 
                     RKW_testData_birdnet6, TKW_testData_birdnet6])

FPData_birdnet06 = ALLData_birdnet06[ALLData_birdnet06['FP'] == True]



# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_birdnet06['Class'].unique()}


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'UndBio':'#e377c2'}   


relevant_classes = ['SRKW', 'TKW']
roc_results_06 = plot_one_vs_others_roc(FPData_birdnet06,  
                                     ALLData_birdnet06,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 6", 
                                     class_colors= class_colors)



#%%
# To show recall

Malahat_srkw_birdnet06['Score'] = Malahat_srkw_birdnet06['SRKW']
Malahat_tkw_birdnet06['Score'] = Malahat_tkw_birdnet06['TKW']
Malahat_HW_birdnet06['Score'] = Malahat_HW_birdnet06['HW']



# Get the 90% probability thresholds
aa, bb =plot_logistic_fit_with_cutoffs(Malahat_HW_birdnet06, score_column="HW", 
                                   class_column="Class", truth_column="Truth")
HW_90_cutoff = find_cutoff(aa, 1)[1]# 95th percentile 



aa, bb = plot_logistic_fit_with_cutoffs(Malahat_tkw_birdnet06, score_column="TKW", 
                                   class_column="Class", truth_column="Truth")
TKW_90_cutoff = find_cutoff(aa, 1)[1] # 95th percentile 


aa, bb = plot_logistic_fit_with_cutoffs(Malahat_srkw_birdnet06, score_column="SRKW", 
                                   class_column="Class", truth_column="Truth")
SRKW_90_cutoff = find_cutoff(aa, 1)[1]# 95th percentile 


# Using a dictionary threshold:
custom_thresholds = {
    "SRKW": SRKW_90_cutoff,
    "TKW": TKW_90_cutoff,
    'HW':HW_90_cutoff}


EvalDat  = pd.concat([Malahat_srkw_birdnet06, 
                      Malahat_tkw_birdnet06, 
                      Malahat_HW_birdnet06])


plot_one_vs_others_pr(EvalDat,
                      relevant_classes=['SRKW', 'TKW', 'HW'], class_colors=None,
                          titleStr="One-vs-Others Precision–Recall Curve")


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
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 6", 
                                     class_colors= class_colors)


cm_df = plot_confusion_matrix(EvalDat, threshold=custom_thresholds)






#%% Birdnet 7 birdnet Class
########################################################################
# Exactly the same as birdnet 6 but run with 
###########################################################################

# Malahat TKW model Loc
malahatTKW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\TKW\\'
malahatSRKW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\SRKW\\'
malahatHW_folder = 'C:/TempData\\AllData_forBirdnet\\MalahatValidation\\HW\\'

# SMRU SRKW folder
SMRUSRKW_folder = 'C:/TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'

# SanctSound locs
UNK_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
MN_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\' 
AB_bio_folder = 'C:/TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\' 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model and labels
model_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\fix_mn_srkw_offshore_tkw_balanced_4k\\Output\\CustomClassifier_10_15khz.tflite"
label_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\fix_mn_srkw_offshore_tkw_balanced_4k\\Output\\CustomClassifier_10_15khz_Labels.txt"

# Example usage
audio_folder = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\SMRU_test\\'


output_csv = "predictions_output.csv"

# Batch process audio files in folder and export to CSV
processor = Eco.BirdNetPredictor(model_path, label_path, AB_bio_folder)
AB_testData_birdnet7 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, malahatTKW_folder)
TKW_testData_birdnet7, Malahat_tkw_birdnet07 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_tkw_birdnet07['Class'] = Malahat_tkw_birdnet07[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_tkw_birdnet07['Truth']= 'TKW'

processor = Eco.BirdNetPredictor(model_path, label_path, malahatSRKW_folder)
RKW_testData_Malahat_birdnet7, Malahat_srkw_birdnet07 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_srkw_birdnet07['Class'] = Malahat_srkw_birdnet07[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_srkw_birdnet07['Truth']= 'SRKW'

processor = Eco.BirdNetPredictor(model_path, label_path, malahatHW_folder)
HW_testData_Malahat_birdnet7, Malahat_HW_birdnet07 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
Malahat_HW_birdnet07['Class'] = Malahat_HW_birdnet07[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
Malahat_HW_birdnet07['Truth']= 'HW'




processor = Eco.BirdNetPredictor(model_path, label_path, SMRUSRKW_folder)
RKW_testData_birdnet7 = processor.batch_process_audio_folder(output_csv)


processor = Eco.BirdNetPredictor(model_path, label_path, UNK_bio_folder)
UNDBIO_testData_birdnet7 = processor.batch_process_audio_folder(output_csv)

processor = Eco.BirdNetPredictor(model_path, label_path, MN_bio_folder)
MN_testData_birdnet7 = processor.batch_process_audio_folder(output_csv)



#%%
# Figure out the false positives

MN_testData_birdnet7['FP'] = MN_testData_birdnet7['Class'] != 'HW'
MN_testData_birdnet7['Class'].value_counts()
MN_testData_birdnet7['Truth'] = 'HW' 

AB_testData_birdnet7['FP'] = AB_testData_birdnet7['Class'] != 'Background'
AB_testData_birdnet7['Class'].value_counts()
AB_testData_birdnet7['Truth'] = 'Background'


TKW_testData_birdnet7['FP'] = TKW_testData_birdnet7['Class'] != 'TKW'
TKW_testData_birdnet7['Class'].value_counts()
TKW_testData_birdnet7['Truth'] = 'TKW'


RKW_testData_birdnet7['FP'] = RKW_testData_birdnet7['Class'] != 'SRKW'
RKW_testData_birdnet7['Class'].value_counts()
RKW_testData_birdnet7['Truth'] = 'SRKW'


RKW_testData_Malahat_birdnet7['FP'] = RKW_testData_Malahat_birdnet7['Class'] != 'SRKW'
RKW_testData_Malahat_birdnet7['Class'].value_counts()
RKW_testData_Malahat_birdnet7['Truth'] = 'SRKW'

UNDBIO_testData_birdnet7['FP'] = UNDBIO_testData_birdnet7['Class'] != 'Background'
UNDBIO_testData_birdnet7['Class'].value_counts()
UNDBIO_testData_birdnet7['Truth'] = 'Background'


ALLData_birdnet07 = pd.concat([MN_testData_birdnet7, AB_testData_birdnet7,
                     UNDBIO_testData_birdnet7, RKW_testData_Malahat_birdnet7, 
                     RKW_testData_birdnet7, TKW_testData_birdnet7])

FPData_birdnet07 = ALLData_birdnet07[ALLData_birdnet07['FP'] == True]



# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_birdnet07['Class'].unique()}


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'UndBio':'#e377c2'}   


relevant_classes = ['SRKW', 'TKW']
roc_results_07 = plot_one_vs_others_roc(FPData_birdnet07,  
                                     ALLData_birdnet07,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 7", 
                                     class_colors= class_colors)



#%%
# To show recall

Malahat_srkw_birdnet07['Score'] = Malahat_srkw_birdnet07['SRKW']
Malahat_tkw_birdnet07['Score'] = Malahat_tkw_birdnet07['TKW']
Malahat_HW_birdnet07['Score'] = Malahat_tkw_birdnet07['HW']



# Get the 90% probability thresholds
aa, bb =plot_logistic_fit_with_cutoffs(Malahat_HW_birdnet07, score_column="HW", 
                                   class_column="Class", truth_column="Truth")
HW_90_cutoff = find_cutoff(aa, 1)[1]



aa, bb = plot_logistic_fit_with_cutoffs(Malahat_tkw_birdnet07, score_column="TKW", 
                                   class_column="Class", truth_column="Truth")
TKW_90_cutoff = find_cutoff(aa, 1)[1]


aa, bb = plot_logistic_fit_with_cutoffs(Malahat_HW_birdnet07, score_column="HW", 
                                   class_column="Class", truth_column="Truth")
SRKW_90_cutoff = find_cutoff(aa, 1)[1]


# Using a dictionary threshold:
custom_thresholds = {
    "SRKW": SRKW_90_cutoff,
    "TKW": TKW_90_cutoff,
    'HW':HW_90_cutoff}


EvalDat  = pd.concat([Malahat_srkw_birdnet07, 
                      Malahat_tkw_birdnet07, 
                      Malahat_HW_birdnet07])

cm_df = plot_confusion_matrix(EvalDat, threshold=.1)

plot_confusion_matrix(EvalDat, threshold=0.8)


#%% Plot birdnet results together


# Define colors for SRKW and TKW
models = ['03', '04', '05', '06']
colors = {'SRKW': 'blue', 'TKW': 'orange'}
line_styles = ['-', '--', ':', '-.']  # One for each model

# ROC results for different models
roc_models = {'03': roc_results_03, 
              '04': roc_results_04, 
              '05': roc_results_05,
              '06': roc_results_06}


# Create subplots: one for SRKW, one for TKW
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=True)

# Titles for subplots
titles = {'SRKW': 'SRKW ROC Curves', 'TKW': 'TKW ROC Curves'}

# Loop over SRKW and TKW, plotting on separate subplots
for i, key in enumerate(['SRKW', 'TKW']):
    ax = axes[i]
    for j, model in enumerate(models):
        results = eval(f'roc_results_{model}')
        ax.plot(results[key]['fpr'], results[key]['tpr'], 
                linestyle=line_styles[j], color=colors[key], label=f'Model {model}')
    
    # Formatting
    ax.set_title(titles[key])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid()


# Set x-axis limit for the TKW plot
axes[1].set_xlim([0, 0.06])  # Adjust this as needed

# Adjust layout and show plot
plt.tight_layout()
plt.show()
############################################################################
#%% Save everything
#############################################################################
from joblib import dump, load


workspace = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, list, dict, tuple, set, bool))}

dump(workspace, "C://Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\ModelEval_2020_03_01.joblib")

# from joblib import load
# # Load variables and update global namespace
# globals().update(load("ModelEval.joblib"))
raven_file = 'C://TempData\\DCLDE_EVAL\\Malahat_JASCO\\Annotations\\BirdnetDetector_05.txt'


#MalahatStn3 = MalahatStn3.rename(columns={'Slection': 'Selection'})

# Save as a tab-delimited text file
#MalahatStn3.to_csv(raven_file, sep='\t', index=False, quoting=3) 

#with open(raven_file, 'w') as f:
#    # Write header
    # f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tCommon name\tScore\n")

    # for _, row in MalahatStn3.iterrows():
    #     Selection = row['Selection']
    #     View = row['View']
    #     Channel = row['Channel']
    #     start = row['Begin Time (S)']
    #     end = row['End Time (S)']
    #     label = row['Common name']
    #     confidence = row['Score']
        
    #     line = f"{Selection}\t{View}\t{Channel}\t{start}\t{end}\t{label}\t{confidence}\n"
    #     f.write(line)


####################################################################
#%% Get predictions on training data to identify challanging ones
####################################################################
import os
# Model and labels
model_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\Bckrnd_mn_srkw_tkw_offshore_TKW_balanced_4k\\CustomClassifier_100_calls_Balanced_calltypes.tflite"
label_path = "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\Bckrnd_mn_srkw_tkw_offshore_TKW_balanced_4k\\CustomClassifier_100_calls_Balanced_calltypes_Labels.txt"

# Example usage
audio_folder = 'C:\\TempData\\BirdNet5class_balacedTKWCalls'


output_csv = "TrainigDataPredictions.csv"

# Batch process audio files in folder and export to CSV
processor = Eco.BirdNetPredictor(model_path, label_path, audio_folder)
TrainData = processor.batch_process_audio_folder(output_csv)
# Extract folder name
TrainData['truth'] = TrainData['FilePath'].apply(lambda x: os.path.basename(os.path.dirname(x)))

# Get the wrong predictions
wrongDf = TrainData[TrainData['Class'] != TrainData['truth']]



import os
import shutil
from pathlib import Path

# Define the new base folder
new_base_folder = r"C:\TempData\BirdNet5class_balacedTKWCalls_hardExamples"

# Iterate over the incorrect predictions and copy files
for _, row in wrongDf.iterrows():
    old_path = row["FilePath"]  # Original file path
    class_name = row["Class"]  # Subfolder (class name)
    
    # Define new path (keeping subfolder structure)
    new_folder = os.path.join(new_base_folder, class_name)
    new_path = os.path.join(new_folder, os.path.basename(old_path))

    # Create the new folder if it doesn’t exist
    os.makedirs(new_folder, exist_ok=True)

    # Copy the file
    shutil.copy2(old_path, new_path)  # copy2 preserves metadata

print("Files copied successfully!")





