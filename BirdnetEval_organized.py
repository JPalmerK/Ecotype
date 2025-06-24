# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 23:29:54 2025

@author: kaity
"""



from keras.models import load_model
import EcotypeDefs as Eco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

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

def plot_confusion_matrix(
    data,
    threshold=0.5,
    class_column="Predicted Class",
    score_column="Score",
    truth_column="Truth",
    titleStr = 'Model x'):
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
    plt.title(titleStr)
    plt.show()

    # Clean up
    df.drop(columns=["_ThresholdToUse"], inplace=True, errors="ignore")
    return cm_df

def plot_confusion_matrix_multilabel(
    data,
    score_cols,
    truth_column="Truth",
    threshold=0.5,
    titleStr = 'Model x'):
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
    ax.set_title(titleStr)
    plt.show()

    return cm_df

def plot_logistic_fit_with_cutoffs(data, score_column="Score", 
                                   class_column="Predicted Class", truth_column="Truth",
                                   titleStr = ":SRKW", plotLogit= False):
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
    conf_model = sm.Logit(data["Correct"],  
                          sm.add_constant(data[score_column])).fit(disp=False)
    logit_model = sm.Logit(data["Correct"], 
                           sm.add_constant(data["Logit_Score"])).fit(disp=False)

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
    
    if plotLogit:

        # Plot Confidence Score Model
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[score_column], y=data["Correct"], alpha=0.3, label="Observations")
        plt.plot(conf_range, conf_pred, color="red", label="Logistic Fit (Confidence Score)")
        for i, cutoff in enumerate(conf_cutoffs):
            plt.axvline(cutoff, linestyle="--", color=["orange", "red", "magenta"][i], label=f"p={0.9 + i*0.05:.2f}")
        plt.xlabel("BirdNET Confidence Score")
        plt.ylabel("Pr(Correct Detection)")
        plt.title(f"Logistic Fit: Confidence Score {titleStr}")
        plt.legend()
        plt.grid()
    
        # Plot Logit Score Model
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data["Logit_Score"], y=data["Correct"], alpha=0.3, label="Observations")
        plt.plot(logit_range, logit_pred, color="blue", label="Logistic Fit (Logit Score)")
        for i, cutoff in enumerate(logit_cutoffs):
            plt.axvline(cutoff, linestyle="--", color=["orange", "red", "magenta"][i], label=f"p={0.9 + i*0.05:.2f}")
        plt.xlabel(f"Logit of BirdNET Confidence Score {titleStr}")
        plt.ylabel("Pr(Correct Detection)")
        plt.title("Logistic Fit: Logit Score")
        plt.legend()
        plt.grid()
    
        plt.show()

    return conf_model, logit_model

def plot_one_vs_others_pr(all_df, 
                          relevant_classes=None, 
                          class_colors=None,
                          titleStr="One-vs-Others Precision–Recall Curve"):
    """
    Generates a One-vs.-Others Precision–Recall curve for each class in the dataset,
    deriving false positives and true positives internally.

    Parameters:
    - all_df: DataFrame containing 'Truth', 'Class', and 'Score' columns.
    - relevant_classes: List of class names to include in the comparison. If None, uses all classes.
    - class_colors: Dictionary mapping class names to specific colors.
    - titleStr: Title for the Precision–Recall curve plot.

    Returns:
    - pr_data: Dictionary containing precision-recall data for each class.
    - auc_pr_dict: Dictionary containing AUC-PR values for each class.
    - mean_ap: Mean Average Precision (mAP) across all classes.
    """

    # If no classes specified, use all unique ones from 'Class'
    if relevant_classes is None:
        relevant_classes = np.unique(all_df['Class'])
        
    truthClasses = np.unique(all_df['Truth'])

    # Assign a color to each class if not provided
    default_colors = {
        cls: plt.cm.get_cmap("tab10").colors[i % 10]
        for i, cls in enumerate(relevant_classes)
    }
    color_map = default_colors if class_colors is None else {**default_colors, **class_colors}

    # Define thresholds to sweep over
    thresholds = np.linspace(0, 1, 200)
    #thresholds = np.unique(all_df['Top Score'])

    # Storage for precision–recall data and AUC-PR values
    pr_data = {}
    auc_pr_dict = {}  # Separate dictionary for AUC-PR values

    plt.figure(figsize=(8, 6))

    for cls in relevant_classes:
        precision_list, recall_list = [], []
    
        # Binary masks for positive class (truth)
        is_truth_cls = all_df['Truth'] == cls

        for threshold in thresholds:
            # TP: Correct predictions above threshold
            TP = ((is_truth_cls) & (all_df[cls] >= threshold)).sum()
    
            # FP: Incorrect predictions above threshold
            FP = ((~is_truth_cls) & (all_df[cls] >= threshold)).sum()
    
            # FN: Missed positives (either wrong class or too low score)
            FN = (is_truth_cls & (all_df[cls] < threshold)).sum()
    
            # Precision calculation
            precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    
            # Recall calculation
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
            precision_list.append(precision)
            recall_list.append(recall)
        
        # Compute area under PR curve (AUC-PR)
        auc_pr = auc(recall_list, precision_list)

        # Store results
        pr_data[cls] = {
            'thresholds': thresholds,
            'precision': np.array(precision_list),
            'recall': np.array(recall_list)
        }

        # Store AUC-PR in a separate dictionary
        auc_pr_dict[cls] = auc_pr

        # Plot Precision–Recall curve
        plt.plot(recall_list, precision_list, label=cls, 
                 color=color_map.get(cls, "black"))

    # Compute mean average precision (mAP)
    mean_ap = np.mean(list(auc_pr_dict.values()))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.show()

    return pr_data, auc_pr_dict, mean_ap  # Return mAP as a separate output

def find_cutoff(model, coef_index):
    return (np.log([0.90 / 0.10, 0.95 / 0.05, 0.99 / 0.01]) - model.params[0]) / model.params[coef_index]

# --- Helper Functions ---

def process_dataset(model_path, label_path, audio_folder, truth_label,
                    output_csv="predictions_output.csv", 
                    return_scores=True, 
                    classes=['SRKW','OKW','HW','TKW'],
                    thresh=0):
    """
    Process an audio folder using Eco.BirdNetPredictor.
    Returns a DataFrame with raw scores, the predicted class (as the max among classes),
    and a 'Truth' column set to the given truth_label.
    """
    processor = Eco.BirdNetPredictorNew(model_path, label_path, 
                                        audio_folder, 
                                        confidence_thresh=thresh)
    df = processor.batch_process_audio_folder(output_csv, 
                                              return_raw_scores=True)
    return df
    #df['Class_Est'] = df[classes].idxmax(axis=1)
    
    
    # if return_scores:
    #     df, scores = processor.batch_process_audio_folder(output_csv, 
    #                                                       return_raw_scores=True)
    #     scores['Class_Est'] = scores[classes].idxmax(axis=1)
    #     scores['Truth'] = truth_label
    #     return scores
    # else:
    #     return processor.batch_process_audio_folder(output_csv)

def process_multiple_datasets(model_path, label_path, folder_truth_map,
                              output_csv="predictions_output.csv", 
                              classes=['SRKW','OKW','HW','TKW'], 
                              thresh = 0):
    """
    Process multiple datasets defined by a dictionary mapping truth labels to folder paths.
    Returns a combined DataFrame.
    
    allPreds: logical, whether to threshold 
    """
    datasets = []
    for truth, folder in folder_truth_map.items():
        df = process_dataset(model_path, label_path, folder, truth,
                             output_csv=output_csv, classes=classes,
                             thresh=thresh)
        datasets.append(df)
    return pd.concat(datasets, ignore_index=True)

def compute_threshold(data, score_column, class_column, plotLogit = False, 
                      title_suffix=""):
    """
    Compute a threshold cutoff for a given class using a logistic fit.
    (Assumes plot_logistic_fit_with_cutoffs and find_cutoff are available.)
    
    Returns the 90th percentile
    """
    logit, _ = plot_logistic_fit_with_cutoffs(data, 
                                              score_column=score_column, 
                                              class_column=class_column,
                                              truth_column="Truth", 
                                              titleStr=title_suffix,
                                              plotLogit=plotLogit)
    cutoff = find_cutoff(logit, 1)[0]
    return cutoff

def evaluate_model(eval_df, 
                   custom_thresholds, 
                   pr_title="Precision–Recall Curve", 
                   roc_title="ROC Curve", 
                   plotPR = True, 
                   plotROC= False):
    """
    Given an evaluation DataFrame and custom thresholds, compute scores,
    identify false positives, and then plot PR curve, ROC curve, and confusion matrix.
    Returns the confusion matrix DataFrame.
    """
    # Compute score for each row based on its predicted class
    eval_df['Score'] = eval_df.apply(lambda row: row[row['Predicted Class']], axis=1)
    
    
    
    # Mark false positives
    eval_df['FP'] = eval_df['Predicted Class'] != eval_df['Truth']
    fp_data = eval_df[eval_df['FP']]
    

    
    metrics = dict({})
    # Plot ROC curve (using preset colors)
    class_colors = {'SRKW': '#1f77b4', 'TKW': '#ff7f0e', 'HW': '#2ca02c', 'OKW': '#e377c2'}
    if plotPR:
        # Plot Precision–Recall curve
        pr_data, auc_pr_dict, mean_ap = plot_one_vs_others_pr(
            eval_df, relevant_classes=list(custom_thresholds.keys()), 
                              class_colors=None, titleStr=pr_title)
       
        metrics['AUC'] = auc_pr_dict
        metrics['pr_data']= pr_data
        metrics['MAP'] = mean_ap
    

    if plotROC:
        plot_one_vs_others_roc(fp_data, eval_df,
                               titleStr=roc_title, 
                               class_colors=class_colors)
    
    # Plot and return confusion matrix
    cm_df = plot_confusion_matrix(eval_df,
                                  threshold=custom_thresholds,
                                    titleStr =roc_title)
    metrics['cm']= cm_df
    
    
    return metrics

# --- Main Evaluation Block ---
if __name__ == "__main__":
    
    # ---- DCLDE Evaluation ----
    # Folder paths for DCLDE data; keys serve as the truth labels.
    dclde_folders = {
        "Background": r"C:\TempData\DCLDE_HOLDOUT\Background",
        "HW":         r"C:\TempData\DCLDE_HOLDOUT\HW",
        "SRKW":       r"C:\TempData\DCLDE_HOLDOUT\SRKW",
        "TKW":        r"C:\TempData\DCLDE_HOLDOUT\TKW",
    }
        
    
    # ---- Malahat Evaluation ----
    # Folder paths for Malahat data.
    # (For Malahat you may only have a subset of classes; here we use TKW, SRKW, and HW.)
    malahat_folders = {
        "TKW":  r"C:\TempData\Malahat_HOLDOUT\TKW",
        "SRKW": r"C:\TempData\Malahat_HOLDOUT\SRKW",
        "HW":   r"C:\TempData\Malahat_HOLDOUT\HW",
        "Background": r"C:\TempData\Malahat_HOLDOUT\Background",
        "UndKW": r"C:\TempData\Malahat_HOLDOUT\KW_und"
    }    
    
    output_csv = "predictions_output.csv"
    
    #%% Birdnet 01- 2000 clips randomly selected filtered 15khz
    ########################################################################
    # 2000 clips randomly selected filtered 15khz
    
    model_config_01 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet01\Birdnet01.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet01\Birdnet01_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_01 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet01_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_01 = process_multiple_datasets(
                model_config_01["model_path"],
                model_config_01["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet01_DCLDE_eval.csv')
            
            eval_dclde_birdnet_01.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet01_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_01 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet01_Malahat_eval.csv')
        eval_malahat_birdnet_01 = eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_01 = process_multiple_datasets(
            model_config_01["model_path"], 
            model_config_01["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet01_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_01.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet01_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff,
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_01 = evaluate_model(eval_dclde_birdnet_01, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 01", 
                                roc_title="DCLDE ROC Curve 01")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_01 = evaluate_model(eval_malahat_birdnet_01, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 01",
                                  roc_title="Malahat ROC Curve 01")
    
    
    
    #%% Birdnet 02- 2000 clips balanced with 100 call type examples 15khz
    ########################################################################
    # In this model we will have a total of 2k annotations for each class, however
    # We will balance across call types with at least 100 in each calltype.
    
    model_config_01 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet02\Birdnet02.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet02\Birdnet02_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_02 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet02_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_02 = process_multiple_datasets(
                model_config_01["model_path"],
                model_config_01["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet02_DCLDE_eval.csv')
            
            eval_dclde_birdnet_02.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet02_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_02 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet02_Malahat_eval.csv')
        eval_malahat_birdnet_02 = eval_malahat_birdnet_02[eval_malahat_birdnet_02['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_01 = process_multiple_datasets(
            model_config_01["model_path"], 
            model_config_01["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet02_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_01.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet02_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_01[eval_dclde_birdnet_01['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_01[eval_malahat_birdnet_01['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_02 = evaluate_model(eval_dclde_birdnet_01, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 02", 
                                roc_title="DCLDE ROC Curve 02")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_02 = evaluate_model(eval_malahat_birdnet_01, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 02",
                                  roc_title="Malahat ROC Curve 02")
    
    #%% Birdnet 03 -2000 clips 100 call types, background 1k undbio/abiotic
    ########################################################################
    # Balanced per class with calltype control for killer whales
    
    model_config_03 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet03\Birdnet03.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet03\Birdnet03_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_03 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet03_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_03 = process_multiple_datasets(
                model_config_03["model_path"],
                model_config_03["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet03_DCLDE_eval.csv')
            
            eval_dclde_birdnet_03.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet03_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_03 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet03_Malahat_eval.csv')
        eval_malahat_birdnet_03 = eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_03 = process_multiple_datasets(
            model_config_03["model_path"], 
            model_config_03["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet03_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_03.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet03_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_03[eval_dclde_birdnet_03['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_03[eval_dclde_birdnet_03['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_03[eval_dclde_birdnet_03['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_03[eval_malahat_birdnet_03['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_03 = evaluate_model(eval_dclde_birdnet_03, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 03", 
                                roc_title="DCLDE ROC Curve 03")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_03 = evaluate_model(eval_malahat_birdnet_03, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 03",
                                  roc_title="Malahat ROC Curve 03")
    
    #%% Birdnet 04 -3000 clips 200 call replicates, background 1.5k undbio/abiotic 
    ########################################################################
    # 3000 calls per class 200 replicates of each call type
   
    
    model_config_04 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet04\Birdnet04.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet04\Birdnet04_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_04 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet04_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_04 = process_multiple_datasets(
                model_config_04["model_path"],
                model_config_04["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet04_DCLDE_eval.csv')
            
            eval_dclde_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet04_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_04 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet04_Malahat_eval.csv')
        eval_malahat_birdnet_04 = eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_04 = process_multiple_datasets(
            model_config_04["model_path"], 
            model_config_04["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet04_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet04_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_04[eval_dclde_birdnet_04['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_04[eval_dclde_birdnet_04['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_04[eval_dclde_birdnet_04['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_04[eval_malahat_birdnet_04['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_04 = evaluate_model(eval_dclde_birdnet_04, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 04", 
                                roc_title="DCLDE ROC Curve 04")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_04 = evaluate_model(eval_malahat_birdnet_04, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 04",
                                  roc_title="Malahat ROC Curve 04")
    #%% Birdnet 05- 6000 clips randomly selected filtered 15khz
    ########################################################################
    # # Same as first birdnet save 6k detections

    
    model_config_05 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet05\Birdnet05.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet05\Birdnet05_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_05 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet05_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_05 = process_multiple_datasets(
                model_config_05["model_path"],
                model_config_05["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet05_DCLDE_eval.csv')
            
            eval_dclde_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet05_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_05 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet05_Malahat_eval.csv')
        eval_malahat_birdnet_05 = eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] != 'KW_und'] 

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_05 = process_multiple_datasets(
            model_config_05["model_path"], 
            model_config_05["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet05_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet05_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_05[eval_dclde_birdnet_05['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_05[eval_dclde_birdnet_05['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_05[eval_dclde_birdnet_05['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_05[eval_malahat_birdnet_05['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_05 = evaluate_model(eval_dclde_birdnet_05, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 05", 
                                roc_title="DCLDE ROC Curve 05")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_05 = evaluate_model(eval_malahat_birdnet_05, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 05",
                                  roc_title="Malahat ROC Curve 05")

    #%% Birdnet 06- Birdnet 04 filtered 8000hz in the app
    ########################################################################
    # Trained with birdnet04 [3000 calls per class 200 replicates of each call type]
    # data but restricted to 8khz in the app
    
    model_config_06 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet06\Birdnet06.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet06\Birdnet06_Labels.txt"
    }
    
    

    try:
        eval_dclde_birdnet_06 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet06_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_06 = process_multiple_datasets(
                model_config_06["model_path"],
                model_config_06["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet06_DCLDE_eval.csv')
            
            eval_dclde_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet06_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_06 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet06_Malahat_eval.csv')
        eval_malahat_birdnet_06 = eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] != 'KW_und'] 
        

    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_06 = process_multiple_datasets(
            model_config_06["model_path"], 
            model_config_06["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet06_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_04.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet06_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_06[eval_dclde_birdnet_06['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_06[eval_dclde_birdnet_06['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_06[eval_dclde_birdnet_06['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_06[eval_malahat_birdnet_06['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_06 = evaluate_model(eval_dclde_birdnet_06, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 06", 
                                roc_title="DCLDE ROC Curve 06")
    

    # Evaluate and plot Malahat performance
    metrics_malahat_06 = evaluate_model(eval_malahat_birdnet_06, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 06",
                                  roc_title="Malahat ROC Curve 06")
    #%% Birdnet 07- 4.5k examples, shifting for augmentation OKW, NRKW
    ########################################################################
    # Run at 15khz in birdnet app
    
    
    model_config_07 = {
        "model_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet07\Birdnet07.tflite",
        "label_path": r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdnetOrganized\Birdnet07\Birdnet07_Labels.txt"
    }
    
    
    
    try:
        eval_dclde_birdnet_07 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet07_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_07 = process_multiple_datasets(
                model_config_07["model_path"],
                model_config_07["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet07_DCLDE_eval.csv')
            
            eval_dclde_birdnet_07.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet07_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_07 = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet07_Malahat_eval.csv')
        # Exclude unidentified KW detections for test
        eval_malahat_birdnet_07 = eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] != 'KW_und'] 
    
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_07 = process_multiple_datasets(
            model_config_07["model_path"], 
            model_config_07["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet07_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_07.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet07_Malahat_eval.csv', index=False)
        
        
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_07[eval_dclde_birdnet_07['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_07[eval_dclde_birdnet_07['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_07[eval_dclde_birdnet_07['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_07[eval_malahat_birdnet_07['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_07 = evaluate_model(eval_dclde_birdnet_07, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 07", 
                                roc_title="DCLDE ROC Curve 07")
    
    
    # Evaluate and plot Malahat performance
    metrics_malahat_07 = evaluate_model(eval_dclde_birdnet_07, 
                                        custom_thresholds=custom_thresholds_malahat, 
                                  pr_title="Malahat Precision–Recall Curve 07",
                                  roc_title="Malahat ROC Curve 07")
 
    
    
    
    #%% Birdnet Previous model 3
    ########################################################################
    # Run at 15khz in birdnet app
    
    
    model_config_03_previous = {
        "model_path": r"C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\Birdnet7Class_ONC_Larger\\CustomClassifier.tflite",
        "label_path": r"C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNET\\Birdnet7Class_ONC_Larger\\CustomClassifier_Labels.txt"
    }
    
    
    
    try:
        eval_dclde_birdnet_03_previous = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet03_previous_DCLDE_eval.csv')
    except:
            # Process all DCLDE datasets into one DataFrame.
            eval_dclde_birdnet_03_previous = process_multiple_datasets(
                model_config_03_previous["model_path"],
                model_config_03_previous["label_path"], 
                dclde_folders, 
                output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet03_previous_DCLDE_eval.csv')
            
            eval_dclde_birdnet_03_previous.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\BirdnetOrganized\Birdnet03_previous_DCLDE_eval.csv', index=False)
    
    
    
    try:
        eval_malahat_birdnet_03_previous = pd.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\Birdnet03_previous_Malahat_eval.csv')
        eval_malahat_birdnet_03_previous = eval_malahat_birdnet_03_previous[eval_malahat_birdnet_03_previous['Truth'] != 'KW_und'] 
   
    except:            
        # Process Malahat datasets; note that we adjust the list of classes if OKW is not present.
        eval_malahat_birdnet_03_previous = process_multiple_datasets(
            model_config_03_previous["model_path"], 
            model_config_03_previous["label_path"], 
            malahat_folders, 
            output_csv='C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet03_previous_Malahat_eval.csv', 
            classes=['SRKW','HW','TKW'])  
        eval_malahat_birdnet_03_previous.to_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\Birdnet03_previous_Malahat_eval.csv', index=False)
        
    # Compute custom thresholds for DCLDE (for OKW we use a fixed value)
    hw_cutoff   = compute_threshold(eval_dclde_birdnet_03_previous[eval_dclde_birdnet_03_previous['Truth'] == "HW"],
                                    score_column="HW",  
                                    class_column="Predicted Class",  
                                    title_suffix="Humpback")
    tkw_cutoff  = compute_threshold(eval_dclde_birdnet_03_previous[eval_dclde_birdnet_03_previous['Truth'] == "TKW"], 
                                    score_column="TKW", class_column="Predicted Class", title_suffix="TKW")
    srkw_cutoff = compute_threshold(eval_dclde_birdnet_03_previous[eval_dclde_birdnet_03_previous['Truth'] == "SRKW"], 
                                    score_column="SRKW", class_column="Predicted Class", title_suffix="SRKW")
    custom_thresholds_dclde = {
        "HW": hw_cutoff,
        "TKW": tkw_cutoff,
        "SRKW": srkw_cutoff
    }
    
    # Compute thresholds for Malahat data
    hw_cutoff_m   = compute_threshold(eval_malahat_birdnet_03_previous[eval_malahat_birdnet_03_previous['Truth'] == "HW"], class_column="Predicted Class", score_column="HW", title_suffix="Malahat HW")
    tkw_cutoff_m  = compute_threshold(eval_malahat_birdnet_03_previous[eval_malahat_birdnet_03_previous['Truth'] == "TKW"], class_column="Predicted Class", score_column="TKW", title_suffix="Malahat TKW")
    srkw_cutoff_m = compute_threshold(eval_malahat_birdnet_03_previous[eval_malahat_birdnet_03_previous['Truth'] == "SRKW"], class_column="Predicted Class", score_column="SRKW", title_suffix="Malahat SRKW")
    custom_thresholds_malahat = {
        "HW": hw_cutoff_m,
        "TKW": tkw_cutoff_m,
        "SRKW": srkw_cutoff_m
    }
    # Evaluate and plot DCLDE performance
    metrics_DCLDE_03_previous = evaluate_model(eval_dclde_birdnet_03_previous, 
                                      custom_thresholds=custom_thresholds_dclde, 
                                pr_title="DCLDE Precision–Recall Curve 03_previous", 
                                roc_title="DCLDE ROC Curve 03_previous")
    
    
    # Evaluate and plot Malahat performance
    metrics_malahat_03_previous = evaluate_model(eval_malahat_birdnet_03_previous, 
                                        custom_thresholds=custom_thresholds_dclde, 
                                  pr_title="Malahat Precision–Recall Curve 03_previous",
                                  roc_title="Malahat ROC Curve 03_previous")  
    
    # Extract missed SRKW
    