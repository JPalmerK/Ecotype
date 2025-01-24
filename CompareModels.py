# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:23:41 2024

Compare the performance of various models including resnet and birdnet


@author: kaity
"""
from keras.models import load_model
import pandas as pd
import numpy as np
import EcotypeDefs as Eco
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
 
#######################################################################
#
#########################################################################


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

class BirdNetEvaluator:
    def __init__(self, truth_files_dict):
        """
        Initialize the evaluator with truth files.
        
        Parameters
        ----------
        truth_files_dict : dict
            Dictionary mapping species labels to file paths of BirdNet selection tables.
        """
        self.truth_files_dict = truth_files_dict
        self.y_true_accum = []
        self.y_pred_accum = []
        self.score_accum = []
        self.all_pred_classes = set()  # Set to store all possible predicted classes

    def load_and_evaluate(self):
        """
        Load BirdNet selection tables and evaluate predictions against ground truth.
        """
        for species, file_path in self.truth_files_dict.items():
            selection_table = pd.read_csv(file_path, sep="\t")
            predictions = selection_table["Common Name"].values
            scores = selection_table.filter(like="Confidence").values  # Assuming score columns are present
            
            # True labels and scores
            truth_labels = [species] * len(predictions)
            self.y_true_accum.extend(truth_labels)
            self.y_pred_accum.extend(predictions)
            self.score_accum.extend(scores)

            # Add the predicted classes to the set of all predicted classes
            self.all_pred_classes.update(predictions)

        self.y_true_accum = np.array(self.y_true_accum)
        self.y_pred_accum = np.array(self.y_pred_accum)
        self.score_accum = np.array(self.score_accum)

    def confusion_matrix(self):
        """Computes a confusion matrix with human-readable labels and accuracy."""
        # Convert the set of all predicted classes to a list and use it as labels
        all_classes = sorted(self.all_pred_classes | set(self.truth_files_dict.keys()))

        # Compute confusion matrix using the updated list of classes
        conf_matrix_raw = confusion_matrix(self.y_true_accum, 
                                           self.y_pred_accum, 
                                           labels=all_classes)
        
        # Normalize confusion matrix by rows
        conf_matrix_percent = conf_matrix_raw.astype(np.float64)
        row_sums = conf_matrix_raw.sum(axis=1, keepdims=True)
        conf_matrix_percent = np.divide(conf_matrix_percent, row_sums, where=row_sums != 0) * 100
        
        # Format confusion matrix to two decimal places
        conf_matrix_percent_formatted = np.array([[f"{value:.2f}" for value in row]
                                                  for row in conf_matrix_percent])
        
        # Create DataFrame
        conf_matrix_df = pd.DataFrame(
            conf_matrix_percent_formatted, 
            index=all_classes, 
            columns=all_classes
        )
        
        # Compute overall accuracy
        accuracy = accuracy_score(self.y_true_accum, self.y_pred_accum)
        
        return conf_matrix_df, conf_matrix_raw, accuracy

    def score_distributions(self):
        """Generates a DataFrame of score distributions for true positives and false positives."""
        score_data = []
        for i, true_label in enumerate(self.y_true_accum):
            pred_label = self.y_pred_accum[i]
            scores = self.score_accum[i]
            
            for class_idx, score in enumerate(scores):
                class_name = list(self.truth_files_dict.keys())[class_idx]
                label_type = "True Positive" if (true_label == class_name == pred_label) else "False Positive"
                score_data.append({
                    "Class": class_name,
                    "Score": score,
                    "Type": label_type
                })

        score_df = pd.DataFrame(score_data)
        
        # Plot paired violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=score_df, x="Class", y="Score", hue="Type", split=True, inner="quartile", palette="muted")
        plt.title("Score Distributions for True Positives and False Positives")
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(title="Type")
        plt.tight_layout()
        plt.show()
        
        return score_df
    def precision_recall_curves(self):
        """Computes and plots precision-recall curves for all classes."""
        num_classes = len( self.truth_files_dict.keys())
        precision_recall_data = {}
    
        plt.figure(figsize=(10, 8))
    
        # Calculate PR curves for each class
        for sppClass in self.truth_files_dict.keys():
            print(sppClass)
            # Check if the class is present in the dataset
            class_present = (self.y_true_accum == sppClass).any()
    
            if not class_present:
                print(f"Class {self.label_dict[class_idx]} is not present in the validation dataset.")
                # Store empty results for missing class
                precision_recall_data[self.label_dict[class_idx]] = {
                    "precision": None,
                    "recall": None,
                    "average_precision": None
                }
                continue
    
            # Binarize true labels for the current class
            true_binary = (self.y_true_accum == sppClass).astype(int)
    
            # Retrieve scores for the current class
            class_scores = self.score_accum[:, class_idx]
    
            # Compute precision, recall, and average precision score
            precision, recall, _ = precision_recall_curve(true_binary, class_scores)
            avg_precision = average_precision_score(true_binary, class_scores)
    
            # Store the data
            precision_recall_data[self.label_dict[class_idx]] = {
                "precision": precision,
                "recall": recall,
                "average_precision": avg_precision
            }
    
            # Plot PR curve
            plt.plot(recall, precision, label=f"{self.label_dict[class_idx]} (AP={avg_precision:.2f})")
    
        # Finalize plot
        plt.title("Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.show()
    
        return precision_recall_data




# Define the base path
base_path = r"C:\Users\kaity\Documents\GitHub\Ecotype\Experiments\BirdNet\KWs_nonKW\SelectionTables\MalahatEval"

# Define the truth files dictionary using the base path
truth_files_dict = {
    "Negative": f"{base_path}\\Negative\\BirdNET_SelectionTable.txt",
    "RKW": f"{base_path}\\RKW\\BirdNET_SelectionTable.txt",
    "TKW": f"{base_path}\\Negative\\BirdNET_SelectionTable.txt"}

evaluator = BirdNetEvaluator(truth_files_dict)
# Run evaluation
evaluator.load_and_evaluate()

# Get confusion matrix and accuracy
conf_matrix_df, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
pr_curves = evaluator.precision_recall_curves()

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")

# Get precision and recall metrics
metrics_df = evaluator.precision_recall()
print("\nMetrics:")
print(metrics_df)



############################################################################
# Confusion matrix precision Recall from resnet
###########################################################################

# C:\Users\kaity\Documents\GitHub\Ecotype\NorthWestAtlantic_EcotypeData_RTO_hop25_balancedHumps_specNorm

label_dict = class_names = {
    0: 'AB',
    1: 'HW',
    2: 'RKW',
    3: 'OKW',
    4: 'TKW',
    5: 'UndBio'}



h5_file_validation1 = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\HumpbackBalanced_8khz\\Malahat_Balanced_melSpec_8khz_PCEN_RTW.h5'
model1 = load_model('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\HumpbackBalanced\\MnBalanced_8khz_Resnet18_8khz_RTOt_hop25.keras')

valLoader1 =  Eco.BatchLoader2(h5_file_validation1, 
                            trainTest = 'train', batch_size=250,  n_classes=6,  
                            minFreq=0,   return_data_labels = False)


# Instantiate the evaluator
evaluator = Eco.ModelEvaluator( loaded_model=model1, 
                               val_batch_loader = valLoader1, 
                               label_dict =label_dict)

# Run the evaluation (only once)
evaluator.evaluate_model()

# Get the various outputs for model checking
conf_matrix_df, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
scoreDF = evaluator.score_distributions()
pr_curves = evaluator.precision_recall_curves()

###############################################################################










