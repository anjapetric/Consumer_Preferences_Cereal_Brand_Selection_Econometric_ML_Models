import pandas as pd
import numpy as np
from xlogit import MixedLogit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, brier_score_loss, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


# Define a function to calculate the multiclass Brier score loss
def multiclass_brier_score_loss(true_labels, probability_estimates):
    # Calculate the number of classes (brands) based on the columns in the probability estimates
    n_classes = probability_estimates.shape[1]
    brier_scores = []  # Initialize an empty list to store Brier scores for each class
    
    # Loop through each class to calculate its individual Brier score
    for class_index in range(n_classes):
        # Generate binary outcomes for the current class where 1 indicates the class was chosen, 0 otherwise
        true_binary = (true_labels == class_index).astype(int)
        # Extract the predicted probabilities for the current class from the array
        preds_for_class = probability_estimates[:, class_index]
        # Calculate the Brier score for the current class and add it to the list
        brier_scores.append(brier_score_loss(true_binary, preds_for_class))
    
    # Return the average Brier score across all classes
    return np.mean(brier_scores)

# Load and preprocess your data
df = pd.read_excel("/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/year 3/Block 5/QM/data/data_purchase_long_Iryna.xlsx")
df['Y'] = df['Y'].astype(int)
df['price'] = df['price'].astype(float)
df['dis'] = df['dis'].astype(float)
df['feat'] = df['feat'].astype(float)
df['FMYSize'] = df['FMYSize'].astype(float)
df['Income'] = df['Income'].astype(float)

print(df["discount"])
# Define variable names
varnames = ['price', 'dis', 'feat', 'FMYSize', 'Income', 'discount', 'lagged_choice', 'brand_loyalty']
randvars = {'price': 'ln','dis': 'ln',
                         'feat': 'ln', 'discount': 'ln','lagged_choice': 'n', 'brand_loyalty': 'n'}
### Coefficients ###

mixedLogit = MixedLogit()
mixedLogit.fit(X=df[varnames],
               y=df['Y'],
               varnames=varnames,
               alts=df['brand'],
               ids=df['id'],
               panels=df['panelist'],
               randvars={'price': 'ln','dis': 'ln',
                         'feat': 'ln', 'discount': 'ln','lagged_choice': 'n', 'brand_loyalty': 'n'},
               isvars=['FMYSize', 'Income'],
               n_draws=1000,
               base_alt = 2,
               fit_intercept=True)
mixedLogit.summary() 
predicted_choices, proba, freq = mixedLogit.predict(
        X=df[varnames], 
        varnames=varnames, 
        ids=df['id'], 
        alts=df['brand'], 
        panels=df['panelist'], 
        isvars=['FMYSize', 'Income'], 
        n_draws=100, 
        return_proba=True, 
        return_freq=True
    )
    
print(predicted_choices)

### Prediction part ###
# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
brier_scores = []
all_predicted_choices = []
all_actual_choices = []
f1_scores = []
    
# Perform K-Fold Cross-Validation
for train_index, test_index in kf.split(df['id'].unique()):
    train_ids = df['id'].unique()[train_index]
    test_ids = df['id'].unique()[test_index]
    
    train_df = df[df['id'].isin(train_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()
    # Standardize the features
    scaler = StandardScaler()
    train_df[varnames] = scaler.fit_transform(train_df[varnames]).astype(np.float64)
    test_df[varnames] = scaler.transform(test_df[varnames]).astype(np.float64)
    
    # Initialize and fit the Mixed Logit model on the training data
    model = MixedLogit()
    model.fit(
            X=train_df[varnames],
            y=train_df['Y'],
            varnames=varnames,
            alts=train_df['brand'],
            ids=train_df['id'],
            panels=train_df['panelist'],
            randvars= randvars,
            isvars=['FMYSize', 'Income'],
            n_draws=100,
            fit_intercept=True
        )    
    predicted_choices, proba, freq = model.predict(
        X=test_df[varnames], 
        varnames=varnames, 
        ids=test_df['id'], 
        alts=test_df['brand'], 
        panels=test_df['panelist'], 
        isvars=['FMYSize', 'Income'], 
        n_draws=100, 
        return_proba=True, 
        return_freq=True
    )
    
    # Filter the rows where 'chosen' is 1
    chosen_brands = test_df[test_df['Y'] == 1]
    final_df = chosen_brands[['id', 'brand']]
    actual_choices = final_df['brand']
    all_actual_choices.extend(actual_choices)
    all_predicted_choices.extend(predicted_choices)
    # Calculate the prediction accuracy
    accuracy = np.mean(predicted_choices == actual_choices)
    accuracies.append(accuracy)
    print(f"Fold accuracy: {accuracy}")

    # Calculate Brier score
    # Reshape the dataframe such that each 'id' has a single row and each column represents a brand.
    # The values in the matrix will be 0 or 1, where 1 indicates the chosen brand for that shopping trip.
    choice_matrix = test_df.pivot(index='id', columns='brand', values='Y')
    # Convert the pivot table to a NumPy array, as it's required for some operations like calculating the Brier score.
    choice_array = choice_matrix.to_numpy()
    actual_choices_array = choice_matrix.idxmax(axis=1).to_numpy()
    current_brier_score = multiclass_brier_score_loss(actual_choices_array, proba)
    brier_scores.append(current_brier_score)
    print(f"Fold Brier score: {current_brier_score}")
    
    current_f1_score = f1_score(actual_choices, predicted_choices, average='weighted')
    f1_scores.append(current_f1_score)
    print(f"Fold F1 score: {current_f1_score}")
    
final_confusion_matrix = confusion_matrix(all_actual_choices, all_predicted_choices, labels=df['brand'].unique())

mean_accuracy = np.mean(accuracies)
sd_accuracy = np.std(accuracies)
mean_brier = np.mean(brier_scores)
sd_brier = np.std(brier_scores)
mean_f1 = np.mean(f1_scores)
sd_f1 = np.std(f1_scores)

print(f"Mean Prediction Accuracy: {mean_accuracy} and standard deviation: {sd_accuracy}")
print(f"Mean Brier score: {mean_brier} and standard deviation: {sd_brier}")
print(f"Mean F1 score: {mean_f1} and standard deviation: {sd_f1}")
# Create a visually appealing confusion matrix using Seaborn
labels = np.sort(df['brand'].unique())  # Sort or order as needed for consistency
plt.figure(figsize=(10, 7))  # Set the figure size
sns.heatmap(final_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#results with old variables
#Mean Prediction Accuracy: 0.4257443443443443
#Mean Brier score: 0.1468338521279586


# logical check to see if accuracy from confusion matrix matches the one before
corrects = np.trace(final_confusion_matrix)  # Sum of diagonal elements
total = np.sum(final_confusion_matrix)       # Total sum of the matrix

hit_rate = corrects / total
print(hit_rate)

### calculating f1 scores for mixed logit with 1000 draws from other file and general logit ###
# Calculate precision, recall, and F1 score from the confusion matrix
confusion_matrix_1000_draws = np.array([
    [818, 612, 27, 26, 2, 6],
    [400, 1645, 27, 16, 4, 9],
    [137, 296, 70, 5, 3, 7],
    [131, 260, 8, 55, 0, 2],
    [40, 130, 2, 2, 15, 0],
    [32, 100, 14, 0, 1, 94]
])

confusion_matrix_iryna = np.array([[701, 690, 41, 47, 3, 9],
                                   [413, 1568, 52, 39, 12, 17],
                                   [158, 240, 98, 6, 4, 12],
                                   [138, 217, 13, 81, 5, 2],
                                   [51, 82, 2, 3, 49, 2],
                                   [43, 55, 12, 3, 2, 126]])

print(confusion_matrix_iryna.transpose())
                                   
TP = np.diag(confusion_matrix_1000_draws)
FP = np.sum(confusion_matrix_1000_draws, axis=0) - TP
FN = np.sum(confusion_matrix_1000_draws, axis=1) - TP

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Handle cases where precision and recall might be NaN (due to division by zero)
precision = np.nan_to_num(precision)
recall = np.nan_to_num(recall)
f1 = np.nan_to_num(f1)

# Calculate the weighted average of F1 score
support = np.sum(confusion_matrix_1000_draws, axis=1)  # Number of true instances for each label
weighted_f1 = np.sum(f1 * support) / np.sum(support)

print(f"Precision per class: {precision}")
print(f"Recall per class: {recall}")
print(f"F1 score per class: {f1}")
print(f"Weighted F1 score: {weighted_f1}")

                              
TP = np.diag(confusion_matrix_iryna)
FP = np.sum(confusion_matrix_iryna) - TP
FN = np.sum(confusion_matrix_iryna, axis=1) - TP

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Handle cases where precision and recall might be NaN (due to division by zero)
precision = np.nan_to_num(precision)
recall = np.nan_to_num(recall)
f1 = np.nan_to_num(f1)

# Calculate the weighted average of F1 score
support = np.sum(confusion_matrix_1000_draws, axis=1)  # Number of true instances for each label
weighted_f1 = np.sum(f1 * support) / np.sum(support)

print(f"Precision per class: {precision}")
print(f"Recall per class: {recall}")
print(f"F1 score per class: {f1}")
print(f"Weighted F1 score: {weighted_f1}")
