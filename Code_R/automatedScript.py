import pandas as pd
import numpy as np
from xlogit import MixedLogit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelWriter

class MixedLogitModel:
    def __init__(self, data_path, varnames, randvars):
        self.data = self.load_and_preprocess_data(data_path)
        self.varnames = varnames
        self.randvars = randvars
        self.scaler = StandardScaler()

    @staticmethod
    def load_and_preprocess_data(data_path):
        df = pd.read_excel(data_path)
        df['Y'] = df['Y'].astype(int)
        df['price'] = df['price'].astype(float)
        df['dis'] = df['dis'].astype(float)
        df['feat'] = df['feat'].astype(float)
        df['FMYSize'] = df['FMYSize'].astype(float)
        df['Income'] = df['Income'].astype(float)
        return df

    def fit_model(self, n_draws):
        self.model = MixedLogit()
        self.model.fit(
            X=self.data[self.varnames],
            y=self.data['Y'],
            varnames=self.varnames,
            alts=self.data['brand'],
            ids=self.data['id'],
            panels=self.data['panelist'],
            randvars=self.randvars,
            isvars=['FMYSize', 'Income'],
            n_draws=n_draws,
            fit_intercept=True
        )
        self.model.summary()

    def predict(self, n_draws):
        predicted_choices, proba, freq = self.model.predict(
            X=self.data[self.varnames],
            varnames=self.varnames,
            ids=self.data['id'],
            alts=self.data['brand'],
            panels=self.data['panelist'],
            isvars=['FMYSize', 'Income'],
            n_draws=n_draws,
            return_proba=True,
            return_freq=True
        )
        return predicted_choices, proba

    @staticmethod
    def multiclass_brier_score_loss(true_labels, probability_estimates):
        n_classes = probability_estimates.shape[1]
        brier_scores = []
        for class_index in range(n_classes):
            true_binary = (true_labels == class_index).astype(int)
            preds_for_class = probability_estimates[:, class_index]
            brier_scores.append(brier_score_loss(true_binary, preds_for_class))
        return np.mean(brier_scores)

    def cross_validate(self, n_draws):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        brier_scores = []
        all_predicted_choices = []
        all_actual_choices = []

        for train_index, test_index in kf.split(self.data['id'].unique()):
            train_ids = self.data['id'].unique()[train_index]
            test_ids = self.data['id'].unique()[test_index]

            train_df = self.data[self.data['id'].isin(train_ids)].copy()
            test_df = self.data[self.data['id'].isin(test_ids)].copy()

            train_df[self.varnames] = self.scaler.fit_transform(train_df[self.varnames]).astype(np.float64)
            test_df[self.varnames] = self.scaler.transform(test_df[self.varnames]).astype(np.float64)

            model = MixedLogit()
            model.fit(
                X=train_df[self.varnames],
                y=train_df['Y'],
                varnames=self.varnames,
                alts=train_df['brand'],
                ids=train_df['id'],
                panels=train_df['panelist'],
                randvars=self.randvars,
                isvars=['FMYSize', 'Income'],
                n_draws=n_draws,
                fit_intercept=True
            )

            predicted_choices, proba = model.predict(
                X=test_df[self.varnames],
                varnames=self.varnames,
                ids=test_df['id'],
                alts=test_df['brand'],
                panels=test_df['panelist'],
                isvars=['FMYSize', 'Income'],
                n_draws=n_draws,
                return_proba=True,
                return_freq=False
            )

            chosen_brands = test_df[test_df['Y'] == 1]
            actual_choices = chosen_brands['brand']
            all_actual_choices.extend(actual_choices)
            all_predicted_choices.extend(predicted_choices)

            accuracy = np.mean(predicted_choices == actual_choices)
            accuracies.append(accuracy)

            choice_matrix = test_df.pivot(index='id', columns='brand', values='Y')
            actual_choices_array = choice_matrix.idxmax(axis=1).to_numpy()
            current_brier_score = self.multiclass_brier_score_loss(actual_choices_array, proba)
            brier_scores.append(current_brier_score)

        final_confusion_matrix = confusion_matrix(all_actual_choices, all_predicted_choices, labels=self.data['brand'].unique())
        return accuracies, brier_scores, final_confusion_matrix

    def plot_confusion_matrix(self, final_confusion_matrix):
        labels = np.sort(self.data['brand'].unique())
        plt.figure(figsize=(10, 7))
        sns.heatmap(final_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def export_results_to_excel(self, results, file_path):
        with ExcelWriter(file_path) as writer:
            for n_draws, (mean_accuracy, sd_accuracy, mean_brier, sd_brier, confusion_matrix) in results.items():
                result_df = pd.DataFrame({
                    'Metric': ['Mean Accuracy', 'SD Accuracy', 'Mean Brier Score', 'SD Brier Score'],
                    'Value': [mean_accuracy, sd_accuracy, mean_brier, sd_brier]
                })
                result_df.to_excel(writer, sheet_name=f'{n_draws}_draws_summary', index=False)
                confusion_df = pd.DataFrame(confusion_matrix, index=self.data['brand'].unique(), columns=self.data['brand'].unique())
                confusion_df.to_excel(writer, sheet_name=f'{n_draws}_draws_confusion_matrix')


def main():
    results = {}
    data_path = "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/year 3/Block 5/QM/data/data_purchase_long_Iryna.xlsx"
    varnames = ['price', 'dis', 'feat', 'FMYSize', 'Income', 'discount', 'lagged_choice', 'brand_loyalty']
    randvars = {'price': 'ln', 'dis': 'ln', 'feat': 'ln', 'discount': 'ln', 'lagged_choice': 'n', 'brand_loyalty': 'n'}
    model = MixedLogitModel(data_path, varnames, randvars)

    for n_draws in [100, 500, 1000]:
        print(f"Running model with {n_draws} draws...")
        model.fit_model(n_draws)
        accuracies, brier_scores, final_confusion_matrix = model.cross_validate(n_draws)
        
        mean_accuracy = np.mean(accuracies)
        sd_accuracy = np.std(accuracies)
        mean_brier = np.mean(brier_scores)
        sd_brier = np.std(brier_scores)
        
        print(f"Mean Prediction Accuracy for {n_draws} draws: {mean_accuracy} and standard deviation: {sd_accuracy}")
        print(f"Mean Brier score for {n_draws} draws: {mean_brier} and standard deviation: {sd_brier}")
        
        results[n_draws] = (mean_accuracy, sd_accuracy, mean_brier, sd_brier, final_confusion_matrix)

    model.export_results_to_excel(results, 'mixed_logit_results_final.xlsx')

if __name__ == "__main__":
    main()

'''
    if __name__ == "__main__":
    results = {}
    data_path = "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/year 3/Block 5/QM/data/data_purchase_long_Iryna.xlsx"
    varnames = ['price', 'dis', 'feat', 'FMYSize', 'Income', 'discount', 'lagged_choice', 'brand_loyalty']
    randvars = {'price': 'ln', 'dis': 'ln', 'feat': 'ln', 'discount': 'ln', 'lagged_choice': 'n', 'brand_loyalty': 'n'}
    model = MixedLogitModel(data_path, varnames, randvars)
    n_draws = 100
    print(f"Running model with {100} draws...")
    model.fit_model(n_draws)
    accuracies, brier_scores, final_confusion_matrix = model.cross_validate(n_draws)
        
    mean_accuracy = np.mean(accuracies)
    sd_accuracy = np.std(accuracies)
    mean_brier = np.mean(brier_scores)
    sd_brier = np.std(brier_scores)
        
    print(f"Mean Prediction Accuracy for {n_draws} draws: {mean_accuracy} and standard deviation: {sd_accuracy}")
    print(f"Mean Brier score for {n_draws} draws: {mean_brier} and standard deviation: {sd_brier}")
        
    model.plot_confusion_matrix(final_confusion_matrix)


    results[n_draws] = (mean_accuracy, sd_accuracy, mean_brier, sd_brier, final_confusion_matrix)

    model.export_results_to_excel(results, 'mixed_logit_results.xlsx')

'''
