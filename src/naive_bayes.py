import string
from typing import List

import pandas as pd


class NaiveBayes:

    def __init__(self) -> None:
        self.__k = 0.1
        self.__spam_count = 0
        self.__ham_count = 0
        self.__total_word_count = 0
        self.__spam_prob = 0.0
        self.__ham_prob = 0.0
        self.__word_dictionary = {}

    def clean_message(self, message: str) -> List[str]:
        message = message.lower()
        result = message.translate(str.maketrans(
            '', '', string.punctuation)).split()
        return [word for word in result if word.isalpha()]

    def fit(self, training_data: pd.DataFrame) -> None:
        print("Fitting model...")
        for _, row in training_data.iterrows():
            # clean the message
            message = self.clean_message(row['message'])

            for word in message:
                if word not in self.__word_dictionary:
                    self.__word_dictionary[word] = {'spam': 0, 'ham': 0}
                self.__word_dictionary[word][row['label']] += 1

        self.__total_word_count = len(self.__word_dictionary)
        self.__spam_count = len(
            [1 for word in self.__word_dictionary if self.__word_dictionary[word]['spam'] != 0])
        self.__ham_count = len(
            [word for word in self.__word_dictionary if self.__word_dictionary[word]['ham'] != 0])

        self.__spam_prob = self.__spam_count / self.__total_word_count
        self.__ham_prob = self.__ham_count / self.__total_word_count

    def predict(self, message: str, ) -> str:
        message = self.clean_message(message)

        pm_spam, pm_ham = 1.0, 1.0

        for word in message:
            pm_spam *= ((self.__word_dictionary[word]['spam'] if word in self.__word_dictionary else 0) + self.__k) / (
                self.__spam_count + self.__total_word_count * self.__k)
            pm_ham *= ((self.__word_dictionary[word]['ham'] if word in self.__word_dictionary else 0) + self.__k) / (
                self.__ham_count + self.__total_word_count * self.__k)

        pspam = (pm_spam * self.__spam_prob) / \
            ((pm_spam * self.__spam_prob) + (pm_ham * self.__ham_prob))

        pham = (pm_ham * self.__ham_prob) / \
            ((pm_spam * self.__spam_prob) + (pm_ham * self.__ham_prob))

        res = 'spam' if pspam > pham else 'ham'
        return res

    def test(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        print("Testing naive bayes model...")
        labels = []
        for _, row in testing_data.iterrows():
            labels.append(self.predict(row['message']))

        testing_data['label'] = labels
        return testing_data

    def generateTestResult(self, result_data: pd.DataFrame, path: str) -> None:
        print("Generating BelarminoResultData.csv...")
        result_data.to_csv(path)

    def grade(self, result_data: pd.DataFrame, labeled_data: pd.DataFrame) -> None:
        print("Grading performance of model on testing dataset...")
        TP, TN, FP, FN = 0, 0, 0, 0
        for ind in labeled_data.index:
            if labeled_data['label'][ind] == 'spam' and result_data['label'][ind] == 'spam':
                TP += 1
            elif labeled_data['label'][ind] == 'ham' and result_data['label'][ind] == 'ham':
                TN += 1
            elif labeled_data['label'][ind] == 'ham' and result_data['label'][ind] == 'spam':
                FP += 1
            else:
                FN += 1

        print('********RESULTS********')
        print(f'True Positives: {TP}')
        print(f'True Negatives: {TN}')
        print(f'False Positive: {FP}')
        print(f'False Negatives: {FN}', end='\n\n')

        print(f'Precision: {self.computePrecision(TP, FP) * 100.0}%')
        print(f'Recall: {self.computeRecall(TP, FN) * 100.0}%')

    def computePrecision(self, TP: int, FP: int) -> float:
        return TP / (TP + FP)

    def computeRecall(self, TP: int, FN: int) -> float:
        return TP / (TP + FN)
