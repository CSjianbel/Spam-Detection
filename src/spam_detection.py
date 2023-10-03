import os

import pandas as pd

from naive_bayes import NaiveBayes


""" CONSTANTS """
DATA_DIR = 'data'
TRAINING_FILE = 'TrainingData.csv'
TESTING_FILE = 'TestData.csv'
RESULT_FILE = 'BelarminoResultData.csv'
LABELED_FILE = 'LabeledTestData.csv'


""" MAIN FUNCTION """
def main():

    # Read training data
    training_data = pd.read_csv(os.path.join(DATA_DIR, TRAINING_FILE))

    # Create NaiveBayes Model
    model = NaiveBayes()

    model.fit(training_data)

    #  Read testing data
    testing_data = pd.read_csv(os.path.join(DATA_DIR, TESTING_FILE))
    result_data = model.test(testing_data)

    # Generate result data
    model.generateTestResult(result_data, os.path.join(DATA_DIR, RESULT_FILE))

    # Read labeled data
    labeled_data = pd.read_csv(os.path.join(DATA_DIR, LABELED_FILE))
    model.grade(result_data, labeled_data)


if __name__ == "__main__":
    main()
