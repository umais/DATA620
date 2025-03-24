import nltk
import random
from nltk.corpus import names
from nltk.classify import apply_features
from nltk.classify import NaiveBayesClassifier

nltk.download('names')
# Load the names dataset
def gender_features(name):
    return {
        "last_letter": name[-1],
        "last_two": name[-2:],
        "last_three": name[-3:],
        "first_letter": name[0],
        "length": len(name),
        "vowel_count": sum(1 for char in name.lower() if char in 'aeiou')
    }

# Load and shuffle the labeled names dataset
labeled_names = [(name, 'male') for name in names.words('male.txt')] + \
                [(name, 'female') for name in names.words('female.txt')]
random.shuffle(labeled_names)

# Split the dataset into training, dev-test, and test sets
train_names = labeled_names[1000:]
dev_test_names = labeled_names[500:1000]
test_names = labeled_names[:500]

# Extract features for each dataset
train_set = [(gender_features(n), g) for (n, g) in train_names]
dev_test_set = [(gender_features(n), g) for (n, g) in dev_test_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]

# Train the classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate on dev-test set
dev_test_accuracy = nltk.classify.accuracy(classifier, dev_test_set)
print("Dev-test Accuracy:", dev_test_accuracy)

# Evaluate on test set
test_accuracy = nltk.classify.accuracy(classifier, test_set)
print("Test Accuracy:", test_accuracy)

# Show most informative features
classifier.show_most_informative_features(10)

def predict_gender(name):
    features = gender_features(name)
    return classifier.classify(features)

print(predict_gender("Alice"))  # Expected: 'female'
print(predict_gender("John"))   # Expected: 'male'
print(predict_gender("Kristy")) # Expected: 'Female'
print(predict_gender("Samantha")) # Expected: 'Female'
print(predict_gender("Taylor")) # Could be male or female