from sklearn import datasets
from sklearn.model_selection import train_test_split
import DecisionTrees

dataset = datasets.load_wine()
X = dataset['data']
Y = dataset['target']

train_data = []
test_data = []

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.5)

for index, row in enumerate(X_Train):
    row = row.tolist()
    row.append(Y_Train[index])
    DecisionTrees.data_headers.append('Class'+str(index))
    train_data.append(row)

for index, row in enumerate(X_Test):
    row = row.tolist()
    row.append(Y_Test[index])
    test_data.append(row)


# Learn Tree structure.
T = DecisionTrees.build_tree(train_data)
DecisionTrees.printTree(T)
# Test Classifier
for test_sample in test_data:
    print('Predicted: ', DecisionTrees.run_classifier(test_data,T), 'Real', test_sample[-1])
