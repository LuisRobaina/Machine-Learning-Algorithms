###
# ML Algorithms: Decision Tree.
# The code bellow implements a simple decision tree classifier.
# Will be implementing a CART (Classification and regression tree).

###

# Each row is an example formatted as:
# Feature, Feature, Label.

def unique_vals(dataset: [[any]], col: int) -> set:
    """Find unique values for a column (Avoids repeated features)"""
    return set([row[col] for row in dataset])


def class_count(dataset: [[any]]) -> dict:
    """Returns a dictionary with labels and the number of times it appears in the data-set"""
    count = {}  # a dictionary of label,count.
    for row in dataset:
        label = row[-1]  # Assumes label is at the last column.
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count


def is_numeric(value):
    """Test if a value is numeric"""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """
    A question is used to partition the dataset, questions work as a filter.
    a questions stores the index of the feature in the dataset as well as the value
    for that specific question being asked. Example Questions(0,'Red') in the training data
    above is equivalent to asking 'Is the color Red ?
    """

    def __init__(self, col, val):
        self.column = col
        self.value = val

    def match(self, example):
        """
        Given an example, is it a match to this question?
        An example is a match if its value to a specific feature satisfies the question
        """
        val = example[self.column]

        # If the feature is numeric, we are making thresholds by a greater-than-equal logic.
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        """Helper method to print a question readable"""
        logic = 'equal'
        if is_numeric(self.value):
            logic = ">="
        return "Is %s %s %s?" % (data_headers[self.column], logic, str(self.value))


def partition(data: [[any]], question: Question):
    """Partition the data by the given question"""
    true_samples, false_samples = [], []

    for sample in data:
        if question.match(sample):
            true_samples.append(sample)
        else:
            false_samples.append(sample)

    return true_samples, false_samples


def gini(data):
    """
    Calculate the Gini impurity for a set of examples in a given node of the tree
    Gini impurity measures the probability that a randomly chosen element in the dataset is labeled
    incorrectly if we chose a label in the same dataset also randomly. You can see that this impurity
    is 0 when all the samples have the same label (No matter what you chose the labeling will be correct).
    """

    # Gini Impurity is 1 - (The sum for every label (i) in the given node of the fraction of elements in the
    # node that are labeled (i).

    count = class_count(data)
    pk = 0
    for lbl in count:
        pk += (count[lbl] / len(data)) ** 2

    return 1 - pk


def information_gain(left_node, right_node, parent_impurity):
    """
    The impurity of the parent node minus the weighed impurity of the two child nodes.
    Find the question that helps reduce the impurity the most.
    """
    p = len(left_node) / (len(left_node) + len(right_node))
    i_gain = parent_impurity - (p * gini(left_node) - (1 - p) * gini(right_node))
    return i_gain


def find_best_split(data):
    """Find the best question to ask: The one that maximizes our infomrations gain."""
    best_gain = 0
    best_question = None
    current_impurity = gini(data)

    features_count = len(data[0]) - 1
    for feature in range(features_count):  # For each feature.
        values = set([row[feature] for row in data])
        for val in values:
            q = Question(feature, val)
            l, r = partition(data, q)

            if len(l) == 0 or len(r) == 0:
                continue
            gain = information_gain(l, r, current_impurity)
            if gain > best_gain:
                best_gain = gain
                best_question = q

    return best_question, best_gain


class Leaf:
    """
    A leaf classifies the input.
    Holds a dictionary of labels and the number of times it appears, in the data that reaches this
    leaf.
    """

    def __init__(self, data):
        self.prediction = class_count(data)


class Decision_Node:

    def __init__(self,
                 question,
                 true_node,
                 false_node):
        self.question = question
        self.true_node = true_node
        self.false_node = false_node


def build_tree(data):
    question, gain = find_best_split(data)
    if gain == 0:
        return Leaf(data)

    # If we reach here we found a useful feature to partition the data so we
    # attempt it.
    true_data, false_data = partition(data, question)

    true_branch = build_tree(true_data)
    false_branch = build_tree(false_data)

    return Decision_Node(question, true_branch, false_branch)

def printTree(headNode, index=''):

    if isinstance(headNode, Leaf):
        print(index + 'Predict =', headNode.prediction)
        return

    print(index + str(headNode.question))
    print(index + 'True')
    printTree(headNode.true_node,index+'  ')
    print(index + 'False')
    printTree(headNode.false_node, index + '  ')



def _classify(data, node: Decision_Node):
    if isinstance(node, Leaf):
        return node.prediction

    if node.question.match(data):
        return _classify(data, node.true_node)

    else:
        return _classify(data, node.false_node)


def run_classifier(input, T) -> dict:
    prediction = _classify(input, T)
    total = sum(prediction.values())
    probs = {}
    for lbl in prediction:
        probs[lbl] = str((prediction[lbl] / total) * 100) + "%"

    return probs

# Data headers will be set by the data used to train the classifier.
data_headers = []

#
# ################
# # Demo:
# input = ['Yellow', '3']
#
# training_data = [
#     ['Green', 3, 'Apple'],
#     ['Yellow', 3, 'Apple'],
#     ['Red', 1, 'Grape'],
#     ['Red', 1, 'Grape'],
#     ['Yellow', 3, 'Lemon'],
# ]
#
# T = build_tree(training_data)
#
#
# # Testing the classifier with the training data.
# for data_sample in training_data:
#     result = 'Actual %s, Predicted %s' % (data_sample[-1], str(run_classifier(data_sample, T)))
#     print(result)
