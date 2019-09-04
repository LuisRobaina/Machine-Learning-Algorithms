import random


###
# Implementing a Random Expression Tree (RET).
# A RET is composed of Expression Nodes, these Nodes can take any symbol from a set of real numbers,
# symbols based on the input data as well as arithmetic operations.
#

class _ExpressionNode:
    operations = ['*', '/', '+', '-']

    def __init__(self, symbol):
        self.symbol = symbol
        self.left: _ExpressionNode = None
        self.right: _ExpressionNode = None

    @staticmethod
    def is_operation(symbol):
        return symbol in _ExpressionNode.operations


class _RandomExpressionTree:

    def __init__(self, symbol):
        """Head Node must always be an operation."""
        if _ExpressionNode.is_operation(symbol):
            self.head: _ExpressionNode = _ExpressionNode(symbol)
        else:
            self.head = None

    def insert_node(self, symbol):
        """
        To insert a new symbol into the expression Tree, randomnly walk the Tree until a node with missing children is
        found. Add the new symbol there.
        Note: There is no guarantee that this Tree will represent a valid operation but that is fine for our purposes.
        """
        node: _ExpressionNode = self.head
        if node is None: return
        while True:
            coin = random.random()
            if (coin <= 0.5):
                if (node.left == None):
                    node.left = _ExpressionNode(symbol)
                    return
                node = node.left
            else:
                if (node.right == None):
                    node.right = _ExpressionNode(symbol)
                    return
                node = node.right

    @staticmethod
    def _print_tree(node):
        exp = ''

        if node is None:
            exp += ''
            return exp

        op = False

        if _ExpressionNode.is_operation(node.symbol):
            exp += '('
            op = True

        exp += str(_RandomExpressionTree._print_tree(node.left))
        exp += str(node.symbol)
        exp += str(_RandomExpressionTree._print_tree(node.right))

        if op:
            exp += ')'

        return exp

    def __repr__(self):
        return _RandomExpressionTree._print_tree(self.head)

    @staticmethod
    def evaluate(node, **kwargs):

        if not _ExpressionNode.is_operation(node.symbol):

            # Is it a number?
            if isinstance(node.symbol, int):
                return node.symbol

            # Is it a variable based on the data.
            if node.symbol in kwargs:
                return kwargs[node.symbol]

        if node.symbol == '*':
            return _RandomExpressionTree.evaluate(node.left, **kwargs) * _RandomExpressionTree.evaluate(node.right,
                                                                                                        **kwargs)
        elif node.symbol == '/':
            # The result of a valid sub-tree evaluation will not yield zero as it was taken care in the validation code.
            return _RandomExpressionTree.evaluate(node.left, **kwargs) / _RandomExpressionTree.evaluate(node.right,
                                                                                                        **kwargs)
        elif node.symbol == '+':
            return _RandomExpressionTree.evaluate(node.left, **kwargs) + _RandomExpressionTree.evaluate(node.right,
                                                                                                        **kwargs)
        else:
            return _RandomExpressionTree.evaluate(node.left, **kwargs) - _RandomExpressionTree.evaluate(node.right,
                                                                                                        **kwargs)

    def valid_tree(self, node: _ExpressionNode, **kwargs) -> bool:
        """
        Ensures the RET represents a valid mathematical expression.
        An optimization function could assign a 0 fitness to an invalid
        RET that way we can ignore them as the expressions evolve.
        """

        # Check the head node (Must be an operation).
        if node is self.head and node is None:
            return False

        # Assume the RET is valid.
        valid = True

        # Base case for the recursive method.
        if node is None:
            return True

        # Rule 1. Head node must be an operation.
        elif node is self.head and not _ExpressionNode.is_operation(node.symbol):
            valid = False

        # Rule 2. Operands cannot act on operands.
        elif not _ExpressionNode.is_operation(node.symbol):
            if node.left is not None and not _ExpressionNode.is_operation(node.left.symbol):
                valid = False
            if node.right is not None and not _ExpressionNode.is_operation(node.right.symbol):
                valid = False

        # Rule 3. Operations must contain two operands, Div by zero is undefined.
        elif _ExpressionNode.is_operation(node.symbol):
            if node.left is None:
                valid = False
            elif node.right is None:
                valid = False
            elif node.symbol == '/':

                if node.right.symbol == 0:
                    valid = False
                    # At this point we have to evaluate the sub-tree to the right of the division node
                    # To ensure it does not evaluate to zero given the variables at kwargs.
                else:
                    sub_tree = _RandomExpressionTree(node.right.symbol)
                    if not sub_tree.valid_tree(sub_tree.head, **kwargs) or _RandomExpressionTree.evaluate(node.right,
                                                                                                          **kwargs) == 0:
                        valid = False

        valid = valid and self.valid_tree(node.left)
        if not valid: return False
        valid = valid and self.valid_tree(node.right)
        if not valid: return False

        return valid


###
# Demo:
operations = ['*', '/', '+', '-']
operations.extend([x for x in range(5)])
operations.extend(['m', 'a'])
import time

# Generate n RET, print and evaluate those that are valid.
for i in range(50000):
    k = random.randint(0, len(operations) - 1)
    T = _RandomExpressionTree(operations[int(k)])
    for q in range(random.randint(2, 8)):
        k = random.randint(0, len(operations) - 1)
        T.insert_node(operations[int(k)])

    if T.valid_tree(T.head, a=10, m=5):
        try:
            print(T, '=')
            result = _RandomExpressionTree.evaluate(T.head, a=10, m=5)
            print(result)
        except Exception as ex:
            print(ex)
            time.sleep(10)
        print('----------------')
