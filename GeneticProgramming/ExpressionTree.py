import random


###
# Implementing a Random Expression Tree (RET).
# A RET is composed of Expression Nodes, these Nodes can take any symbol from a set of real numbers,
# symbols based on the input data as well as arithmetic operations.
###

class ExpressionNode:
    arithmetic_operations = ['*', '/', '+', '-']
    operations = []

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.left: ExpressionNode = None
        self.right: ExpressionNode = None

    def _copy(self, node):
        self.symbol = node.symbol
        self.left: ExpressionNode = node.left
        self.right: ExpressionNode = node.right

    @staticmethod
    def is_operation(symbol):
        return symbol in ExpressionNode.arithmetic_operations

    @staticmethod
    def rand_symbol():
        s = random.randint(0, len(ExpressionNode.operations) - 1)
        return ExpressionNode.operations[s]

    @staticmethod
    def extend_symbols(vars: [any], k_range: int):
        ExpressionNode.operations.extend(ExpressionNode.arithmetic_operations)
        ExpressionNode.operations.extend(vars)
        ExpressionNode.operations.extend([x for x in range(k_range + 1)])
        random.shuffle(ExpressionNode.operations)


class RandomExpressionTree:

    def __init__(self, **kwargs):
        """Head Node must always be an operation."""
        self.nodes_count = 0
        if 'symbol' in kwargs:
            self.root = ExpressionNode(kwargs['symbol'])
        else:
            k = random.randint(0, len(ExpressionNode.arithmetic_operations) - 1)
            symbol = ExpressionNode.arithmetic_operations[int(k)]
            self.root = ExpressionNode(symbol)

    def insert_node(self, symbol):
        """
        To insert a new symbol into the expression Tree, randomnly walk the Tree until a node with missing children is
        found. Add the new symbol there.
        Note: There is no guarantee that this Tree will represent a valid operation but that is fine for our purposes.
        """
        node: ExpressionNode = self.root
        if node is None: return

        while True:
            coin = random.random()
            if (coin <= 0.5):
                if (node.left == None):
                    node.left = ExpressionNode(symbol)
                    self.nodes_count += 1
                    return
                node = node.left
            else:
                if (node.right == None):
                    node.right = ExpressionNode(symbol)
                    self.nodes_count += 1
                    return
                node = node.right

    @staticmethod
    def _print_tree(node):
        exp = ''

        if node is None:
            exp += ''
            return exp

        op = False

        if ExpressionNode.is_operation(node.symbol):
            exp += '('
            op = True

        exp += str(RandomExpressionTree._print_tree(node.left))
        exp += str(node.symbol)
        exp += str(RandomExpressionTree._print_tree(node.right))

        if op:
            exp += ')'

        return exp

    def __repr__(self):
        return RandomExpressionTree._print_tree(self.root)

    @staticmethod
    def evaluate(node, **kwargs):

        if not ExpressionNode.is_operation(node.symbol):

            # Is it a number?
            if isinstance(node.symbol, int):
                return node.symbol

            # Is it a variable based on the data.
            if node.symbol in kwargs:
                return kwargs[node.symbol]

        if node.symbol == '*':
            return RandomExpressionTree.evaluate(node.left, **kwargs) * RandomExpressionTree.evaluate(node.right,
                                                                                                      **kwargs)
        elif node.symbol == '/':
            # The result of a valid sub-tree evaluation will not yield zero as it was taken care in the validation code.
            return RandomExpressionTree.evaluate(node.left, **kwargs) / RandomExpressionTree.evaluate(node.right,
                                                                                                      **kwargs)
        elif node.symbol == '+':
            return RandomExpressionTree.evaluate(node.left, **kwargs) + RandomExpressionTree.evaluate(node.right,
                                                                                                      **kwargs)
        else:
            return RandomExpressionTree.evaluate(node.left, **kwargs) - RandomExpressionTree.evaluate(node.right,
                                                                                                      **kwargs)

    def valid_tree(self, node: ExpressionNode, **kwargs) -> bool:
        """
        Ensures the RET represents a valid mathematical expression.
        An optimization function could assign a 0 fitness to an invalid
        RET that way we can ignore them as the expressions evolve.
        """

        # Check the root node (Must be an operation).
        if node is self.root and node is None:
            return False

        # Assume the RET is valid.
        valid = True

        # Base case for the recursive method.
        if node is None:
            return True

        # Rule 1. Head node must be an operation.
        elif node is self.root and not ExpressionNode.is_operation(node.symbol):
            valid = False

        # Rule 2. Operands cannot act on operands.
        elif not ExpressionNode.is_operation(node.symbol):
            if node.left is not None and not ExpressionNode.is_operation(node.left.symbol):
                valid = False
            if node.right is not None and not ExpressionNode.is_operation(node.right.symbol):
                valid = False

        # Rule 3. Operations must contain two operands, Div by zero is undefined.
        elif ExpressionNode.is_operation(node.symbol):
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
                    sub_tree = RandomExpressionTree(symbol=node.right.symbol)
                    if not sub_tree.valid_tree(sub_tree.root, **kwargs) or RandomExpressionTree.evaluate(node.right,
                                                                                                         **kwargs) == 0:
                        valid = False

        valid = valid and self.valid_tree(node.left)
        if not valid: return False
        valid = valid and self.valid_tree(node.right)
        if not valid: return False

        return valid

###
# Demo:
# ---------------------------------------------------------

# # Extends the basic set of arithmetic operations to contain:
# # vars : Set of variables in the input data
# # k_range : Range of constants allowed in the RET.
#
# ExpressionNode.extend_symbols(vars = ['a','m'], k_range=5)
#
# # Generate n RETs, print and evaluate those that are valid.
# for i in range(500000):
#     T = RandomExpressionTree()
#     for q in range(random.randint(2, 16)):
#         symbol = ExpressionNode.rand_symbol()
#         T.insert_node(symbol)
#
#     if T.valid_tree(T.root, a=10, m=5):
#         print(T, '=')
#         result = RandomExpressionTree.evaluate(T.root, a=10, m=5)
#         print(result)
#         print('-------------')
