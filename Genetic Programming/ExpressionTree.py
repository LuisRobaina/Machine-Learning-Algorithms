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
        else: self.head = None

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
            return ''

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

    def valid_tree(self,node: _ExpressionNode) -> bool:

        """
        Defines a RET that represents a valid mathematical expression.
        An optimization function could assign a 0 fitness to an invalid
        RET that way we can ignore them as the expressions evolve.
        """

        if node is self.head and node is None:
            return False

        valid = True

        if node is None:
            return True

        # Rule 1. Head node must be an operation
        elif node is self.head and not _ExpressionNode.is_operation(node.symbol):
            valid = False

        # Rule 2. Operands cannot act on operands.
        elif not _ExpressionNode.is_operation(node.symbol):
            if node.left is not None and not _ExpressionNode.is_operation(node.left.symbol):
                valid = False
            if node.right is not None and not _ExpressionNode.is_operation(node.right.symbol):
                valid = False

        # Rule 3. Operations must contain two operands.
        elif _ExpressionNode.is_operation(node.symbol):
            if node.left is None or node.right is None:
                valid = False

        # Rule 4. Division by zero. (For the case the number exists and is not a result of further operations.
        elif node.symbol == '/' and node.right is not None and node.right.symbol == 0:
            valid = False

        valid = valid and self.valid_tree(node.left)
        if not valid: return False
        valid = valid and self.valid_tree(node.right)
        if not valid: return False

        return valid

    def evaluate(self):
        pass

###
# Demo:

operations = ['*', '/', '+', '-']
operations.extend([x for x in range(5)])
operations.extend(['m','a'])

# Generate 600 RET with 5 Nodes, print those that are valid.

for i in range(600):
    k = random.randint(0, len(operations)-1)
    T = _RandomExpressionTree(operations[int(k)])
    for q in range(4):
        k = random.randint(0,len(operations)-1)
        T.insert_node(operations[int(k)])
    if T.valid_tree(T.head):
        print(T)
        print('----------------')
