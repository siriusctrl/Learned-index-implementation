# This is a abstract model that regular the format of every other models.
# Any other concrete model should extend this class.

class Model():

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        X is a list of string, y is bitarray that only has either 0 or 1
        to indicate the existence of corresponding string in X.
        """
        pass

    def predict(self, items):
        """
        run precition on a list of strings and return 0 or 1 as labels
        """
        pass