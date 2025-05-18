import pickle
import os

class DeepNeuralNetwork:
    # existing methods ...

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        If filename does not have .pkl extension, add it.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        Returns the loaded object, or None if filename doesn’t exist.
        """
        if not os.path.isfile(filename):
            return None
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception:
            return None
