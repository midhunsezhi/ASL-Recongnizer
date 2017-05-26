import warnings
from asl_data import SinglesData
import math


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for i in range(0, test_set.num_items):

        local_probabilities = dict()
        for word, model in models.items():
            logL = -math.inf
            X, lengths = test_set.get_item_Xlengths(i)
            try:
                if model is not None:
                    logL = model.score(X, lengths)
            except:
                logL = -math.inf

            local_probabilities[word] = logL

        probabilities.append(local_probabilities)
        best_guess = max(local_probabilities, key=local_probabilities.get)
        guesses.append(best_guess)

    return probabilities, guesses
