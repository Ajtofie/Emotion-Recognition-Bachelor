from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pprint import pprint


def find_best_hyperparams():
    rf = RandomForestClassifier(random_state=42)

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(rf.get_params())
