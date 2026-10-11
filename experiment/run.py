"""The 'toy research code'. Trains logistic regression on scikit-learn's built-in Digits data.

Usage: python run.py --seed 42 --out metrics.json
The seed controls the train/test split, so different seeds give slightly different accuracy.
"""
import argparse
import json
import warnings

warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="metrics.json")
args = parser.parse_args()

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
accuracy = round(100 * model.score(X_test, y_test), 2)

with open(args.out, "w") as handle:
    json.dump({"accuracy": accuracy, "seed": args.seed, "train_size": len(X_train), "test_size": len(X_test)}, handle)
print(f"seed={args.seed} accuracy={accuracy}%")
