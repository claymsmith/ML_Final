import jsonlines as jsl
import numpy as np
import pickle
y = []
allwords = []
maxL = 0
nElem = 0
with jsl.open("clickbait17-validation-170630\instances.jsonl") as reader:
    for obj in reader:
        current_title = obj['targetTitle'].split()
        allwords.extend(current_title)
        if len(current_title)>maxL:
            maxL = len(current_title)

        nElem +=1

uWords = np.unique(allwords)
char_to_ix = {char:ix for ix, char in enumerate(uWords)} #our dictionary
X = np.zeros((nElem, maxL))
vsize = len(uWords)

nIt = 0
with jsl.open("clickbait17-validation-170630\instances.jsonl") as reader:
    for obj in reader:
        current_title = obj['targetTitle'].split()
        X_idx = [char_to_ix[value] for value in current_title]
        X[nIt, :len(X_idx)] = X_idx
        nIt+=1


with jsl.open("clickbait17-validation-170630/truth.jsonl") as reader:
	for obj in reader:
		if obj['truthClass']=="clickbait":
			y.append(1)
		else:
			y.append(0)
y = np.array(y)

#normalize X between -1 and 1
X = 2*X/(vsize-1)-1

pickle.dump((X, y, vsize), open('X_y_word_dict.p', 'wb'))
