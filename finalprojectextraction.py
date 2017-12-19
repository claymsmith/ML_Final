import finalprojectfunctions
import jsonlines as jsl
import numpy as np
import numpy.matlib
import pickle

features = []
features0 = []
ytr = []
yts = []

with jsl.open("clickbait17-train-170331\clickbait17-train-170331\instances.jsonl") as reader:
    for obj in reader:
    	samp_features = finalprojectfunctions.feature_extraction(obj)
    	features.append(samp_features)

with jsl.open("clickbait17-test-170630\clickbait17-validation-170630\instances.jsonl") as reader:
    for obj in reader:
    	samp_features0 = finalprojectfunctions.feature_extraction(obj)
    	features0.append(samp_features0)

with jsl.open("clickbait17-train-170331\clickbait17-train-170331/truth.jsonl") as reader:
	for obj in reader:
		if obj['truthClass']=="clickbait":
			ytr.append(1)
		else:
			ytr.append(0)

with jsl.open("clickbait17-test-170630\clickbait17-validation-170630/truth.jsonl") as reader:
	for obj in reader:
		if obj['truthClass']=="clickbait":
			yts.append(1)
		else:
			yts.append(0)


Xtr = np.vstack(tuple(features))
ytr = np.array(ytr)
Xts = np.vstack(tuple(features0))
yts = np.array(yts)

pickle.dump([Xtr, ytr], open("Xtr_ytr_MnC.p", "wb"))
pickle.dump([Xts, yts], open("Xts_yts_MnC.p", "wb"))