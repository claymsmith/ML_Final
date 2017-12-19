import jsonlines as jsl
import numpy as np
import pickle
#this uses the full text processing feature method using unique indices of letters found in https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

alltitles = []
ytr = []
yts = []
elemCount = 0
eCount0 = 0
maxL = 0
weirdInd = []



#get data max length and stuff for dictionary, so that it is the same between training and test data
with jsl.open("clickbait17-train-170331\clickbait17-train-170331\instances.jsonl") as reader:
    for obj in reader:
    	data = list(obj['targetTitle'])
    	alltitles.extend(data)
    	if (len(data)>maxL)&(len(data)<4000):
    		maxL = len(data)

with jsl.open("clickbait17-test-170630\clickbait17-validation-170630\instances.jsonl") as reader:
    for obj in reader:
    	data = list(obj['targetTitle'])
    	alltitles.extend(data)
    	if (len(data)>maxL)&(len(data)<4000):
    		maxL = len(data)

#dictionary for character to index of unique values
uniqueval = set(alltitles)
char_to_ix = {char:ix for ix, char in enumerate(uniqueval)}
Xtr = np.zeros((elemCount, maxL))
Xts = np.zeros((eCount0, maxL))
it = 0

with jsl.open("clickbait17-train-170331\clickbait17-train-170331\instances.jsonl") as reader0:
    for obj in reader0:
    	data0 = list(obj['targetTitle'])
    	indexes = [char_to_ix[value] for value in data0]
    	if (len(data0)<4000): #remove weird error one
    		Xtr[it, :len(indexes)] = indexes
    	else:
    		weirdInd.append(it)
    	it+=1

it = 0
with jsl.open("clickbait17-test-170630\clickbait17-validation-170630\instances.jsonl") as reader1:
    for obj in reader1:
    	data0 = list(obj['targetTitle'])
    	indexes = [char_to_ix[value] for value in data0]
    	if (len(data0)<4000):
    		Xts[it, :len(indexes)] = indexes
    	it+=1

with jsl.open("clickbait17-train-170331\clickbait17-train-170331/truth.jsonl") as reader:
	for obj in reader:
		elemCount+=1
		if obj['truthClass']=="clickbait":
			ytr.append(1)
		else:
			ytr.append(0)

with jsl.open("clickbait17-test-170630\clickbait17-validation-170630/truth.jsonl") as reader:
	for obj in reader:
		eCount0 +=1
		if obj['truthClass']=="clickbait":
			yts.append(1)
		else:
			yts.append(0)

ytr = np.array(ytr)
yts = np.array(yts)

#save xtr and ytr
pickle.dump( [Xtr, ytr, len(uniqueval)], open( "Xtr_ytr_full.p", "wb" ) )
pickle.dump( [Xts, yts], open("Xts_yts_full.p", "wb"))
print(Xtr.shape, ytr.shape)