def feature_extraction(obj):
	feature_row = []


	#number of words in title
	nWordTitle = find_title_words(obj['targetTitle'])
	feature_row.append(nWordTitle)


	#number of words in body text
	nWordText = find_text_words(obj['targetParagraphs'])
	feature_row.append(nWordText)


	#is title a question?
	titleQuestion = is_question(obj['targetTitle'])
	feature_row.append(titleQuestion)


	#number of questions in title
	nQuestions = find_questions_in_title(obj['targetTitle'])
	feature_row.append(nQuestions)


	#number of bait words classically recognizing as clickbait "this” or “happens next” or “see inside” or “things”
	nBaits = find_number_baits(obj['targetTitle'])
	feature_row.append(nBaits)


	#bait score defined by number of baits in the title per word in the title
	baitScore = find_bait_score(obj['targetTitle'])
	feature_row.append(baitScore)


	#context words in title
	contextTitleCount = find_context_title(obj['targetTitle'])
	feature_row.append(contextTitleCount)


	#context word score in title, number of context words per word in the title
	contextTitleScore = find_context_score_title(obj['targetTitle'], contextTitleCount)
	feature_row.append(contextTitleScore)


	#context words in body
	contextBodyCount = find_context_body(obj['targetParagraphs'])
	feature_row.append(contextBodyCount)

	#context word score in body, number of context words per word in the body
	contextBodyScore = find_context_score_body(obj['targetParagraphs'], contextBodyCount)
	feature_row.append(contextBodyScore)

	#pronoun count title
	pronounTitleCount = pronoun_count(obj['targetTitle'])
	feature_row.append(pronounTitleCount)

	#pronoun score, number of pronouns per word, title
	pronounTitleScore = pronoun_score(obj['targetTitle'], pronounTitleCount)
	feature_row.append(pronounTitleScore)

	#pronoun count body
	pronounBodyCount = pronoun_count(obj['targetParagraphs'])
	feature_row.append(pronounBodyCount)

	#pronoun score body
	pronounBodyScore = pronoun_score(obj['targetParagraphs'], pronounBodyCount)
	feature_row.append(pronounBodyScore)

	#twitter handle
	twitter = find_twitter_handles(obj['targetParagraphs'])
	feature_row.append(twitter)

	#number of characters in title
	chartitle = characters_title(obj['targetTitle'])
	feature_row.append(chartitle)

	#numbers in title
	numsinTitle = numbers_in_title(obj['targetTitle'])
	feature_row.append(numsinTitle)

	#number of sentences in paragraph
	nSentences = find_sentences_paragraph(obj['targetParagraphs'])
	feature_row.append(nSentences)

	return feature_row #row of features for the input sample


#order of features in feature_row vector: 
#1. number of words in title
#2. number of words in text
#3. is title a question?
#4. number of questions in title
#5. number of bait words 
#6. bait score
#7. context word number title
#8. context word score title
#7. context word number body
#8. context word score body
#9. pronoun count title
#10. pronoun score title
#11. pronoun count body
#12. pronoun score body
#13. twitter handle in body
#14. length of title
#15. numbers in the title
#16. number of sentences in paragraph


def find_title_words(title):
	return title.count(' ')+1

def find_text_words(paragraph):
	nW = 0
	for p in paragraph:
		nW += p.count(' ')+1

	return nW

def is_question(title):
	if title.count('?'):
		return 1
	else:
		return 0

def find_questions_in_title(title):
	return title.count('?')

def find_number_baits(title):
	return title.count("this")+title.count("happens")+title.count("next")+title.count("see")+title.count("inside")+title.count("things")

def find_bait_score(title):
	return find_number_baits(title)/find_title_words(title)

def find_context_title(title):
	fp = open("contextwords.txt", "r") #read in context words
	A = fp.read()
	B = A.split()
	nContext = 0
	for contextword in B:
		nContext += title.count(contextword)

	return nContext


def find_context_score_title(title, titlecount):
	#title might be empty, this makes sure that there is no divide by 0 error and just returns a 0 so as not to impact the rest of our features
	if find_title_words(title):
		return titlecount/find_title_words(title)
	else:
		return 0
	
def find_context_body(paragraph):
	fp = open("contextwords.txt", "r") #read in context words
	A = fp.read()
	B = A.split()
	nContextP = 0
	for p in paragraph:
		for contextword in B:
			nContextP += p.count(contextword)

	return nContextP
	
def find_context_score_body(paragraph, bodycount):
	#sometimes the paragraph text is empty, just due to formatting. This will just return 0 so it does not impact the rest of our samples
	if find_text_words(paragraph):
		return bodycount/find_text_words(paragraph)
	else:
		return 0 
	
def pronoun_count(text):
	fp = open("pronouns.txt", 'r')
	pronouns = fp.read().split()
	nP = 0
	for t in text:
		for pronoun in pronouns:
			nP+=t.count(pronoun)

	return nP

def pronoun_score(text, count):
	if find_text_words(text):
		return count/find_text_words(text)
	else:
		return 0
	
## Meredith features
def find_twitter_handles(paragraph):
	atcount = 0
	for p in paragraph:
		if p.count('@'):
			atcount+=p.count('@')

	return atcount

def characters_title(title):
	cT = len(title)
	return cT

def numbers_in_title(title):
	return any(char.isdigit() for char in title)

def find_sentences_paragraph(paragraph):
	#pcount = 0
	#for p in paragraph:
	#	if paragraph.count('.'):
	#		pcount+=paragraph.count('.')

	return len(paragraph)


