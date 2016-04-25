from nltk.tokenize import WordPunctTokenizer
import nltk.data
import numpy as np
import re
import os

root = os.path.dirname(os.path.abspath(__file__))

##################
# TEXTS INVOLVED #
##################
##Alexandre Dumas
# 0:The Three Musketeers
# 1:Twenty Years After (D'Artagnan Series: Part Two)
# 2:The Count of Monte Cristo
##Mark Twain
# 3:Adventures of Huckleberry Finn
# 4:The American Claimant
##Jules Verne
# 5:Around the World in 80 Days
# 6:Twenty Thousand Leagues Under the Sea
##################

# These pull out the core text of their respective stories.
rulesStory = [
	r'our history\.\n{5}(.*)\s+----',
	r'Conclusion\.\n{5}(.*)\s+----',
	r', Pere\n{5}(.*)\n{6}End of',
	r'years ago\n{5}(.*)THE END\. YOURS TRULY, HUCK FINN\.',
	r'goes along.\n{6}(.*)\n{6}APPENDIX',
	r'\n{5}(.*)\n{10}',
	r'\n{6}(.*)\n{10}'
	]
	
# These represent meta elements of the text that must be stripped out, e.g. chapter headings.
rulesMeta = [
	r'\n(\d+.*)\n',
	r'\n(\d+\..*)\n',
	r'\n(Chapter \d+\..*)\n',
	r'\n(Chapter [XVIL]+\.)\n',
	r'\n(Chapter [XVIL]+\.)\n',
	r'\n{2}(Chapter [XVIL]+)\n',
	r'\n{2}(Chapter [XVIL]+)\n'
	]

def getText(idx):
	file = open(root+'/'+str(idx)+'.book', encoding='utf8').read()
	m = re.search(rulesStory[idx],re.sub(rulesMeta[idx], '', file),re.DOTALL)
	if m:
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		text = [WordPunctTokenizer().tokenize(s) for s in tokenizer.tokenize(m.group(1).rstrip().replace('\n', ' '))]
		t = []
		for sentence in text:
			s = []
			for word in sentence:
				r = re.search(r'(-|.)(-|")', word)
				s+=[r.group(1),r.group(2)] if r else [word]
			t+=[s]
		return t
		# return([w for s in t for w in s if w not in '.,:;()!?"\'_-'])
	else:
		raise Exception('Story regex failure in '+str(idx)+'.')

def getFuzzyList(word):
	return [word, word.lower()]+\
		([word[:-1], word[:-1].lower()] if word[-1] == 's' else [])+\
		([word[:-2], word[:-2].lower()] if word[-2:] == 'ed' else [])+\
		([word[:-2], word[:-2].lower()] if word[-2:] == 'er' else [])+\
		([word[:-3], word[:-3].lower()] if word[-3:] == 'ing' else [])+\
		([word[:-3]+'y', word[:-3].lower()+'y'] if word[-3:] == 'ied' else [])
		
def getFuzzyMatch(word, dict):
	for w in getFuzzyList(word):
		if w in dict:
			return w
	return None

def isAdmissible(sentence, dict):
	for word in sentence:
		if not getFuzzyMatch(word, dict):
			return False
	return True

def rate(pos, neg):
	return pos/(pos+neg)

#This sampling code taken from lstm_example.py in the Keras examples subfolder
def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
