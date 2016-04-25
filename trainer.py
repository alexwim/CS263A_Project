from keras.models import Sequential, model_from_json
from keras.layers.core import TimeDistributedDense, Dropout, Dense, Activation
from keras.layers.recurrent import LSTM
from utils import getText, isAdmissible, getFuzzyMatch, rate, sample
import numpy as np
import random
import sys
import os

root = os.path.dirname(os.path.abspath(__file__))

moby_file = os.path.join(root, 'mpos', 'mobyposi - Copy.i')

moby = set([line.rstrip() for line in open(moby_file, encoding = 'mac_roman')])

# print(len(moby))

print('Building dicts...')
word_indices = dict((line.split('*')[0], (i, line.split('*')[1])) for i,line in enumerate(moby))
indices_word = dict((i, line.split('*')[0]) for i,line, in enumerate(moby))

parts = set([pos for line in moby for pos in line.split('*')[1]])
pos_indices = dict((pos, i) for i,pos in enumerate(parts))
indices_pos = dict((i, pos) for i,pos in enumerate(parts))

texts = []
texts.append([s for s in getText(0) if isAdmissible(s, word_indices)])
texts.append([s for s in getText(1) if isAdmissible(s, word_indices)])
texts.append([s for s in getText(2) if isAdmissible(s, word_indices)])
texts.append([s for s in getText(3) if isAdmissible(s, word_indices)])
texts.append([s for s in getText(4) if isAdmissible(s, word_indices)])
texts.append([s for s in getText(5) if isAdmissible(s, word_indices)])
texts.append([s for s in getText(6) if isAdmissible(s, word_indices)])
# for text in texts:
	# print(len(text))

maxlen = 128

def TrainPosModel(model, dimIn, dimOut):
	print('Generating training set...')
	
	text = [w for s in texts[0] for w in s]+\
		[w for s in texts[1] for w in s]+\
		[w for s in texts[2] for w in s]
	
	sentences = []
	next_tokens = []
	step = 3
	for i in range(0, len(text) - dimIn, step):
		sentences.append(text[i: i + dimIn])
		next_tokens.append(text[i + dimIn])
	
	X = np.zeros((len(sentences), dimIn, dimOut), dtype=np.bool)
	y = np.zeros((len(sentences), dimOut), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, token in enumerate(sentence):
			for p in word_indices[getFuzzyMatch(token, word_indices)][1]:
				X[i, t, pos_indices[p]] = 1
		for p in word_indices[getFuzzyMatch(next_tokens[i], word_indices)][1]:
			y[i, pos_indices[p]] = 1
	
	print('Training model..')
	model.fit(X,y,nb_epoch=1,show_accuracy=True)
	return model

def TestPosModel(model, dimIn, dimOut):
	print('Testing model...')
	text = [w for s in texts[5] for w in s]+\
		[w for s in texts[6] for w in s]
	
	success = [0,0,0,0]
	failure = [0,0,0,0]
	for run in range(0,10):
		for iteration in range(0,100):
			start = random.randint(0, len(text) - maxlen - 2)
			x = np.zeros((1, dimIn, dimOut), dtype=np.bool)
			for t,token in enumerate(text[start:start+maxlen]):
				for p in word_indices[getFuzzyMatch(token, word_indices)][1]:
					x[0,t,pos_indices[p]]=1
			next = np.zeros((dimOut), dtype=np.bool)
			
			for p in word_indices[getFuzzyMatch(text[start+maxlen], word_indices)][1]:
				next[pos_indices[p]] = 1
			
			preds = model.predict(x)[0]
			if next[sample(preds)]:
				success[0] += 1
				success[1] += 1
				success[2] += 1
				success[3] += 1
			else:
				failure[0] += 1
				if next[sample(preds)]:
					success[1] += 1
					success[2] += 1
				else:
					failure[1] += 1
					if next[sample(preds)] or next[sample(preds)]:
						success[2] += 1
						success[3] += 1
					else:
						failure[2] += 1
						if next[sample(preds)] or next[sample(preds)] or next[sample(preds)] or next[sample(preds)]:
							success[3] += 1
						else:
							failure[3] += 1
		print('round: '+str(run+1))
	print(rate(success[0],failure[0]))
	print(rate(success[1],failure[1]))
	print(rate(success[2],failure[2]))
	print(rate(success[3],failure[3]))
	
def TrainWordModel(model, dimIn, dimOut):
	print('Generating training set...')
	
	text = [w for s in texts[0] for w in s]+\
		[w for s in texts[1] for w in s]+\
		[w for s in texts[2] for w in s]
	
	sentences = []
	next_tokens = []
	step = 3
	for i in range(0, len(text) - dimIn, step):
		sentences.append(text[i: i + dimIn])
		next_tokens.append(text[i + dimIn])
	
	# X = np.zeros((len(sentences), dimIn, dimOut), dtype=np.bool)
	# y = np.zeros((len(sentences), dimOut), dtype=np.bool)
	# for i, sentence in enumerate(sentences):
		# for t, token in enumerate(sentence):
			# for p in word_indices[getFuzzyMatch(token, word_indices)][1]:
				# X[i, t, dict[p]] = 1
		# for p in word_indices[getFuzzyMatch(next_tokens[i], word_indices)][1]:
			# y[i, dict[p]] = 1
	
	def myGenerator():
		X = np.zeros((dimIn, dimOut), dtype=np.bool)
		y = np.zeros((dimOut), dtype=np.bool)
		while 1:
			for i, sentence in enumerate(sentences):
				for t, token in enumerate(sentence):
					X[t, word_indices[getFuzzyMatch(token, word_indices)][0]] = 1
				y[word_indices[getFuzzyMatch(next_tokens[i], word_indices)][0]] = 1
				yield (X, y)
	
	def BatchGenerator(batch_size):
		X = np.zeros((batch_size, dimIn, dimOut), dtype=np.bool)
		y = np.zeros((batch_size, dimOut), dtype=np.bool)
		for i, sentence in enumerate(sentences):
			for t, token in enumerate(sentence):
					X[i%batch_size, t, word_indices[getFuzzyMatch(token, word_indices)][0]] = 1
			y[i%batch_size, word_indices[getFuzzyMatch(next_tokens[i], word_indices)][0]] = 1
			if i%batch_size == batch_size-1:
				yield X, y
		yield X, y
	
	print('Training model..')
	# model.fit(X,y,nb_epoch=1,show_accuracy=True)
	for X_train, Y_train in BatchGenerator(8192):
		model.fit(X_train,Y_train,batch_size=128,nb_epoch=1,show_accuracy=True)
	# model.fit_generator(myGenerator(), samples_per_epoch = len(text) - dimIn, nb_epoch = 1, show_accuracy = True)
	return model

def BuildModel(dimIn, dimOut):
	print('Building model...')
	model = Sequential()
	model.add(TimeDistributedDense(128, input_shape=(dimIn, dimOut)))
	model.add(LSTM(128, return_sequences=True))
	model.add(LSTM(128, return_sequences=False))
	model.add(Dropout(0.1))
	model.add(Dense(dimOut))
	model.add(Activation('softmax'))
	
	model.compile(loss='categorical_crossentropy',  optimizer='rmsprop')
	return model

def LoadModel(arch_file, weight_file):
	print('Loading model...')
	model = model_from_json(open(arch_file).read())
	model.load_weights(weight_file)
	return model
	
def SaveModel(model, arch_file, weight_file):
	print('Saving model...')
	open(arch_file, 'w').write(model.to_json())
	model.save_weights(weight_file)
	
def SaveTraining(model, weight_file):
	print('Saving training weights...')
	model.save_weights(weight_file)
	
def DoPosModel(id):
	print('----------------------------')
	print('Running part of speech model:')
	arch_file = os.path.join(root, 'arch_data.'+str(id)+'.json')
	weight_file = os.path.join(root, 'weight_data.'+str(id)+'.h5')
	train_file = os.path.join(root, 'train_data.'+str(id)+'.h5')

	if not os.path.isfile(arch_file) or not os.path.isfile(weight_file):
		model = BuildModel(maxlen, len(pos_indices))
		SaveModel(model, arch_file, weight_file)
	elif not os.path.isfile(train_file):
		model = LoadModel(arch_file, weight_file)
	else:
		model = LoadModel(arch_file, train_file)
		
	if not os.path.isfile(train_file):
		model = TrainPosModel(model, maxlen, len(pos_indices))
		SaveTraining(model, train_file)
		
	TestPosModel(model, maxlen, len(pos_indices))
		
def DoWordModel(id):
	print('----------------------------')
	print('Running word model:')
	arch_file = os.path.join(root, 'arch_data.'+str(id)+'.json')
	weight_file = os.path.join(root, 'weight_data.'+str(id)+'.h5')
	train_file = os.path.join(root, 'train_data.'+str(id)+'.h5')

	if not os.path.isfile(arch_file) or not os.path.isfile(weight_file):
		model = BuildModel(maxlen, len(word_indices))
		SaveModel(model, arch_file, weight_file)
	elif not os.path.isfile(train_file):
		model = LoadModel(arch_file, weight_file)
	else:
		model = LoadModel(arch_file, train_file)
		
	if not os.path.isfile(train_file):
		model = TrainWordModel(model, word_indices, maxlen, len(word_indices))
		SaveTraining(model, train_file)
	
def main():
	DoPosModel('pos.dll.128')
	# DoWordModel('word')
	
if __name__ == '__main__':
	main()