import numpy as np
import scipy.io as sio

file = open('../data/Flickr30kEntities/phraseList.txt','r')
sequenceData = sio.loadmat('GTPhrases_Flickr30kEntities.mat')['sequenceData'][0]
filenames = sio.loadmat('GTPhrases_Flickr30kEntities.mat')['filename'][0].tolist()
trainfns = sio.loadmat('../data/Flickr30kEntities/dataSplits.mat')['trainfns']
validationfns = sio.loadmat('../data/Flickr30kEntities/dataSplits.mat')['validationfns']
testfns = sio.loadmat('../data/Flickr30kEntities/dataSplits.mat')['testfns']

# build_vocab
max_length = 5
counts = {}
final_captions = []
phrase_names = []
img_list = []
prev_imageID = ''
# for line in file:
# 	imageID = line[:-1].split(' ')[0]
# 	if imageID not in testfns and imageID not in validationfns:
# 		sentenceNumber = int(line[:-1].split(' ')[1])
# 		phraseNumber = int(line[:-1].split(' ')[2])
# 		words = line[:-1].split(' ')[4:]
# 		phrase_names.append(imageID+'_'+str(sentenceNumber)+'_'+str(phraseNumber))
# 		if prev_imageID != imageID:
# 			print(len(final_captions))
# 			img_list.append(imageID)
# 			final_captions.append([])
# 			prev_imageID = imageID
# 		if words[0] == 'NO_VALID_POS':
# 			imageIdx = filenames.index(imageID+'.txt')
# 			invalid_words = sequenceData[imageIdx]['phrases'][sentenceNumber-1,0]
# 			invalid_words = invalid_words[0,phraseNumber-1]
# 			invalid_words = [iw[0][0].encode('utf-8').lower() for iw in invalid_words]
# 			invalid_words = invalid_words
# 			final_captions[-1].append(invalid_words)
# 			for k,iw in enumerate(invalid_words):
# 				if k < max_length:
# 					counts[iw] = counts.get(iw, 0) + 1
# 		else:
# 			words = [w.lower() for w in words]
# 			final_captions[-1].append(words)
# 			for k,w in enumerate(words):
# 				if k < max_length:
# 					counts[w] = counts.get(w, 0) + 1
for imageID in trainfns:
	imageID = imageID[0][0].encode('utf-8')
	img_list.append(imageID)
	final_captions.append([])
	imageIdx = filenames.index(imageID+'.txt')
	sentenceNumber = 0
	for sentenceNumber, sentence in enumerate(sequenceData[imageIdx]['phrases']):
		if len(sentence[0]) == 0:
			continue
		for phraseNumber, phrase in enumerate(sentence[0][0]):
			phrase = [w[0][0].encode('utf-8').lower() for w in phrase]
			phrase_names.append(imageID+'_'+str(sentenceNumber+1)+'_'+str(phraseNumber+1))
			final_captions[-1].append(phrase)
			for key,word in enumerate(phrase):
				if key < max_length:
					counts[word] = counts.get(word, 0) + 1

for imageID in validationfns:
	imageID = imageID[0][0].encode('utf-8')
	img_list.append(imageID)
	final_captions.append([])
	imageIdx = filenames.index(imageID+'.txt')
	sentenceNumber = 0
	for sentenceNumber, sentence in enumerate(sequenceData[imageIdx]['phrases']):
		for phraseNumber, phrase in enumerate(sentence[0][0]):
			phrase = [w[0][0].encode('utf-8').lower() for w in phrase]
			phrase_names.append(imageID+'_'+str(sentenceNumber+1)+'_'+str(phraseNumber+1))
			final_captions[-1].append(phrase)
			for key,word in enumerate(phrase):
				if key < max_length:
					counts[word] = counts.get(word, 0) + 1

for imageID in testfns:
	imageID = imageID[0][0].encode('utf-8')
	img_list.append(imageID)
	final_captions.append([])
	imageIdx = filenames.index(imageID+'.txt')
	for sentenceNumber,sentence in enumerate(sequenceData[imageIdx]['phrases']):
		for phraseNumber, phrase in enumerate(sentence[0][0]):
			phrase = [w[0][0].encode('utf-8').lower() for w in phrase]
			phrase_names.append(imageID+'_'+str(sentenceNumber+1)+'_'+str(phraseNumber+1))
			final_captions[-1].append(phrase)
			for key,word in enumerate(phrase):
				if key < max_length:
					counts[word] = counts.get(word, 0) + 1

# print some stats
cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
print 'top words and their counts:'
print '\n'.join(map(str,cw[:20]))  
total_words = sum(counts.itervalues())
print 'total words:', total_words

vocab = [w for w,n in counts.iteritems()]
print 'number of words in vocab would be %d' % (len(vocab), )
file.close()

# lets look at the distribution of lengths as well
phrase_lengths = {}
for img in final_captions:
	for phrase in img:
	    nw = len(phrase)
	    phrase_lengths[nw] = phrase_lengths.get(nw, 0) + 1
max_len = max(phrase_lengths.keys())
print 'max length sentence in raw data: ', max_len
print 'sentence length distribution (count, number of words):'
sum_len = sum(phrase_lengths.values())
for i in xrange(max_len+1):
	print '%2d: %10d   %f%%' % (i, phrase_lengths.get(i,0), phrase_lengths.get(i,0)*100.0/sum_len)

# import pickle
# pickle.dump(vocab,open('vocab.pkl', 'wb'))

itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

N = len(filenames)
M = len(phrase_names) # total number of captions
label_arrays = []
label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
label_end_ix = np.zeros(N, dtype='uint32')
label_length = np.zeros(M, dtype='uint32')
phrase_counter = 0
counter = 1
for i,phrases in enumerate(final_captions):
    n = len(phrases)
    assert n > 0, 'error: some image has no captions'
    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(phrases):
    	label_length[phrase_counter] = min(max_length, len(s)) # record the length of this sequence
    	phrase_counter = phrase_counter + 1
    	for k,w in enumerate(s):
    		if k < max_length:
    			Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    counter += n

L = np.concatenate(label_arrays, axis=0) # put all the labels together
assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
assert np.all(label_length > 0), 'error: some caption had no words?'

print 'encoded captions to array of size ', L.shape

import h5py

f = h5py.File('new_data.h5', "w")
f.create_dataset("labels", dtype='uint32', data=L)
f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
f.create_dataset("label_length", dtype='uint32', data=label_length)
dset = f.create_dataset("VGG-DET", (N,100,4096), dtype='float32') # space for resized feature
feat_file = h5py.File('extract-VGG-DET/VGG-DET.hdf5','r')
for i,name in enumerate(img_list):
	# load the feature
	feat = feat_file[name+'.jpg']#img['filename']
	# write to h5
	dset[i] = feat
	if i % 1000 == 0:
		print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
feat_file.close()
f.close()
print 'wrote ', 'new_data.h5'

import json

out = {}
out['ix_to_word'] = itow # encode the (1-indexed) vocab
out['images'] = []
out['phrase_names'] = phrase_names
for i,name in enumerate(img_list):
	jimg = {}
	if name in trainfns:
	  jimg['split'] = 'train'
	elif name in validationfns:
	  jimg['split'] = 'val'
	elif name in testfns:
	  jimg['split'] = 'test'
	else:
	  print('error: no file find in fns')
	  exit(1)
	jimg['filename'] = name
	out['images'].append(jimg)

json.dump(out, open('new_data.json', 'w'))
print 'wrote ', 'data.json'

