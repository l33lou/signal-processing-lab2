# -*- coding: utf-8 -*-
"""

# Pour les filtres en python, voir (lien donné dans le cours) :
#
# https://www.f-legrand.fr/scidoc/docmml/numerique/filtre/filtrenum/filtrenum.html
#
# Note : comme indiqué pour les filtres de matlab, n'utilisez que les filtres FIR !!!
"""

import numpy
import scipy.signal
import matplotlib.pyplot as plt

### filtre passe-bas

def low_pass_filter():

	P=20 #filter order
	b1 = scipy.signal.firwin(numtaps=2*P+1,cutoff=[0.25],window='hann',fs=1)

	plt.figure(1,figsize=(10, 8))
	plt.plot(b1)
	plt.draw()
	#plt.pause(1)

	w,h=scipy.signal.freqz(b1)

	plt.figure(2,figsize=(10, 10))
	plt.subplot(211)
	plt.plot(w/(2*numpy.pi),20*numpy.log10(numpy.absolute(h)))
	plt.xlabel('filtre passe-bas -- reponse en frequence - amplitude')
	plt.subplot(212)      
	plt.plot(w/(2*numpy.pi),numpy.unwrap(numpy.angle(h)))
	plt.xlabel('filtre passe-bas -- reponse en frequence - phase')
	plt.draw()
	plt.pause(1)

	return b1

### filtre passe-haut

def high_pass_filter():

	P=20
	b2 = scipy.signal.firwin(numtaps=2*P+1,cutoff=[0.25],window='hann',fs=1,pass_zero=False)

	plt.figure(3,figsize=(10, 8))
	plt.plot(b2)
	plt.draw()
	plt.pause(1)

	w2,h2=scipy.signal.freqz(b2)

	plt.figure(4,figsize=(10, 10))
	plt.subplot(211)
	plt.plot(w2/(2*numpy.pi),20*numpy.log10(numpy.absolute(h2)))
	plt.xlabel('filtre passe-haut -- reponse en frequence - amplitude')
	plt.subplot(212)      
	plt.plot(w2/(2*numpy.pi),numpy.unwrap(numpy.angle(h2)))
	plt.xlabel('filtre passe-haut -- reponse en frequence - phase')
	plt.draw()
	#plt.pause(2)

	return b2

### apply filter to signal xx

def apply_filter(tt, xx, filter_type, visualize = False):

	if filter_type == "low" :
		b = low_pass_filter()

	elif filter_type == "high" :
		b = high_pass_filter()

	else :
		raise ValueError("filter_type should either be 'low' or 'high'")

	#Filtration
	yy=scipy.signal.filtfilt(b, 1, xx)
	#Sub-sampling
	tt_dec = tt[::2]
	yy_dec = yy[::2]


	if visualize :

		plt.figure(5,figsize=(10, 10))
		plt.subplot(211)
		plt.plot(tt,xx)
		plt.subplot(212)
		plt.plot(tt,yy)
		plt.xlabel(f'filtre passe-{filter_type}')
		plt.pause(1)

	return tt_dec, yy_dec

