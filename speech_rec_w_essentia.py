#from utils import progress_bar_downloader
import os

import essentia
import essentia.standard as ess

import matplotlib.pyplot as plt

#Hosting files on my dropbox since downloading from google code is painful
#Original project hosting is here: https://code.google.com/p/hmm-speech-recognition/downloads/list
#Audio is included in the zip file
link = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
dlname = 'audio.tar.gz'

if not os.path.exists('./%s'%dlname):
    progress_bar_downloader(link, dlname)
    os.system('tar xzf %s'%dlname)
else:
    print '%s already downloaded!'%dlname

fpaths = []
labels = []
spoken = []
for f in os.listdir('audio'):
    for w in os.listdir('audio/' + f):
        fpaths.append('audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print 'Words spoken:',spoken

# Files can be heard in Linux using the following commands from the command line
# cat kiwi07.wav | aplay -f S16_LE -t wav -r 8000
# Files are signed 16 bit raw, sample rate 8000
from scipy.io import wavfile
import numpy as np

data = np.zeros((len(fpaths), 32000))
maxsize = -1
for n,file in enumerate(fpaths):
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]

#Each sample file is one row in data, and has one entry in labels
print 'Number of files total:',data.shape[0]
all_labels = np.zeros(data.shape[0])
for n, l in enumerate(set(labels)):
    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
    
print 'Labels and label indices',all_labels

import scipy

def stft(x, fftsize=64, overlap_pct=.5):   
    #Modified from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]    
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize / 2)]


# This processing (top freq peaks) only works for single speaker case... need better features for multispeaker!
# MFCC (or deep NN/automatic feature extraction) could be interesting

M = 1024
N = 1024
H = 256
fs = 8000
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
mfcc = ess.MFCC(numberCoefficients = 7, inputSize = N/2+1)

all_obs = []

"""for i in range(data.shape[0]):
    
    d = np.abs(stft(data[i, :]))
    n_dim = 6
    obs = np.zeros((n_dim, d.shape[0]))
    for r in range(d.shape[0]):
        _, t = peakfind(d[r, :], n_peaks=n_dim)
        obs[:, r] = t.copy()
    if i % 10 == 0:
        print "Processed obs %s"%i
    all_obs.append(obs) """

#for i in range(data.shape[0]):
for n,file in enumerate(fpaths):
    #y = np.zeros((32000))
    x = ess.MonoLoader(filename = file, sampleRate = fs)()

    #y[:x.shape[0]] = x

    #x = np.resize(x, 32000)
    x = np.pad(x, (0,32000-len(x)), mode='constant')
    
    mfccs = []
    
    maxsize = 32000
    #x = x[:maxsize]
    
    for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):
        mX = spectrum(window(frame))
        mfcc_bands, mfcc_coeffs = mfcc(mX)
        mfccs.append(mfcc_coeffs)

    mfccs = np.array(mfccs)

    all_obs.append(np.transpose(mfccs))
    
all_obs = np.atleast_3d(all_obs)


import scipy.stats as st

class gmmhmm:
    #This class converted with modifications from https://code.google.com/p/hmm-speech-recognition/source/browse/Word.m
    def __init__(self, n_states):
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)
        
        #Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))
        
        self.mu = None
        self.covs = None
        self.n_dims = None
           
    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
         
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha
    
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape);
           
        beta[:, -1] = np.ones(B.shape[0])
            
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            #Needs scipy 0.14
            B[s, :] = st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
            #This function can (and will!) return values >> 1
            #See the discussion here for the equivalent matlab function
            #https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            #Key line: "Probabilities have to be less than 1,
            #Densities can be anything, even infinite (at individual points)."
            #This is evaluating the density at individual points...
        return B
    
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
    
    def _em_init(self, obs):
        #Using this _em_init function allows for less required constructor args
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]
        return self
    
    def _em_step(self, obs): 
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]
        
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)
        
        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T))
        
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)
              
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)
        
        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)
        
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        #Ensure positive semidefinite by adding diagonal loading
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
        
        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood
    
    def fit(self, obs, n_iter=15):
        #Support for 2D and 3D arrays
        #2D should be n_features, n_dims
        #3D should be n_examples, n_features, n_dims
        #For example, with 6 features per speech segment, 105 different words
        #this array should be size
        #(105, 6, X) where X is the number of frames with features extracted
        #For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self
    
    def transform(self, obs):
        #Support for 2D and 3D arrays
        #2D should be n_features, n_dims
        #3D should be n_examples, n_features, n_dims
        #For example, with 6 features per speech segment, 105 different words
        #this array should be size
        #(105, 6, X) where X is the number of frames with features extracted
        #For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
                out[n] = log_likelihood
            return out

if __name__ == "__main__":
    """rstate = np.random.RandomState(0)
    t1 = np.ones((4, 40)) + .001 * rstate.rand(4, 40)
    t1 /= t1.sum(axis=0)
    t2 = rstate.rand(*t1.shape)
    t2 /= t2.sum(axis=0)
    
    m1 = gmmhmm(2)
    m1.fit(t1)
    m2 = gmmhmm(2)
    m2.fit(t2)
    
    m1t1 = m1.transform(t1)
    m2t1 = m2.transform(t1)
    print "Likelihoods for test set 1"
    print "M1:",m1t1
    print "M2:",m2t1
    print "Prediction for test set 1"
    print "Model", np.argmax([m1t1, m2t1]) + 1
    print 
    
    m1t2 = m1.transform(t2)
    m2t2 = m2.transform(t2)
    print "Likelihoods for test set 2"
    print "M1:",m1t2
    print "M2:",m2t2
    print "Prediction for test set 2"
    print "Model", np.argmax([m1t2, m2t2]) + 1"""

    from sklearn.cross_validation import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)

    for n,i in enumerate(all_obs):
    	all_obs[n] /= all_obs[n].sum(axis=0)

    for train_index, test_index in sss:
    	X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
    	y_train, y_test = all_labels[train_index], all_labels[test_index]
    	print 'Size of training matrix:', X_train.shape
    	print 'Size of testing matrix:', X_test.shape

    ys = set(all_labels)
    ms = [gmmhmm(6) for y in ys]
    _ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
    ps = [m.transform(X_test) for m in ms]
    res = np.vstack(ps)
    predicted_labels = np.argmax(res, axis=0)
    missed = (predicted_labels != y_test)
    print 'Test accuracy:%.2f percent'%(100 * (1 - np.mean(missed)))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predicted_labels)
    plt.matshow(cm, cmap='gray')
    ax = plt.gca()
    _ = ax.set_xticklabels([" "] + [l[:2] for l in spoken])
    _ = ax.set_yticklabels([" "] + spoken)
    plt.title('Confusion matrix, single speaker')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	



