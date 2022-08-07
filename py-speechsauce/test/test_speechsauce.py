import numpy as np
from speechsauce import mfcc, preemphasis, cmvn, stack_frames


# Ramdom signal generation for testing
mu, sigma = 0, 0.1 # mean and standard deviation
signal = np.random.normal(mu, sigma, 1000000)
fs = 16000

 # Generating stached frames with SpeechPy
frame_length = 0.02
frame_stride = 0.02
num_filters=40

def test_mfcc():
       
    num_cepstral = 13
    mfcc = mfcc(signal, sampling_frequency=fs,
                            frame_length=0.020, num_cepstral=num_cepstral, frame_stride=0.01,
                            num_filters=num_filters, fft_length=512, low_frequency=0,
                            high_frequency=None)

    # Shape matcher
    assert mfcc.shape[1] == num_cepstral

def test_cmvn(self):
    
    feature_vector = np.random.rand(50,100)
    normalized_feature = processing.cmvn(feature_vector, variance_normalization=True)
    
    print(f"normalized_feature shape {normalized_feature.shape}")
    # Shape match
    assert normalized_feature.shape == feature_vector.shape
    
    # Check the std and mean of the output vector
    assert np.allclose(np.mean(normalized_feature,axis=0), np.zeros((1,normalized_feature.shape[1])))
    assert np.allclose(np.std(normalized_feature,axis=0), np.ones((1,normalized_feature.shape[1])))

def test_preemphasis(self):
    
    # Performing the operation on the generated signal.
    signal_preemphasized = preemphasis(signal, cof=0.98)
    
    # Shape matcher
    assert signal_preemphasized.ndim == 1
    assert signal_preemphasized.shape == signal.shape