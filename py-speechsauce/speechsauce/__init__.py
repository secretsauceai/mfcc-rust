from functools import lru_cache


@lru_cache(maxsize=32)
def _get_speech_config(
    sampling_frequency,
    frame_length=0.020,
    frame_stride=0.01,
    num_cepstral=13,
    num_filters=40,
    fft_length=512,
    low_frequency=0,
    high_frequency=None,
    dc_elimination=True,
):
    """pay no attention to the man behind the curtain

    this function returns a config object to be used by the rust code, avoids recomputing elements where possible
    """
    pass


def mfcc(
    signal,
    sampling_frequency,
    frame_length=0.020,
    frame_stride=0.01,
    num_cepstral=13,
    num_filters=40,
    fft_length=512,
    low_frequency=0,
    high_frequency=None,
    dc_elimination=True,
):
    """Compute MFCC features from an audio signal.
    Args:
         signal (array): the audio signal from which to compute features.
             Should be an N x 1 array
         sampling_frequency (int): the sampling frequency of the signal
             we are working with.
         frame_length (float): the length of each frame in seconds.
             Default is 0.020s
         frame_stride (float): the step between successive frames in seconds.
             Default is 0.02s (means no overlap)
         num_filters (int): the number of filters in the filterbank,
             default 40.
         fft_length (int): number of FFT points. Default is 512.
         low_frequency (float): lowest band edge of mel filters.
             In Hz, default is 0.
         high_frequency (float): highest band edge of mel filters.
             In Hz, default is samplerate/2
         num_cepstral (int): Number of cepstral coefficients.
         dc_elimination (bool): hIf the first dc component should
             be eliminated or not.
    Returns:
        array: A numpy array of size (num_frames x num_cepstral) containing mfcc features.
    """
    pass
