import soundfile
import numpy as np
import sys
from scipy.fftpack import fft, fftfreq
from scipy.signal import decimate
from os import listdir
from os.path import isfile, join

# set frequency bounds for male and female
MALE_LOWEST = 85
MF_LIMIT = 175
FEMALE_HIGHEST = 355


def read_file(filename):
    wav, rate = soundfile.read(filename, always_2d=True)
    # convert voice file to mono
    if wav.shape[1] != 2:
        wav = wav[:, 0]
    else:
        wav = (wav[:, 0] + wav[:, 1]) / 2
    return wav, rate


def prepare_data(data, rate):
    # prepare array of frequencies with len of data
    frequencies = fftfreq(len(data), d=1 / rate)
    # apply window function to given data
    data = data * np.kaiser(len(data), 50)
    # calc fft of a signal
    fft_data = abs(fft(data))
    return frequencies, fft_data


def hps(signal, n):
    output = signal.copy()
    for k in range(2, n + 1):
        # down_sample original signal by k
        down_sampled = decimate(signal, k)
        # multiply original signal with down_sampled one
        output[:len(down_sampled)] *= down_sampled
    return output


def detect_voice(frequencies, data):
    # get data with frequencies from 85Hz to 355Hz
    voice_freqs = []
    voice_data = []
    for i in range(len(frequencies)):
        if (MALE_LOWEST <= frequencies[i]) & (frequencies[i] <= FEMALE_HIGHEST):
            voice_freqs.append(frequencies[i])
            voice_data.append(data[i])

    # get the fundamental frequency from hps data
    fundamental_freq = voice_freqs[np.argmax(voice_data)]
    # check the fundamental frequency
    if MALE_LOWEST <= fundamental_freq < MF_LIMIT:
        return 'M'
    if MF_LIMIT <= fundamental_freq < FEMALE_HIGHEST:
        return 'K'
    return 'M'


def recognize(filename):
    data, rate = read_file(filename=filename)
    frequencies, fft_data = prepare_data(data, rate)
    hps_data = hps(fft_data, 4)
    gender = detect_voice(frequencies, hps_data)
    return gender


def check_filename(file):
    if "K" in file: return 'K'
    if "M" in file: return 'M'


def main():
    # expected - actual
    m_m = 0
    k_k = 0
    m_k = 0
    k_m = 0
    files = [f for f in listdir("./train/") if isfile(join("./train/", f))]
    for i, file in enumerate(files):
        expected = check_filename(file)
        actual = recognize("./train/" + file)
        print(f"{i} - {expected} {actual}\n---------")

        if expected == 'M' and actual == 'M': m_m += 1
        if expected == 'M' and actual == 'K': m_k += 1
        if expected == 'K' and actual == 'M': k_m += 1
        if expected == 'K' and actual == 'K': k_k += 1

    print("---------------------------------")
    print("exp. m act m", m_m)
    print("exp. m act k", m_k)
    print("exp. k act m", k_m)
    print("exp. k act k", k_k)


if __name__ == '__main__':
    main()
