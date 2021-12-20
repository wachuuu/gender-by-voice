import soundfile
import numpy as np
import sys
from scipy.fftpack import fft, fftfreq
from scipy.signal import decimate


def read_file(filename):
  wav, rate = soundfile.read(filename, always_2d=True)
  # convert voice file to mono
  if wav.shape[1] == 2: wav = (wav[:,0] + wav[:,1]) / 2
  else: wav = wav[:,0]
  return wav, rate

def prepare_data(data, rate):
  # prepare array of frequenceis with len of data
  frequencies = fftfreq(len(data), d=1 / rate)
  # apply window function to given data
  data = data * np.kaiser(len(data), 50)
  # calc fft of a signal
  fft_data = abs(fft(data))
  return frequencies, fft_data

def hps(signal, n):
  output = signal.copy()
  for k in range(2, n+1):
    # downsample original signal by k
    downsampled = decimate(signal, k)
    # multiply original signal with downsampled one
    output[:len(downsampled)] *= downsampled
  output[:10] = 0
  return output

def detect_voice(frequencies, data):
  # set frequency bounds for male and female
  male_lowest = 85
  male_highest = 175
  female_lowest = 175
  female_highest = 355

  # get data with frequencies from 85Hz to 355Hz
  voice_freqs = []
  voice_data = []
  for i in range(len(frequencies)):
    if (male_lowest <= frequencies[i]) & (frequencies[i] <= female_highest):
      voice_freqs.append(frequencies[i])
      voice_data.append(data[i])

  # get the fundamental frequency from hps data
  fundamental_freq = voice_freqs[np.argmax(voice_data)]
  # check the fundamental frequency
  if male_lowest <= fundamental_freq < male_highest: return 'M'
  if female_lowest <= fundamental_freq < female_highest: return 'K'
  return 'M'

def main():
  filename = sys.argv[1]
  data, rate = read_file(filename)
  frequencies, fft_data = prepare_data(data, rate)
  hps_data = hps(fft_data, 4)
  gender = detect_voice(frequencies, hps_data)
  print(gender)

if __name__ == '__main__':
  main()