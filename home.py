import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
sr= 22050
Audio_path=r'c:\Users\rajan\Downloads\Mahaan-128kbps-MassTamilan.fm\Naan-Naan-MassTamilan.fm.mp3'
if Audio_path:
    y, sr = lib.load(Audio_path, sr=sr)
    plt.figure(figsize=(10, 10))
    plt.plot(y)
    plt.title('Naan Naan from Mahaan')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()
import librosa as lib
import matplotlib.pyplot as plt
import numpy as np
Sr= 22050
window_size=2048
hop_lenght=512
Audio_path=r'c:\Users\rajan\Downloads\Mahaan-128kbps-MassTamilan.fm\Naan-Naan-MassTamilan.fm.mp3'
y,sr=lib.load(Audio_path,sr=Sr,mono=True)
print('1D samples',y.shape,round(len(y)/sr,1),'seconds')
S=lib.stft(y,n_fft=window_size,hop_length=hop_lenght)
s_decibel=lib.amplitude_to_db(np.abs(S),ref=np.max)
print('2d spectogram',s_decibel.shape,'= (frequency bins,frames)')
print("One frame =", round(hop_lenght / sr * 1000, 1), "ms")
print("One frequency bin =", round(sr / window_size, 2), "Hz")
plt.figure(figsize=(16,7))
img=lib.display.specshow(s_decibel,sr=sr,hop_length=hop_lenght,x_axis='time',y_axis='hz')
from scipy.ndimage import maximum_filter
Box=30
min_db=-30
is_max=maximum_filter(s_decibel,size=Box)==s_decibel
is_louder=s_decibel>min_db
freq,time=np.where(is_max & is_louder)
print('Number of peaks:',len(freq))
print('Peaks per second',round(len(freq)/len(y)/sr,1))
peaktime=time*hop_lenght/sr
peakfreq=freq*sr/window_size
plt.colorbar(img, format='%+2.0f dB')

plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.scatter(peaktime, peakfreq, s=4, color='blue')
plt.show()

