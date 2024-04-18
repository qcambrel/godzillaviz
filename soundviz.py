import numpy as np
import matplotlib as mpl
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file (replace 'your_audio_file.wav' with the path to your audio file)
audio_file = 'godzilla.m4a'
y, sr = librosa.load(audio_file)

# Calculate tempo and beat frames
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Convert beat frames to time
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Create a spectrogram
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

def generate_color_map(levels):
    # blue = np.array([30, 144, 245])  # Blue-white shade
    # green = np.array([0, 255, 0])  # Green color
    
    # # Generate intermediate colors using linear interpolation
    # colors = [blue + (green - blue) * (i / (levels - 1)) for i in range(levels)]

    dark_green = np.array([0, 100, 0])  # Dark green color
    black = np.array([0, 0, 0])  # Black color
    fiery_blue = np.array([0, 0, 255])  # Fiery blue color
    
    # Generate intermediate colors using linear interpolation
    colors = [dark_green + (black - dark_green) * (i / (levels - 1)) for i in range(levels)]
    
    # Replace black with fiery blue
    colors[-1] = fiery_blue
    
    return np.array(colors) / 255

# Generate color map with 64 levels
cmap = mpl.colors.ListedColormap(generate_color_map(64))

# print(cmap)

# Plot spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='RdYlGn')
plt.colorbar(format='%+2.0f dB')
plt.title('Godzilla Theme Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')

# Plot beat markers
plt.vlines(beat_times, 0, sr, color='black', alpha=0.75, linestyle='--', label='Beats')

plt.savefig('godzilla_analysis.png')
