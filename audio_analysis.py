import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.spatial.distance import euclidean
        
st.title("Vocal Performance Comparison Tool")

# Upload original and user audio files
st.write("Upload the original and user audio files for comparison:")
original_file = st.file_uploader("Upload the Original Audio", type=["wav", "mp3"])
user_file = st.file_uploader("Upload the User's Audio", type=["wav", "mp3"])

if original_file and user_file:
    # Load original and user audio using librosa
    y_original, sr_original = librosa.load(original_file, sr=None)
    y_user, sr_user = librosa.load(user_file, sr=None)
    
    # Display the waveforms
    st.write("Waveform of the Original Audio:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y_original, sr=sr_original, ax=ax)
    st.pyplot(fig)
    
    st.write("Waveform of the User's Audio:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y_user, sr=sr_user, ax=ax)
    st.pyplot(fig)

    # Compute MFCCs (Mel-frequency cepstral coefficients) for pitch comparison
    mfcc_original = librosa.feature.mfcc(y=y_original, sr=sr_original)
    mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_user)

    # Ensure correct shape for DTW
    original_mfcc = mfcc_original.reshape(1, -1) if mfcc_original.ndim == 1 else mfcc_original
    user_mfcc = mfcc_user.reshape(1, -1) if mfcc_user.ndim == 1 else mfcc_user

    # MFCCs are usually (n_mfcc, n_frames), transpose them for DTW
    original_mfcc = mfcc_original.T
    user_mfcc = mfcc_user.T

    # Compute DTW (Dynamic Time Warping) for rhythm comparison
    dtw_result = dtw(original_mfcc, user_mfcc, dist=lambda x, y: np.linalg.norm(x - y))
    
    # dtw_result is a tuple
    distance = dtw_result[0]  # First element is the distance
    path = dtw_result[2]  # Third element is the path (index)

    # Display the MFCC comparison
    st.write(f"DTW distance (rhythm comparison): {distance:.2f}")
    
    # Show the DTW path on the MFCC spectrogram of the original
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc_original, sr=sr_original, x_axis='time')
    ax.plot(np.array(path).T, color='r')
    st.pyplot(fig)

    # Compute pitch difference
    pitches_original, magnitudes_original = librosa.piptrack(y=y_original, sr=sr_original)
    pitches_user, magnitudes_user = librosa.piptrack(y=y_user, sr=sr_user)

    pitch_diff = np.abs(np.mean(pitches_original) - np.mean(pitches_user))
    st.write(f"Pitch difference: {pitch_diff:.2f} Hz")
    
    # Visualize the pitch difference
    fig, ax = plt.subplots()
    ax.plot(pitches_original.mean(axis=1), label="Original Pitch")
    ax.plot(pitches_user.mean(axis=1), label="User Pitch")
    ax.legend(loc='upper right')
    st.pyplot(fig)

    ## Volume (Amplitude) Analysis
    # Compute RMS (Root Mean Square) for volume analysis
    def compute_rms(y):
        return np.sqrt(np.mean(y**2))
        
    rms_original = compute_rms(y_original)
    rms_user = compute_rms(y_user)

    st.write(f"RMS Volume of Original Audio: {rms_original:.2f}")
    st.write(f"RMS Volume of User's Audio: {rms_user:.2f}")

    # Visualize RMS
    fig, ax = plt.subplots()
    ax.bar(['Original', 'User'], [rms_original, rms_user])
    ax.set_ylabel('RMS Amplitude')
    st.pyplot(fig)

    ## Timber
    # Compute spectral features for timbre analysis
    spectral_centroid_original = librosa.feature.spectral_centroid(y=y_original, sr=sr_original)
    spectral_centroid_user = librosa.feature.spectral_centroid(y=y_user, sr=sr_user)

    # Visualize spectral centroid
    fig, ax = plt.subplots()
    ax.plot(spectral_centroid_original.T, label='Original Centroid')
    ax.plot(spectral_centroid_user.T, label='User Centroid')
    ax.set_ylabel('Spectral Centroid (Hz)')
    ax.legend()
    st.pyplot(fig)

    # Compute spectral bandwidth for both audio files
    bandwidth_original = librosa.feature.spectral_bandwidth(y=y_original, sr=sr_original)
    bandwidth_user = librosa.feature.spectral_bandwidth(y=y_user, sr=sr_user)

    # Visualize spectral bandwidth
    fig, ax = plt.subplots()
    ax.plot(bandwidth_original.T, label='Original Bandwidth')
    ax.plot(bandwidth_user.T, label='User Bandwidth')
    ax.set_ylabel('Spectral Bandwidth (Hz)')
    ax.legend()
    st.pyplot(fig)

    st.success("Analysis Complete!")

else:
    st.warning("Please upload both original and user audio files to proceed.")
