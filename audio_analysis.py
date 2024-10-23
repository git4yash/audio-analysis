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


    # Create overlay plot
    st.write("Performance Comparison Overlay:")

    fig, ax = plt.subplots()

    # Plot waveforms
    librosa.display.waveshow(y_original, sr=sr_original, ax=ax, label='Original', alpha=0.5)
    librosa.display.waveshow(y_user, sr=sr_user, ax=ax, label='User', alpha=0.5)

    # Highlight areas where improvements are needed
    diff_threshold = 0.1  # Set threshold for significant differences

    # Overlay pitch differences
    pitch_diffs = np.abs(pitches_original - pitches_user)
    time_axis = np.linspace(0, len(y_original)/sr_original, num=len(pitch_diffs))
    ax.plot(time_axis, pitch_diffs.mean(axis=1), color='red', label="Pitch Difference", alpha=0.7)

    # Mark areas where the pitch difference exceeds the threshold
    for i, diff in enumerate(pitch_diffs.mean(axis=1)):
        if diff > diff_threshold:
            ax.axvspan(time_axis[i], time_axis[i+1] if i+1 < len(time_axis) else time_axis[i],
                       color='yellow', alpha=0.3, label="Pitch Correction Needed" if i == 0 else "")

    # Overlay volume differences
    time_original = np.linspace(0, len(y_original)/sr_original, num=len(y_original))
    ax.plot(time_original, np.abs(y_original - y_user), color='blue', label="Volume Difference", alpha=0.7)

    ax.set_ylabel('Amplitude / Pitch Difference')
    ax.set_xlabel('Time (s)')
    ax.legend()

    # Display plot
    st.pyplot(fig)

    st.write(f"DTW distance (rhythm comparison): {distance:.2f}")
    st.write(f"Pitch difference: {pitch_diff:.2f} Hz")
    st.write(f"RMS Volume of Original: {rms_original:.2f}, User: {rms_user:.2f}")
    st.write(f"Spectral Centroid Difference: {np.mean(spectral_centroid_original - spectral_centroid_user):.2f} Hz")
    st.write(f"Spectral Bandwidth Difference: {np.mean(bandwidth_original - bandwidth_user):.2f} Hz")

    ## New graph
    # Compute DTW for rhythm comparison
    dtw_result = dtw(mfcc_original.T, mfcc_user.T, dist=lambda x, y: np.linalg.norm(x - y))
    distance = dtw_result[0]
    path = dtw_result[2]

    # Compute pitch contours
    pitches_original, magnitudes_original = librosa.piptrack(y=y_original, sr=sr_original)
    pitches_user, magnitudes_user = librosa.piptrack(y=y_user, sr=sr_user)

    # Extract the dominant pitch for each time frame
    pitch_contour_original = [np.max(pitches_original[:, t]) for t in range(pitches_original.shape[1])]
    pitch_contour_user = [np.max(pitches_user[:, t]) for t in range(pitches_user.shape[1])]

    # Create pitch difference array and time axis
    pitch_diff = np.abs(np.array(pitch_contour_original) - np.array(pitch_contour_user))
    time_axis = np.linspace(0, len(pitch_contour_original)/sr_original, num=len(pitch_contour_original))

    # Volume (Amplitude) Analysis
    def compute_rms(y):
        return np.sqrt(np.mean(y**2))

    rms_original = compute_rms(y_original)
    rms_user = compute_rms(y_user)

    # Create overlay plot
    st.write("Performance Comparison Overlay (Including Tune Differences):")

    fig, ax = plt.subplots()

    # Plot waveforms
    librosa.display.waveshow(y_original, sr=sr_original, ax=ax, label='Original', alpha=0.5)
    librosa.display.waveshow(y_user, sr=sr_user, ax=ax, label='User', alpha=0.5)

    # Plot pitch contours (for tune comparison)
    ax2 = ax.twinx()  # Create a secondary y-axis for pitch
    ax2.plot(time_axis, pitch_contour_original, label="Original Pitch", color='green', alpha=0.7)
    ax2.plot(time_axis, pitch_contour_user, label="User Pitch", color='orange', alpha=0.7)

    # Highlight areas where tune needs improvement
    for i, diff in enumerate(pitch_diff):
        if diff > 1:  # 1 Hz threshold for significant difference
            ax2.axvspan(time_axis[i], time_axis[i+1] if i+1 < len(time_axis) else time_axis[i], 
                        color='red', alpha=0.3, label="Tune Correction Needed" if i == 0 else "")

    ax.set_ylabel('Amplitude')
    ax2.set_ylabel('Pitch (Hz)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display plot
    st.pyplot(fig)

    st.write(f"DTW distance (rhythm comparison): {distance:.2f}")
    st.write(f"Pitch difference (mean): {np.mean(pitch_diff):.2f} Hz")
    st.write(f"RMS Volume of Original: {rms_original:.2f}, User: {rms_user:.2f}")
        
    ## With radio button for selection
    # Radio button to select which attribute to display
    attribute = st.radio(
        "Select the attribute to display on the graph:",
        ("Pitch Difference", "Volume Difference")
    )

    # Plot the selected attribute
    fig, ax = plt.subplots()

    # Plot waveforms in the background for both original and user
    librosa.display.waveshow(y_original, sr=sr_original, ax=ax, label='Original', alpha=0.5)
    librosa.display.waveshow(y_user, sr=sr_user, ax=ax, label='User', alpha=0.5)

    # Plot the selected attribute
    if attribute == "Pitch Difference":
        # Plot pitch contours
        ax2 = ax.twinx()  # Create a secondary y-axis for pitch
        ax2.plot(time_axis, pitch_contour_original, label="Original Pitch", color='green', alpha=0.7)
        ax2.plot(time_axis, pitch_contour_user, label="User Pitch", color='orange', alpha=0.7)
        
        # Highlight areas where tune needs improvement
        for i, diff in enumerate(pitch_diff):
            if diff > 1:  # 1 Hz threshold for significant difference
                ax2.axvspan(time_axis[i], time_axis[i+1] if i+1 < len(time_axis) else time_axis[i], 
                            color='red', alpha=0.3, label="Tune Correction Needed" if i == 0 else "")
        
        ax2.set_ylabel('Pitch (Hz)')
        ax2.legend(loc='upper right')

    elif attribute == "Volume Difference":
        # Plot RMS volume for both original and user
        ax.bar("Original Volume", rms_original, color='blue', alpha=0.7)
        ax.bar("User Volume", rms_user, color='orange', alpha=0.7)
        st.write(f"Volume Difference: {volume_diff:.2f}")
    
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper left')

    # Display plot
    st.pyplot(fig)
    ##

    st.success("Analysis Complete!")

else:
    st.warning("Please upload both original and user audio files to proceed.")
