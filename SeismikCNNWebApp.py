import streamlit as st
import numpy as np
from obspy import read
import tensorflow as tf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Konfigurasi model ---
MODEL_SAVE = "model_seismik_cnn"
LABEL_TEXT = {0: "Noise", 1: "Clear Signal", 2: "Earthquake"}
TARGET_LEN = 2048

# --- Fungsi utilitas ---
def classify_sta_lta(wf, sr, sta=1.0, lta=10.0, th_clear=1.8, th_eq=3.5):
    wf = np.asarray(wf, dtype="float32")
    nsta = max(1, int(sta * sr))
    nlta = max(1, int(lta * sr))
    if nlta >= len(wf):
        return 0
    sta_ = np.convolve(np.abs(wf), np.ones(nsta)/nsta, mode="same")
    lta_ = np.convolve(np.abs(wf), np.ones(nlta)/nlta, mode="same")
    lta_[lta_ == 0] = 1e-9
    ratio = sta_ / lta_
    peak = np.max(ratio)
    if peak > th_eq:
        return 2
    elif peak > th_clear:
        return 1
    else:
        return 0

def load_waveform(path, target_len=TARGET_LEN):
    tr = read(path)[0]
    wf = tr.data.astype("float32")
    sr = tr.stats.sampling_rate
    if len(wf) < target_len:
        out = np.zeros(target_len, dtype="float32")
        out[:len(wf)] = wf
        wf = out
    else:
        wf = wf[:target_len]
    return wf, sr

def wf_to_spec(wf, sr):
    f, t, Sxx = spectrogram(
        wf, fs=sr, nperseg=256, noverlap=128,
        scaling="spectrum", mode="magnitude"
    )
    Sxx = 10 * np.log10(Sxx + 1e-8)
    return Sxx.astype("float32")

# --- Load model with caching ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_SAVE)

model = load_model()

# --- Streamlit App ---
st.title("Seismic CNN Web App — Viewer & Performance")

uploaded_files = st.file_uploader(
    "Upload .mseed file(s)", type=["mseed"], accept_multiple_files=True
)
filter_option = st.selectbox("Filter Prediksi", ["All"] + list(LABEL_TEXT.values()))

# --- Session state ---
if 'files_data' not in st.session_state:
    st.session_state.files_data = []
    st.session_state.current_idx = 0
    st.session_state.uploaded_names = []

# --- Proses file baru ---
new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_names]
for file in new_files:
    tmp_path = f"tmp_{file.name}"
    with open(tmp_path, "wb") as f:
        f.write(file.getbuffer())
    wf, sr = load_waveform(tmp_path)
    spec = wf_to_spec(wf, sr)
    inp = spec[np.newaxis, ..., np.newaxis]
    pred_probs = np.squeeze(model.predict(inp, verbose=0))
    pred_label = int(np.argmax(pred_probs))
    gt_label = classify_sta_lta(wf, sr)
    if filter_option != "All" and LABEL_TEXT[pred_label] != filter_option:
        os.remove(tmp_path)
        continue
    st.session_state.files_data.append({
        "name": file.name,
        "wf": wf,
        "sr": sr,
        "spec": spec,
        "pred_label": pred_label,
        "pred_probs": pred_probs,
        "gt_label": gt_label
    })
    st.session_state.uploaded_names.append(file.name)
    os.remove(tmp_path)

# --- Hapus file yang sudah tidak dipilih ---
current_names = [f.name for f in uploaded_files]
st.session_state.files_data = [f for f in st.session_state.files_data if f['name'] in current_names]
st.session_state.uploaded_names = [f['name'] for f in st.session_state.files_data]
if st.session_state.current_idx >= len(st.session_state.files_data):
    st.session_state.current_idx = max(0, len(st.session_state.files_data)-1)

# --- Navigation buttons ---
if st.session_state.files_data:
    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        if st.button("<< Previous"):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
    with col3:
        if st.button("Next >>"):
            st.session_state.current_idx = min(len(st.session_state.files_data)-1, st.session_state.current_idx + 1)

    # --- Tampilkan file saat ini ---
    file = st.session_state.files_data[st.session_state.current_idx]
    fig, axs = plt.subplots(2,1, figsize=(10,6))
    t = np.arange(len(file['wf'])) / file['sr']
    axs[0].plot(t, file['wf'])
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[1].imshow(file['spec'], aspect="auto", origin="lower", cmap="turbo")
    axs[1].set_title("Spectrogram")
    axs[1].set_xlabel("Time bins")
    axs[1].set_ylabel("Frequency bins")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader(f"File: {file['name']} ({st.session_state.current_idx+1}/{len(st.session_state.files_data)})")
    st.write(f"Ground Truth (STA/LTA): {LABEL_TEXT[file['gt_label']]}")
    st.write(f"Predicted: {LABEL_TEXT[file['pred_label']]}")
    st.write(f"Confidence: {file['pred_probs']}")

    # --- Plot Ground Truth vs Prediksi ---
    y_true = [f['gt_label'] for f in st.session_state.files_data]
    y_pred = [f['pred_label'] for f in st.session_state.files_data]
    fig_perf, ax_perf = plt.subplots()
    ax_perf.plot(range(1,len(y_true)+1), y_true, label='Ground Truth', marker='o')
    ax_perf.plot(range(1,len(y_pred)+1), y_pred, label='Prediksi', marker='x')
    ax_perf.set_xlabel('File index')
    ax_perf.set_ylabel('Label')
    ax_perf.set_title('Ground Truth vs Prediksi')
    ax_perf.legend()
    st.pyplot(fig_perf)

    # --- Tabel hasil + CSV ---
    df_results = pd.DataFrame([{
        "File": f['name'],
        "Ground Truth": LABEL_TEXT[f['gt_label']],
        "Predicted": LABEL_TEXT[f['pred_label']],
        **{f"Prob_{LABEL_TEXT[i]}": f['pred_probs'][i] for i in range(len(f['pred_probs']))}
    } for f in st.session_state.files_data])
    st.write("### Table of Predictions")
    st.dataframe(df_results)
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
