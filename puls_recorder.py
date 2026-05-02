#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scientific Pulse Analysis Pipeline for DIY Particle Detector.
Transitions from simple event counting to high-fidelity feature extraction.
Modules: BLR, Matched Filter, Robust MAD detection, FWHM/Integral Characterization.

Author: Antigravity
Date: May 2026
"""

import sys
import time
import queue
import threading
import datetime
import os
import csv
from collections import deque

import numpy as np
import pandas as pd
import pyaudio
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import signal

# --- SCIENTIFIC CONFIGURATION ---
RATE = 48000            # Sample rate (ADC)
FRAME_SIZE = 2048       # Buffer block size
DATA_FOLDER = "./data"  
HPF_CORNER = 100        # Baseline HPF corner frequency (Hz)
K_THRESHOLD = 6.0       # Threshold multiplier (k * MAD)
DEAD_TIME_MS = 10.0     # Non-paralyzable dead-time in milliseconds

# Pulse Shape Model (Fast Rise + Exponential Decay)
TAU_DECAY = 0.0005      # 500 us decay
TAU_RISE = 0.00005      # 50 us rise

# Ensure data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

class FeatureExtractor:
    """Computes high-fidelity features for each detected pulse."""
    
    @staticmethod
    def extract_all(wf_corrected, p_idx, rate):
        """
        Extracts pulse parameters from a baseline-corrected signal.
        Expected wf_corrected: np.array where pulses are POSITIVE spikes.
        """
        # Define extraction window (5ms)
        win_samples = int(0.005 * rate)
        start = max(0, p_idx - int(0.1 * win_samples))
        end = min(len(wf_corrected), p_idx + int(0.9 * win_samples))
        pulse_wf = wf_corrected[start:end]
        
        # Local peak in window (for sub-sample precision or window alignment)
        local_p_idx = p_idx - start
        amplitude = pulse_wf[local_p_idx]
        
        # 1. Integral (Area under curve in the window)
        integral = np.sum(pulse_wf)
        
        # 2. FWHM (Full Width at Half Maximum) via interpolation
        fwhm = 0.0
        try:
            half_max = amplitude / 2.0
            # Search left
            left_side = pulse_wf[:local_p_idx]
            l_idx = np.where(left_side < half_max)[0][-1]
            # Linear interpolation for sub-sample left crossing
            l_val1, l_val2 = left_side[l_idx], left_side[l_idx+1]
            l_interp = l_idx + (half_max - l_val1) / (l_val2 - l_val1)
            
            # Search right
            right_side = pulse_wf[local_p_idx:]
            r_idx = np.where(right_side < half_max)[0][0]
            # Linear interpolation for sub-sample right crossing
            r_val1, r_val2 = right_side[r_idx-1], right_side[r_idx]
            r_interp = (local_p_idx + r_idx - 1) + (half_max - r_val1) / (r_val2 - r_val1)
            
            fwhm = (r_interp - l_interp) / rate
        except:
            pass # Fallback to 0 if signal is too noisy or short
            
        # 3. Rise Time (10% to 90%)
        rise_time = 0.0
        try:
            v10 = 0.1 * amplitude
            v90 = 0.9 * amplitude
            t10 = np.where(pulse_wf >= v10)[0][0]
            t90 = np.where(pulse_wf >= v90)[0][0]
            rise_time = (t90 - t10) / rate
        except:
            pass

        return {
            "amplitude": float(amplitude),
            "integral": float(integral),
            "fwhm_ms": float(fwhm * 1000),
            "rise_time_us": float(rise_time * 1e6)
        }

class SignalProcessor:
    """Advanced Multi-stage DSP: Baseline -> Matched Filter -> MAD Estimation."""
    def __init__(self, rate):
        self.rate = rate
        # 1. High-pass filter for coarse DC removal (4th order Butterworth SOS)
        self.sos = signal.butter(4, HPF_CORNER, 'hp', fs=rate, output='sos')
        self.filter_state = np.zeros((self.sos.shape[0], 2))
        
        # 2. Matched Filter Kernel (Rise/Decay model)
        t_kernel = np.linspace(0, 0.005, int(rate * 0.005))
        self.kernel = (1 - np.exp(-t_kernel/TAU_RISE)) * np.exp(-t_kernel/TAU_DECAY)
        self.kernel /= np.sum(self.kernel) # Normalize for unity area
        
        # Baseline buffer for slow drift estimation
        self.last_mad = 0.001

    def process(self, frame_int16):
        # Normalize
        x = frame_int16.astype(np.float32) / 32768.0
        
        # Stage 1: Coarse Filtering
        x_filtered, self.filter_state = signal.sosfilt(self.sos, x, zi=self.filter_state)
        
        # Stage 2: Matched Filtering (Convolve with expected pulse shape)
        # Note: Invert signal early as we expect negative spikes
        x_inverted = -x_filtered
        x_matched = signal.convolve(x_inverted, self.kernel, mode='same')
        
        # Stage 3: Robust Noise Estimation (MAD)
        # 1.4826 constant for consistency with Gaussian sigma
        median = np.median(x_matched)
        mad = 1.4826 * np.median(np.abs(x_matched - median))
        self.last_mad = 0.95 * self.last_mad + 0.05 * mad # Moving average MAD
        
        return x_inverted, x_matched, self.last_mad

class PulseDetector:
    """Non-paralyzable dead-time logic and peak hunting."""
    def __init__(self, rate, dead_time_ms):
        self.rate = rate
        self.dt_samples = int((dead_time_ms / 1000.0) * rate)
        self.obs_count = 0
        self.start_time = time.time()
        
    def find_peaks(self, signal_data, threshold):
        # Uses scipy find_peaks with distance constraint for dead-time
        peaks, _ = signal.find_peaks(signal_data, height=threshold, distance=self.dt_samples)
        return peaks

    def get_rates(self):
        elapsed = time.time() - self.start_time
        if elapsed <= 0: return 0.0, 0.0
        
        r_obs = self.obs_count / elapsed
        # Dead-time correction: R_true = R_obs / (1 - R_obs * Tau)
        tau = DEAD_TIME_MS / 1000.0
        r_true = r_obs / (1 - r_obs * tau) if (r_obs * tau) < 1.0 else r_obs
        
        return r_obs, r_true

class DataManager:
    """Scientific Data Logger (CSV/Pickle)."""
    def __init__(self):
        self.records = []
        self.lock = threading.Lock()
        self.filename = datetime.datetime.now().strftime("pulses_%Y-%m-%d_%H-%M-%S")

    def log(self, timestamp, features):
        with self.lock:
            # Classification logic based on FWHM/RiseTime
            # Simple heuristic: Alphas are usually wider/larger
            classification = "alpha" if features["integral"] > 0.5 or features["fwhm_ms"] > 0.5 else "beta"
            
            record = {
                "timestamp": timestamp,
                "type": classification,
                **features
            }
            self.records.append(record)

    def save_all(self):
        if not self.records: return
        
        # Save as Pickle (Full precision)
        df = pd.DataFrame(self.records)
        df.to_pickle(os.path.join(DATA_FOLDER, self.filename + ".pkl"))
        
        # Save as CSV (Analysis ready)
        df.to_csv(os.path.join(DATA_FOLDER, self.filename + ".csv"), index=False)
        print(f"Logged {len(self.records)} pulses to {DATA_FOLDER}")

class AudioStream:
    """Minimal PyAudio input."""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.queue = queue.Queue(maxsize=30)
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
            stream_callback=self._callback
        )

    def _callback(self, in_data, frame_count, time_info, status):
        # Capture absolute ADC time if possible
        now = time.time()
        samples = np.frombuffer(in_data, dtype=np.int16)
        try:
            self.queue.put_nowait((now, samples))
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class ScientificScope(QtWidgets.QMainWindow):
    """Scientific Display & Processing Hub."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Particle Analyzer v2.0")
        self.resize(1200, 750)
        
        # Logic
        self.proc = SignalProcessor(RATE)
        self.det = PulseDetector(RATE, DEAD_TIME_MS)
        self.data = DataManager()
        self.audio = AudioStream()
        
        # UI
        self.init_ui()
        
        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_loop)
        self.timer.start(15)

    def init_ui(self):
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Top Panel: Main Signal
        self.plt_signal = pg.PlotWidget(title="Baseline-Corrected & Matched Signal")
        self.plt_signal.addLegend()
        self.plt_signal.setYRange(0, 0.5)
        self.curve_matched = self.plt_signal.plot(pen='c', name="Matched Output")
        self.line_thl = self.plt_signal.plot(pen=pg.mkPen('r', style=QtCore.Qt.DashLine), name="Adaptive MAD THL")
        self.scatter_peaks = pg.ScatterPlotItem(size=10, brush='m', name="Detected")
        self.plt_signal.addItem(self.scatter_peaks)
        layout.addWidget(self.plt_signal)
        
        # Stats Panel
        self.lbl_stats = QtWidgets.QLabel("Status: Streaming...")
        self.lbl_stats.setFont(QtGui.QFont("Monospace", 12))
        self.lbl_stats.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border-radius: 5px; padding: 10px;")
        layout.addWidget(self.lbl_stats)

    def process_loop(self):
        while not self.audio.queue.empty():
            ts_frame, raw_data = self.audio.queue.get()
            
            # DSP Stage
            corrected_raw, matched, mad = self.proc.process(raw_data)
            thl = K_THRESHOLD * mad
            
            # Detection Stage
            peaks = self.det.find_peaks(matched, thl)
            
            # Feature Extraction Stage
            for p_idx in peaks:
                # Precise timestamp
                abs_ts = ts_frame + (p_idx / RATE)
                feats = FeatureExtractor.extract_all(corrected_raw, p_idx, RATE)
                self.data.log(abs_ts, feats)
                self.det.obs_count += 1
                
            # Update GUI
            self.curve_matched.setData(matched)
            self.line_thl.setData([thl] * FRAME_SIZE)
            self.scatter_peaks.setData(x=peaks, y=matched[peaks])
            
            r_obs, r_true = self.det.get_rates()
            self.lbl_stats.setText(
                f"Observed Rate: {r_obs:6.2f} CPS | Corrected Rate: {r_true:6.2f} CPS\n"
                f"Background Noise (MAD): {mad:.6f} | Events Logged: {self.det.obs_count}"
            )

    def closeEvent(self, event):
        print("Finalizing session...")
        self.audio.stop()
        self.data.save_all()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, useOpenGL=True) # Modern high-perf rendering
    view = ScientificScope()
    view.show()
    sys.exit(app.exec_())