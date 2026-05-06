#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIY Particle Detector - Professional Pulse Recorder
Optimized for Sr-90 (Strontium-90) Beta Particle Detection.

Features:
- Real-time digital band-pass filtering (Butterworth)
- High-performance queue-based processing (no buffer under-runs)
- Adaptive and manual thresholding
- Advanced spectral analysis (Pulse Height Distribution)
- Dark-mode professional GUI with CPS history

Author: Antigravity (Advanced Agentic Coding)
Date: May 2026
"""

import sys
import time
import datetime
import os
import queue
import threading
from typing import List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from scipy import signal
import pyaudio
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

# --- Configuration Constants ---
RATE = 48000               # Sampling rate (Hz)
FRAME_SIZE = 4096          # Buffer size
DEFAULT_THRESHOLD = -400   # Initial trigger level
MIN_ALPHA_PEAK = -2500     # Threshold to distinguish alpha/beta (calibration dependent)
DEAD_TIME_S = 0.002        # Signal dead-time in seconds
DATA_FOLDER = "./data"     # Where to save recordings
FILTER_LOW_FC = 1000       # High-pass corner (Hz) to remove hum
FILTER_HIGH_FC = 12000     # Low-pass corner (Hz) to remove noise
SAVE_RAW_WAVEFORMS = True  # Save pulse shapes for post-analysis
PULSE_WINDOW_SIZE = 128    # Samples to save around each peak
CALIBRATION_FACTOR = 1.0   # Rough CPS to Energy/Activity scale

# --- Theme Configuration ---
COLOR_BACKGROUND = '#121212'
COLOR_SIGNAL = '#00f2ff'   # Cyan
COLOR_TRIGGER = '#ff3b3b'  # Red
COLOR_CPS = '#ffd700'      # Gold
COLOR_HIST = '#a8ff00'     # Lime

class PulseLogger(threading.Thread):
    """Background thread to handle data persistence without blocking GUI/Audio."""
    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self.queue = queue.Queue()
        self._running = True
        self.output_file = None
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"{folder}/pulses_raw_{timestamp}.pkl"

    def run(self):
        accumulated = []
        last_save = time.time()
        
        while self._running or not self.queue.empty():
            try:
                # Use timeout to allow checking self._running
                pulse_data = self.queue.get(timeout=0.5)
                accumulated.append(pulse_data)
                
                # Periodically dump to disk or if we have enough
                if len(accumulated) >= 100 or (time.time() - last_save > 10):
                    self._save_batch(accumulated)
                    accumulated = []
                    last_save = time.time()
            except queue.Empty:
                if accumulated:
                    self._save_batch(accumulated)
                    accumulated = []
                continue

    def _save_batch(self, batch):
        """Append batch to pickle or other format. For simplicity here, we use a growing list in a dataframe if it's the first time, but for real async we could use HDF5 or just append rows to a CSV."""
        # We don't save the full waveform snippet in CSV (too bulky), just peak info
        # but to satisfy "background logger", we could append to a CSV.
        temp_df = pd.DataFrame(batch)
        csv_name = self.filename.replace(".pkl", ".csv")
        header = not os.path.exists(csv_name)
        # We don't save the full waveform snippet in CSV (too bulky), just peak info
        summary_df = temp_df.drop(columns=['waveform'])
        summary_df.to_csv(csv_name, mode='a', header=header, index=False)
        
    def log(self, pulse_dict):
        self.queue.put(pulse_dict)

    def stop(self):
        self._running = False

class SignalProcessor:
    """Handles digital filtering and pulse detection."""
    def __init__(self, rate: int):
        self.rate = rate
        self.smoothing_win = 5 # Small moving average
        # 4th order Butterworth bandpass
        nyq = 0.5 * rate
        low = FILTER_LOW_FC / nyq
        high = FILTER_HIGH_FC / nyq
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        self.zi = signal.lfilter_zi(self.b, self.a)
        
        self.dead_time_samples = int(DEAD_TIME_S * rate)
        self.last_pulse_sample = -self.dead_time_samples
        self.current_sample_idx = 0
        self.last_noise_rms = 0.0

    def update_filter(self, low_fc: float, high_fc: float):
        """Update Butterworth filter coefficients dynamically."""
        nyq = 0.5 * self.rate
        low = max(0.001, low_fc / nyq)
        high = min(0.999, high_fc / nyq)
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        # We don't reset zi to avoid discontinuities, or we do it if noise is too high
        # self.zi = signal.lfilter_zi(self.b, self.a)

    def process(self, chunk: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[dict]]:
        """Filters a chunk and returns detected pulses."""
        filtered, self.zi = signal.lfilter(self.b, self.a, chunk, zi=self.zi)
        
        # Apply additional smoothing to remove high-freq jitter
        if self.smoothing_win > 1:
            filtered = np.convolve(filtered, np.ones(self.smoothing_win)/self.smoothing_win, mode='same')

        # Calculate Noise RMS (rough estimate from first parts of chunk)
        self.last_noise_rms = np.std(filtered[:int(len(filtered)/4)]) if len(filtered) > 100 else 0
        
        pulses = []
        # Detection logic: find local minima below threshold
        # We use a simple peek detection with dead-time
        for i in range(1, len(filtered) - 1):
            val = filtered[i]
            global_idx = self.current_sample_idx + i
            
            # Check for peak (local minimum) and threshold
            if val < threshold and val < filtered[i-1] and val < filtered[i+1]:
                # Check for dead-time
                if (global_idx - self.last_pulse_sample) > self.dead_time_samples:
                    self.last_pulse_sample = global_idx
                    # Extract waveform snippet centered on peak
                    start = max(0, i - PULSE_WINDOW_SIZE // 2)
                    end = min(len(filtered), i + PULSE_WINDOW_SIZE // 2)
                    waveform_snippet = filtered[start:end].copy()
                    
                    pulses.append({
                        'idx': global_idx,
                        'peak': val,
                        'timestamp': datetime.datetime.now(),
                        'waveform': waveform_snippet
                    })
        
        self.current_sample_idx += len(chunk)
        return filtered, pulses

class AudioStreamProvider(QtCore.QThread):
    """Low-level audio capture thread using PyAudio."""
    chunk_ready = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self._running = False

    def run(self):
        self._running = True
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAME_SIZE,
                stream_callback=self._callback
            )
            while self._running:
                time.sleep(0.1)
        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.stop()

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        self.chunk_ready.emit(samples)
        return (None, pyaudio.paContinue)

    def stop(self):
        self._running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class DetectorMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Particle Detector - Sr-90 Optimized")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(f"background-color: {COLOR_BACKGROUND}; color: white;")

        # Data State
        self.threshold = DEFAULT_THRESHOLD
        self.total_counts = 0
        self.counts_history = []
        self.time_history = []
        self.amplitude_history = []
        self.start_time = time.time()
        self.recorded_pulses = [] # List of dicts for efficient storage
        self.alpha_counts = 0
        self.beta_counts = 0
        self.min_alpha_peak = MIN_ALPHA_PEAK
        self.filter_low = FILTER_LOW_FC
        self.filter_high = FILTER_HIGH_FC
        
        self.is_paused = False
        self.last_cps_calc = time.time()
        self.counts_in_interval = 0
        self.current_cps = 0.0

        # UI Setup
        self.setup_ui()
        self.setup_audio()
        
        # Async Logger
        self.logger = PulseLogger(DATA_FOLDER)
        self.logger.start()
        
        # Timers
        self.stats_timer = QtCore.QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Header Info
        self.header = QtWidgets.QLabel("Initializing System...")
        self.header.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        layout.addWidget(self.header)

        # Plot Area
        plots_layout = QtWidgets.QHBoxLayout()
        
        # Left: Main Oscilloscope
        self.scope_widget = pg.PlotWidget(title="Live Pulse View")
        self.scope_widget.setBackground(COLOR_BACKGROUND)
        self.scope_widget.setYRange(-15000, 15000)
        self.scope_curve = self.scope_widget.plot(pen=pg.mkPen(COLOR_SIGNAL, width=1))
        self.threshold_line = pg.InfiniteLine(pos=self.threshold, angle=0, pen=pg.mkPen(COLOR_TRIGGER, style=QtCore.Qt.DashLine))
        self.scope_widget.addItem(self.threshold_line)
        plots_layout.addWidget(self.scope_widget, 4)

        # Right sidebar: CPS History and Histogram
        side_layout = QtWidgets.QVBoxLayout()
        
        self.history_widget = pg.PlotWidget(title="Count Rate (CPS)")
        self.history_widget.setBackground(COLOR_BACKGROUND)
        self.history_curve = self.history_widget.plot(pen=pg.mkPen(COLOR_CPS, width=2))
        side_layout.addWidget(self.history_widget)
        
        self.hist_widget = pg.PlotWidget(title="Amplitude Distribution (Spectrum)")
        self.hist_widget.setBackground(COLOR_BACKGROUND)
        self.hist_curve = pg.PlotCurveItem(fillLevel=0, brush=pg.mkBrush(COLOR_HIST + '44'))
        self.hist_widget.addItem(self.hist_curve)
        side_layout.addWidget(self.hist_widget)
        
        plots_layout.addLayout(side_layout, 2)
        layout.addLayout(plots_layout)

        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.btn_pause = QtWidgets.QPushButton("Pause Stream")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_clear = QtWidgets.QPushButton("Clear Data")
        self.btn_clear.clicked.connect(self.clear_data)
        
        self.btn_save = QtWidgets.QPushButton("Save & Exit")
        self.btn_save.clicked.connect(self.close)
        
        for btn in [self.btn_pause, self.btn_clear, self.btn_save]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("background-color: #333; border: 1px solid #555; padding: 5px;")
            controls.addWidget(btn)
            
        layout.addLayout(controls)

        # Settings Panel
        settings_layout = QtWidgets.QHBoxLayout()
        self.lbl_alpha_thl = QtWidgets.QLabel(f"Alpha Threshold: {self.min_alpha_peak}")
        self.slider_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_alpha.setRange(-10000, 0)
        self.slider_alpha.setValue(int(self.min_alpha_peak))
        self.slider_alpha.valueChanged.connect(self.update_alpha_thl)
        
        settings_layout.addWidget(self.lbl_alpha_thl)
        settings_layout.addWidget(self.slider_alpha)
        
        # Filter Controls
        filter_layout = QtWidgets.QVBoxLayout()
        fc_layout = QtWidgets.QHBoxLayout()
        self.lbl_filter = QtWidgets.QLabel(f"Filter: {self.filter_low}-{self.filter_high} Hz")
        self.btn_auto_thl = QtWidgets.QPushButton("Auto-Set Thl (6σ)")
        self.btn_auto_thl.clicked.connect(self.auto_set_threshold)
        
        fc_layout.addWidget(self.lbl_filter)
        fc_layout.addWidget(self.btn_auto_thl)
        filter_layout.addLayout(fc_layout)
        
        # Sliders for Filter
        slider_layout = QtWidgets.QHBoxLayout()
        self.slider_low = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_low.setRange(100, 5000)
        self.slider_low.setValue(int(self.filter_low))
        self.slider_low.valueChanged.connect(self.on_filter_changed)
        
        self.slider_high = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_high.setRange(5000, 20000)
        self.slider_high.setValue(int(self.filter_high))
        self.slider_high.valueChanged.connect(self.on_filter_changed)
        
        slider_layout.addWidget(QtWidgets.QLabel("Low Cut:"))
        slider_layout.addWidget(self.slider_low)
        slider_layout.addWidget(QtWidgets.QLabel("High Cut:"))
        slider_layout.addWidget(self.slider_high)
        filter_layout.addLayout(slider_layout)
        
        layout.addLayout(settings_layout)
        layout.addLayout(filter_layout)

    def setup_audio(self):
        self.processor = SignalProcessor(RATE)
        self.audio_thread = AudioStreamProvider()
        self.audio_thread.chunk_ready.connect(self.on_audio_data)
        self.audio_thread.start()

    def on_audio_data(self, data: np.ndarray):
        if self.is_paused:
            return
            
        # DSP Pipeline
        filtered, detected_pulses = self.processor.process(data, self.threshold)
        
        for p in detected_pulses:
            peak = p['peak']
            self.total_counts += 1
            self.counts_in_interval += 1
            self.amplitude_history.append(abs(peak))
            
            p_type = 'alpha' if peak < self.min_alpha_peak else 'beta'
            if p_type == 'alpha': self.alpha_counts += 1
            else: self.beta_counts += 1

            # Log data
            if SAVE_RAW_WAVEFORMS:
                pulse_record = {
                    'timestamp': p['timestamp'],
                    'peak': peak,
                    'type': p_type,
                    'waveform': p['waveform']
                }
                self.recorded_pulses.append(pulse_record)
                self.logger.log(pulse_record)

        # Update Scope
        self.scope_curve.setData(filtered)
        
    def update_stats(self):
        if self.is_paused:
            return
            
        now = time.time()
        dt = now - self.last_cps_calc
        self.current_cps = self.counts_in_interval / dt
        self.counts_history.append(self.current_cps)
        self.time_history.append(now - self.start_time)
        
        # Keep history reasonable
        if len(self.counts_history) > 100:
            self.counts_history.pop(0)
            self.time_history.pop(0)

        # Update UI
        noise_info = f" | Noise: {self.processor.last_noise_rms:.1f} RMS"
        self.header.setText(f"System Active | Total: {self.total_counts} (α: {self.alpha_counts}, β: {self.beta_counts}) | Rate: {self.current_cps:.2f} CPS{noise_info}")
        self.history_curve.setData(self.time_history, self.counts_history)
        
        # Update Histogram
        if self.amplitude_history:
            y, x = np.histogram(self.amplitude_history, bins=50, range=(0, 20000))
            self.hist_curve.setData(x[:-1], y)
            
        self.counts_in_interval = 0
        self.last_cps_calc = now

    def update_alpha_thl(self, val):
        self.min_alpha_peak = val
        self.lbl_alpha_thl.setText(f"Alpha Threshold: {self.min_alpha_peak}")

    def auto_set_threshold(self):
        """Estimate noise level and set threshold automatically."""
        # We use a 6-sigma rule for very conservative triggering
        new_thl = -6.0 * self.processor.last_noise_rms
        self.threshold = min(-100, new_thl) # Don't set too low to avoid DC offset issues
        self.threshold_line.setValue(self.threshold)
        print(f"Auto-Threshold set to {self.threshold:.1f}")

    def on_filter_changed(self):
        self.filter_low = self.slider_low.value()
        self.filter_high = self.slider_high.value()
        self.lbl_filter.setText(f"Filter: {self.filter_low}-{self.filter_high} Hz")
        self.processor.update_filter(self.filter_low, self.filter_high)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.setText("Resume Stream" if self.is_paused else "Pause Stream")

    def clear_data(self):
        self.total_counts = 0
        self.alpha_counts = 0
        self.beta_counts = 0
        self.counts_history = []
        self.time_history = []
        self.amplitude_history = []
        self.recorded_pulses = []
        self.start_time = time.time()
        
        # Clear GUI
        self.scope_curve.setData([])
        self.history_curve.setData([], [])
        self.hist_curve.setData([], [])
        self.header.setText(f"System Reset | Waiting for data...")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Plus:
            self.threshold -= 20
        elif event.key() == QtCore.Qt.Key_Minus:
            self.threshold += 20
        elif event.key() == QtCore.Qt.Key_P:
            self.toggle_pause()
        elif event.key() == QtCore.Qt.Key_R:
            self.clear_data()
        
        self.threshold = min(0, self.threshold)
        self.threshold_line.setValue(self.threshold)

    def closeEvent(self, event):
        self.audio_thread.stop()
        self.logger.stop()
        self.logger.join()
        
        if self.recorded_pulses:
            print(f"Finalizing {len(self.recorded_pulses)} pulses...")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{DATA_FOLDER}/sr90_data_{timestamp}.pkl"
            df = pd.DataFrame(self.recorded_pulses)
            df.to_pickle(filename)
            print(f"Full dataset saved to {filename}")
            print(f"Metadata summary available in {self.logger.filename.replace('.pkl', '.csv')}")
        
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Set professional dark palette
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)

    window = DetectorMainWindow()
    window.show()
    sys.exit(app.exec_())
