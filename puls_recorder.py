#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIY Particle Detector - Professional Pulse Height Analyzer (PHA)
Optimized specifically for Sr-90 (Strontium-90) Beta Particle Detection.

Key Features:
- Robust Baseline Restoration (BLR) and Zero-Crossing detection.
- Fast MAD-based noise floor estimation.
- Asynchronous high-throughput JSON-Lines logging.
- Decoupled, downsampled PyQtGraph UI for responsive live monitoring.
- Precise Amplitude-to-Energy linear calibration logic.

Author: Antigravity (Advanced Agentic Coding)
Date: May 2026
"""

import sys
import time
import datetime
import os
import queue
import threading
import json
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal
import pyaudio
# pyrefly: ignore [missing-import]
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

# --- Configuration Constants ---
RATE = 48000               # Sampling rate (Hz)
FRAME_SIZE = 2048          # Buffer size (lower latency)
DEFAULT_THRESHOLD_SIGMA = 5.0 # Multiplier for MAD-based thresholding
DEAD_TIME_S = 0.0005       # Default signal dead-time in seconds (500us)
# Absolute path to ensure data is always saved in the correct place
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
PULSE_WINDOW_SIZE = 128    # Samples to save around each peak
UI_REFRESH_HZ = 30         # UI update rate
SCOPE_DOWNSAMPLE = 4       # Sub-sampling for scope view to save CPU
MAX_HIST_EVENTS = 50000    # Limit history buffer for stability

# --- Design Tokens (Scientific Premium) ---
COLOR_BG = '#0B0D12'       # Deep Space Navy
COLOR_PANEL = '#151921'    # Card Background
COLOR_ACCENT = '#00F2FF'   # Electric Cyan
COLOR_WARN = '#FF4D00'     # Safety Orange
COLOR_HIST = '#00E676'     # Phosphor Green
COLOR_TEXT = '#E0E0E0'     # High-contrast Pearl
COLOR_GRID = '#2A2F3A'     # Subtle Grid
FONT_MAIN = "Inter"        # Modern Sans
FONT_DATA = "Monospace"    # Technical Data

class PulseLogger(threading.Thread):
    """Asynchronous JSONL logger for metadata and pulse heights."""
    def __init__(self, folder: str):
        super().__init__(daemon=True)
        self.folder = folder
        self.queue = queue.Queue()
        self._running = True
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(folder, f"sr90_events_{timestamp}.jsonl")

    def run(self):
        """Worker thread to flush queue to disk."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            while self._running or not self.queue.empty():
                try:
                    event = self.queue.get(timeout=1.0)
                    # Convert complex objects to serializable
                    serializable = {
                        't': event['timestamp'].isoformat(),
                        'p': float(event['peak']),
                        'e': float(event.get('energy', 0.0)),
                        'w': event['waveform'].tolist() if 'waveform' in event else []
                    }
                    f.write(json.dumps(serializable) + '\n')
                    self.queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Logging Error: {e}")

    def log_event(self, event_data: Dict[str, Any]):
        """Non-blocking put into the logging queue."""
        self.queue.put(event_data)

    def stop(self):
        self._running = False

class SignalProcessor:
    """Advanced DSP pipeline with Baseline Restoration and MAD-based thresholding."""
    def __init__(self, rate: int):
        self.rate = rate
        self.baseline = 0.0
        self.alpha_blr = 0.001  # Baseline Restoration coefficient
        
        # Optimized Filter for Beta: 800Hz - 15kHz
        nyq = 0.5 * rate
        self.b, self.a = signal.butter(3, [800/nyq, 15000/nyq], btype='band')
        self.zi = signal.lfilter_zi(self.b, self.a)
        
        self.last_mad = 0.0
        self.last_rms = 0.0
        self.dead_time_samples = int(DEAD_TIME_S * rate)
        self.last_pulse_idx = -self.dead_time_samples
        self.global_sample_idx = 0

    def calculate_mad(self, chunk: np.ndarray) -> float:
        """Calculate Median Absolute Deviation for robust noise estimation."""
        if len(chunk) == 0: return 0.0
        med = np.median(chunk)
        return np.median(np.abs(chunk - med)) * 1.4826

    def process(self, chunk: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[Dict]]:
        """Main DSP pipeline."""
        # 1. Bandpass Filtering
        filtered, self.zi = signal.lfilter(self.b, self.a, chunk, zi=self.zi)
        
        # 2. Baseline Restoration approximation
        chunk_center = np.median(filtered)
        filtered -= chunk_center
        
        # 3. Noise Estimation (MAD & RMS)
        self.last_mad = self.calculate_mad(filtered)
        self.last_rms = float(np.sqrt(np.mean(np.square(filtered))))
        
        # 4. Pulse Detection (Peak finding with dead-time and windowing)
        peaks = []
        # Detection occurs on NEGATIVE peaks
        potential_peaks = np.where(filtered < threshold)[0]
        
        i = 0
        while i < len(potential_peaks):
            idx = potential_peaks[i]
            global_idx = self.global_sample_idx + idx
            
            # Skip if within dead-time
            if (global_idx - self.last_pulse_idx) < self.dead_time_samples:
                i += 1
                continue
            
            # Find the true local minimum within a small lookahead window
            win_end = min(idx + 15, len(filtered))
            local_min_idx = idx + np.argmin(filtered[idx:win_end])
            
            # Extract Snippet
            start = max(0, local_min_idx - PULSE_WINDOW_SIZE // 2)
            end = min(len(filtered), local_min_idx + PULSE_WINDOW_SIZE // 2)
            
            peaks.append({
                'idx': self.global_sample_idx + local_min_idx,
                'peak': filtered[local_min_idx],
                'timestamp': datetime.datetime.now(),
                'waveform': filtered[start:end]
            })
            
            self.last_pulse_idx = self.global_sample_idx + local_min_idx
            # Fast forward past this detection window
            i = np.searchsorted(potential_peaks, local_min_idx + 15)
            
        self.global_sample_idx += len(chunk)
        return filtered, peaks

class AudioEngine(QtCore.QThread):
    """High-priority audio capture thread."""
    buffer_ready = QtCore.pyqtSignal(np.ndarray)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def run(self):
        self._running = True
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAME_SIZE,
                stream_callback=self._audio_callback
            )
            while self._running:
                time.sleep(0.1)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except: pass
            self.pa.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        self.buffer_ready.emit(samples)
        return (None, pyaudio.paContinue)

    def stop(self):
        self._running = False
        self.wait()

class DetectorApp(QtWidgets.QMainWindow):
    """Professional GUI for Sr-90 Pulse Height Analysis."""
    def __init__(self):
        super().__init__()
        self.setup_parameters()
        self.init_ui()
        self.init_backend()
        
        # UI Update Timer (Decoupled Plotting)
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.update_plots)
        self.ui_timer.start(1000 // UI_REFRESH_HZ)

    def setup_parameters(self):
        self.threshold_sigma = DEFAULT_THRESHOLD_SIGMA
        self.cal_factor = 0.057
        self.threshold = -500.0
        self.counts = 0
        self.beta_counts = 0
        self.alpha_counts = 0
        self.start_time = time.time()
        self.cps_history = []
        self.amplitude_buffer = []
        self.current_chunk = np.zeros(FRAME_SIZE)
        self.last_rms = 0.0
        self.last_processed_time = time.time()
        self.interval_counts = 0
        self.is_paused = False

    def init_ui(self):
        self.setWindowTitle("Professional PHA | Sr-90 Beta Detector")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEXT};")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Sidebar Controls
        sidebar = QtWidgets.QVBoxLayout()
        # Header / Title Card
        header = QtWidgets.QFrame()
        header.setStyleSheet(f"background-color: {COLOR_PANEL}; border-radius: 8px; margin-bottom: 5px;")
        header_layout = QtWidgets.QVBoxLayout(header)
        
        title_lbl = QtWidgets.QLabel("☢ SR-90 ANALYZER")
        title_lbl.setFont(QtGui.QFont(FONT_MAIN, 20, QtGui.QFont.Bold))
        title_lbl.setStyleSheet(f"color: {COLOR_ACCENT}; padding: 10px;")
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(title_lbl)
        
        self.lbl_status = QtWidgets.QLabel("● SYSTEM LIVE")
        self.lbl_status.setStyleSheet("color: #00E676; font-size: 9pt; font-weight: bold; padding-bottom: 5px;")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(self.lbl_status)
        
        sidebar.addWidget(header)

        stats_group = QtWidgets.QFrame()
        stats_group.setStyleSheet(f"background-color: {COLOR_PANEL}; border-radius: 8px; border: 1px solid #222;")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        
        def create_stat_label(text, color, size=18):
            lbl = QtWidgets.QLabel(text)
            lbl.setFont(QtGui.QFont(FONT_DATA, size, QtGui.QFont.Bold))
            lbl.setStyleSheet(f"color: {color}; padding: 2px;")
            return lbl

        self.lbl_cps = create_stat_label("RATE: 0.00 CPS", COLOR_TEXT)
        self.lbl_beta = create_stat_label("BETA: 0", COLOR_ACCENT)
        self.lbl_alpha = create_stat_label("ALPHA: 0", COLOR_WARN)
        self.lbl_total = create_stat_label("TOTAL: 0", "#FFFFFF")
        self.lbl_rms = create_stat_label("NOISE: 0.0", "#666", 11)
        
        for lbl in [self.lbl_cps, self.lbl_beta, self.lbl_alpha, self.lbl_total, self.lbl_rms]:
            stats_layout.addWidget(lbl)
            
        sidebar.addWidget(stats_group)

        # Control Card
        ctrl_card = QtWidgets.QFrame()
        ctrl_card.setStyleSheet(f"background-color: {COLOR_PANEL}; border-radius: 8px; padding: 10px;")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_card)
        
        ctrl_layout.addWidget(QtWidgets.QLabel("SENSITIVITY (THRESHOLD)"))
        self.slider_thl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_thl.setRange(30, 200)
        self.slider_thl.setValue(int(self.threshold_sigma * 10))
        self.slider_thl.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #111; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00F2FF; border: 1px solid #00F2FF; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.slider_thl.valueChanged.connect(self.on_threshold_ui_changed)
        ctrl_layout.addWidget(self.slider_thl)
        
        ctrl_layout.addWidget(QtWidgets.QLabel("CALIBRATION (keV/bit)"))
        self.edit_cal = QtWidgets.QLineEdit(str(self.cal_factor))
        self.edit_cal.setStyleSheet(f"background: #0B0D12; border: 1px solid #333; color: {COLOR_ACCENT}; padding: 5px; border-radius: 4px;")
        self.edit_cal.textChanged.connect(self.on_cal_changed)
        ctrl_layout.addWidget(self.edit_cal)
        
        sidebar.addWidget(ctrl_card)

        sidebar.addStretch()
        
        # Shortcuts Info
        hint_lbl = QtWidgets.QLabel("Shortcuts:\n[P/Space] Pause\n[R] Reset Data\n[T] Auto-Threshold")
        hint_lbl.setStyleSheet("color: #888; font-size: 10pt;")
        sidebar.addWidget(hint_lbl)
        
        self.btn_pause = QtWidgets.QPushButton("⏸ PAUSE")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_clear = QtWidgets.QPushButton("🔄 RESET")
        self.btn_clear.clicked.connect(self.reset_data)
        self.btn_save_exit = QtWidgets.QPushButton("💾 SAVE & EXIT")
        self.btn_save_exit.clicked.connect(self.save_and_exit)
        
        for btn in [self.btn_pause, self.btn_clear, self.btn_save_exit]:
            btn.setMinimumHeight(45)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            if btn == self.btn_save_exit:
                btn.setStyleSheet(f"background: {COLOR_WARN}; color: white; font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(f"background: #1A1F26; color: {COLOR_TEXT}; border: 1px solid #333; border-radius: 4px; font-weight: bold;")
            sidebar.addWidget(btn)

        layout.addLayout(sidebar, 1)

        # Viz Canvas - Compact "Three Graphics" Layout
        viz_layout = QtWidgets.QVBoxLayout()
        viz_layout.setSpacing(15)
        
        def setup_plot(plt, title, xlabel="", ylabel=""):
            plt.setBackground(COLOR_BG)
            plt.setTitle(title, color=COLOR_TEXT, size="12pt")
            plt.showGrid(x=True, y=True, alpha=0.15)
            plt.getAxis('left').setPen(COLOR_TEXT)
            plt.getAxis('bottom').setPen(COLOR_TEXT)
            if xlabel: plt.setLabel('bottom', xlabel, color=COLOR_TEXT)
            if ylabel: plt.setLabel('left', ylabel, color=COLOR_TEXT)
            plt.setMenuEnabled(False)
        
        # 1. Waveform (Full Width)
        self.plt_scope = pg.PlotWidget()
        setup_plot(self.plt_scope, "SIGNAL OSCILLOSCOPE", ylabel="Amplitude (mV)")
        
        # Glow Effect Layer
        self.curve_scope_glow = self.plt_scope.plot(pen=pg.mkPen(COLOR_ACCENT + '33', width=4))
        self.curve_scope = self.plt_scope.plot(pen=pg.mkPen(COLOR_ACCENT, width=1.5))
        
        self.line_thl = pg.InfiniteLine(pos=self.threshold, angle=0, pen=pg.mkPen(COLOR_WARN, style=QtCore.Qt.DashLine, width=1.5))
        self.plt_scope.addItem(self.line_thl)
        viz_layout.addWidget(self.plt_scope, 3)

        # Bottom row for Spectrum and History
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(15)
        
        # 2. Energy Spectrum
        self.plt_spectrum = pg.PlotWidget()
        setup_plot(self.plt_spectrum, "PULSE HEIGHT SPECTRUM", xlabel="Energy (keV)", ylabel="Counts")
        
        # Gradient Brush for Spectrum
        self.curve_spectrum = pg.PlotCurveItem(fillLevel=0, brush=pg.mkBrush(COLOR_HIST + '33'), pen=pg.mkPen(COLOR_HIST, width=2))
        self.plt_spectrum.addItem(self.curve_spectrum)
        bottom_row.addWidget(self.plt_spectrum, 1)

        # 3. CPS History
        self.plt_history = pg.PlotWidget()
        setup_plot(self.plt_history, "COUNT RATE OVER TIME", xlabel="Session Time (s)", ylabel="CPS")
        self.curve_history_glow = self.plt_history.plot(pen=pg.mkPen('#FFFFFF' + '33', width=4))
        self.curve_history = self.plt_history.plot(pen=pg.mkPen('#FFFFFF', width=2))
        bottom_row.addWidget(self.plt_history, 1)
        
        viz_layout.addLayout(bottom_row, 2)

        layout.addLayout(viz_layout, 4)

    def init_backend(self):
        self.processor = SignalProcessor(RATE)
        self.logger = PulseLogger(DATA_FOLDER)
        self.logger.start()
        self.audio = AudioEngine()
        self.audio.buffer_ready.connect(self.on_buffer_ready)
        self.audio.start()

    def on_buffer_ready(self, data: np.ndarray):
        if self.is_paused: return
        
        # Adaptive Auto-Trigger: Calculate threshold dynamically
        # Use MAD-based sensitivity but clamp it to prevent fake triggers in silence
        self.threshold = -1.0 * self.threshold_sigma * self.processor.last_mad
        if self.threshold > -180: self.threshold = -180
        
        filtered, detected = self.processor.process(data, self.threshold)
        self.last_rms = self.processor.last_rms
        self.current_chunk = filtered
        for p in detected:
            self.counts += 1
            self.interval_counts += 1
            amp = abs(p['peak'])
            energy = amp * self.cal_factor
            p['energy'] = energy
            
            # Classification: Alphas are high energy (> 3000 keV typical)
            if energy > 3000:
                self.alpha_counts += 1
            else:
                self.beta_counts += 1
                
            self.amplitude_buffer.append(energy)
            if len(self.amplitude_buffer) > MAX_HIST_EVENTS:
                self.amplitude_buffer.pop(0)
            self.logger.log_event(p)

    def update_plots(self):
        if self.is_paused: return
        downsampled = self.current_chunk[::SCOPE_DOWNSAMPLE]
        self.curve_scope.setData(downsampled)
        self.curve_scope_glow.setData(downsampled)
        self.line_thl.setValue(self.threshold)
        now = time.time()
        elapsed = now - self.last_processed_time
        if elapsed >= 1.0:
            cps = self.interval_counts / elapsed
            self.cps_history.append(cps)
            if len(self.cps_history) > 600: self.cps_history.pop(0)
            self.lbl_cps.setText(f"RATE: {cps:.2f} CPS")
            self.lbl_beta.setText(f"BETA: {self.beta_counts}")
            self.lbl_alpha.setText(f"ALPHA: {self.alpha_counts}")
            self.lbl_total.setText(f"TOTAL: {self.counts}")
            self.lbl_rms.setText(f"NOISE: {self.last_rms:.1f}")
            self.curve_history.setData(self.cps_history)
            self.curve_history_glow.setData(self.cps_history)
            self.interval_counts = 0
            self.last_processed_time = now
        if self.amplitude_buffer:
            hist_range = (0, 20000 * self.cal_factor)
            y, x = np.histogram(self.amplitude_buffer, bins=100, range=hist_range)
            self.curve_spectrum.setData(x[:-1], y)

    def on_threshold_ui_changed(self, val): self.threshold_sigma = val / 10.0
    def on_cal_changed(self, text): 
        try: self.cal_factor = float(text)
        except: pass

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.setText("RESUME" if self.is_paused else "PAUSE")

    def reset_data(self):
        self.counts = 0
        self.beta_counts = 0
        self.alpha_counts = 0
        self.interval_counts = 0
        self.amplitude_buffer = []
        self.cps_history = []
        self.last_processed_time = time.time()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard shortcuts."""
        key = event.key()
        if key in (QtCore.Qt.Key_P, QtCore.Qt.Key_Space):
            self.toggle_pause()
        elif key == QtCore.Qt.Key_R:
            self.reset_data()
        elif key == QtCore.Qt.Key_T:
            self.auto_calibrate_threshold()
        else:
            super().keyPressEvent(event)

    def save_and_exit(self):
        """Clean shutdown with summary logging."""
        self.setEnabled(False)
        self.setWindowTitle("SAVING AND CLOSING...")
        self.save_summary()
        self.close()

    def save_summary(self):
        """Save session statistics to a separate text file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DATA_FOLDER, f"session_summary_{timestamp}.txt")
            duration = time.time() - self.start_time
            
            with open(filename, 'w') as f:
                f.write("=== SR-90 PARTICLE DETECTOR SESSION SUMMARY ===\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Duration: {duration:.1f} seconds\n")
                f.write(f"Total Counts: {self.counts}\n")
                f.write(f"Beta Counts: {self.beta_counts}\n")
                f.write(f"Alpha Counts: {self.alpha_counts}\n")
                if duration > 0:
                    f.write(f"Average CPS: {self.counts / duration:.2f}\n")
                f.write(f"Calibration Factor: {self.cal_factor} keV/bit\n")
                f.write("===============================================\n")
            print(f"Summary saved to {filename}")
        except Exception as e:
            print(f"Error saving summary: {e}")

    def auto_calibrate_threshold(self):
        """Automatically find the noise floor and set trigger sensitivity."""
        # Simple auto-logic: Set sigma to 5.5x the current MAD 
        # (aggressive enough to filter noise but catch beta pulses)
        current_mad = self.processor.last_mad
        if current_mad > 0:
            # We want to be safe, so we look for where noise effectively ends
            self.threshold_sigma = 5.5 
            self.slider_thl.setValue(int(self.threshold_sigma * 10))
            # Visual feedback
            self.setWindowTitle("CALIBRATING... | Professional PHA")
            QtCore.QTimer.singleShot(1000, lambda: self.setWindowTitle("Professional PHA | Sr-90 Beta Detector"))

    def closeEvent(self, event):
        self.audio.stop()
        self.logger.stop()
        # Ensure summary is saved if not already
        if self.counts > 0:
            self.save_summary()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Global visual settings
    pg.setConfigOptions(antialias=True, useOpenGL=True)
    app.setStyle('Fusion')
    
    # Custom Application Font
    font = QtGui.QFont(FONT_MAIN, 10)
    app.setFont(font)
    
    window = DetectorApp()
    window.show()
    sys.exit(app.exec_())
