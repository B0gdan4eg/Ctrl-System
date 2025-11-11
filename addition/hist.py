#!/usr/bin/env python3
"""
10-bit Image Converter - Mean ± Std Dev Method
Simple converter using statistical (Gaussian) distribution method.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QFileDialog, 
    QMessageBox, QGroupBox, QProgressBar, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageViewer(QLabel):
    """Simple image viewer widget."""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #2b2b2b; color: white;")
        self.setText("No image loaded")
        
    def display_image(self, image):
        """Display OpenCV image (BGR format)."""
        if image is None:
            return
        
        # Convert BGR to RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normalize for display if needed (10-bit to 8-bit for viewing)
        if image.dtype == np.uint16:
            display_image = (rgb_image / 1023.0 * 255.0).astype(np.uint8)
        else:
            display_image = rgb_image
        
        h, w, ch = display_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class HistogramCanvas(FigureCanvas):
    """Custom matplotlib canvas for displaying histogram."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3))
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.input_min = 0
        self.input_max = 1023
        self.histogram_data = None
        
    def plot_histogram(self, image, bit_depth=10):
        """Plot histogram with statistical annotations."""
        self.axes.clear()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        max_value = (2 ** bit_depth) - 1
        hist_size = 2 ** bit_depth
        
        # Scale image to bit depth if needed
        if gray.dtype == np.uint8 and bit_depth > 8:
            gray = (gray.astype(np.float32) * (max_value / 255.0)).astype(np.uint16)
        
        hist = cv2.calcHist([gray], [0], None, [hist_size], [0, max_value + 1])
        self.histogram_data = hist.flatten()
        
        # Calculate statistics
        mean = np.mean(gray)
        std = np.std(gray)
        
        # Plot histogram
        x_values = np.arange(0, len(self.histogram_data))
        self.axes.fill_between(x_values, self.histogram_data, alpha=0.7, color='blue', label='Histogram')
        
        # Add vertical lines for current input min/max
        self.axes.axvline(x=self.input_min, color='red', linestyle='--', linewidth=2, 
                         label=f'Input Min: {self.input_min}', zorder=5)
        self.axes.axvline(x=self.input_max, color='green', linestyle='--', linewidth=2, 
                         label=f'Input Max: {self.input_max}', zorder=5)
        
        # Add mean marker
        self.axes.axvline(x=mean, color='orange', linestyle=':', linewidth=1.5, 
                         label=f'Mean: {mean:.0f}', alpha=0.7)
        
        # Add std dev range (mean ± 2.5σ)
        self.axes.axvspan(mean - 2.5*std, mean + 2.5*std, alpha=0.1, color='orange', 
                         label=f'Mean ± 2.5σ')
        
        self.axes.set_xlabel('Pixel Value')
        self.axes.set_ylabel('Frequency')
        self.axes.set_title(f'Histogram ({bit_depth}-bit) - Mean: {mean:.0f}, Std: {std:.0f}')
        self.axes.legend(loc='upper right', fontsize=8)
        self.axes.grid(True, alpha=0.3)
        
        self.draw()
    
    def update_levels(self, input_min, input_max):
        """Update input level markers on histogram."""
        self.input_min = input_min
        self.input_max = input_max
        
        if self.histogram_data is not None:
            self.axes.clear()
            x_values = np.arange(0, len(self.histogram_data))
            self.axes.fill_between(x_values, self.histogram_data, alpha=0.7, color='blue')
            self.axes.axvline(x=self.input_min, color='red', linestyle='--', linewidth=2, 
                             label=f'Min: {self.input_min}')
            self.axes.axvline(x=self.input_max, color='green', linestyle='--', linewidth=2, 
                             label=f'Max: {self.input_max}')
            self.axes.set_xlabel('Pixel Value')
            self.axes.set_ylabel('Frequency')
            self.axes.set_title('Histogram (10-bit)')
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
            self.draw()


class ImageConverter10Bit(QMainWindow):
    """Main window for 10-bit image converter."""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.original_image = None
        self.converted_image = None
        self.preview_image = None
        self.current_file_path = None
        self.bit_depth = 10
        
        self.init_ui()
        self.apply_style()
        
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("10-bit Image Converter - Mean ± Std Dev")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)
        
        save_layout = QHBoxLayout()
        self.save_10bit_btn = QPushButton("Save as 10-bit")
        self.save_10bit_btn.clicked.connect(lambda: self.save_image(as_10bit=True))
        self.save_10bit_btn.setEnabled(False)
        save_layout.addWidget(self.save_10bit_btn)
        
        self.save_8bit_btn = QPushButton("Save as 8-bit")
        self.save_8bit_btn.clicked.connect(lambda: self.save_image(as_10bit=False))
        self.save_8bit_btn.setEnabled(False)
        save_layout.addWidget(self.save_8bit_btn)
        file_layout.addLayout(save_layout)
        
        self.batch_btn = QPushButton("Batch Convert Folder")
        self.batch_btn.clicked.connect(self.batch_convert)
        file_layout.addWidget(self.batch_btn)
        
        self.batch_recursive_btn = QPushButton("Batch Convert (with subfolders)")
        self.batch_recursive_btn.clicked.connect(self.batch_convert_recursive)
        file_layout.addWidget(self.batch_recursive_btn)
        
        left_layout.addWidget(file_group)
        
        # Image statistics
        info_group = QGroupBox("Image Statistics")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No image loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)
        
        # Manual controls
        manual_group = QGroupBox("Manual Level Adjustment")
        manual_layout = QVBoxLayout(manual_group)
        
        # Input Min
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Input Min:"))
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 1023)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self.on_min_changed)
        min_layout.addWidget(self.min_slider)
        self.min_spinbox = QSpinBox()
        self.min_spinbox.setRange(0, 1023)
        self.min_spinbox.setValue(0)
        self.min_spinbox.valueChanged.connect(self.on_min_spinbox_changed)
        min_layout.addWidget(self.min_spinbox)
        manual_layout.addLayout(min_layout)
        
        # Input Max
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Input Max:"))
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, 1023)
        self.max_slider.setValue(1023)
        self.max_slider.valueChanged.connect(self.on_max_changed)
        max_layout.addWidget(self.max_slider)
        self.max_spinbox = QSpinBox()
        self.max_spinbox.setRange(0, 1023)
        self.max_spinbox.setValue(1023)
        self.max_spinbox.valueChanged.connect(self.on_max_spinbox_changed)
        max_layout.addWidget(self.max_spinbox)
        manual_layout.addLayout(max_layout)
        
        # Buttons
        manual_btn_layout = QVBoxLayout()
        
        self.preview_btn = QPushButton("Preview Levels")
        self.preview_btn.clicked.connect(self.preview_levels)
        self.preview_btn.setEnabled(False)
        manual_btn_layout.addWidget(self.preview_btn)
        
        self.manual_convert_btn = QPushButton("Apply Manual & Convert")
        self.manual_convert_btn.clicked.connect(self.manual_convert)
        self.manual_convert_btn.setEnabled(False)
        manual_btn_layout.addWidget(self.manual_convert_btn)
        
        manual_btn_layout2 = QHBoxLayout()
        self.reset_levels_btn = QPushButton("Reset Levels")
        self.reset_levels_btn.clicked.connect(self.reset_levels)
        manual_btn_layout2.addWidget(self.reset_levels_btn)
        
        self.revert_btn = QPushButton("Revert to Original")
        self.revert_btn.clicked.connect(self.revert_to_original)
        self.revert_btn.setEnabled(False)
        manual_btn_layout2.addWidget(self.revert_btn)
        
        manual_layout.addLayout(manual_btn_layout)
        manual_layout.addLayout(manual_btn_layout2)
        
        left_layout.addWidget(manual_group)
        
        # Auto controls - Mean ± Std Dev only
        auto_group = QGroupBox("Automatic: Mean ± Std Dev (Gaussian)")
        auto_layout = QVBoxLayout(auto_group)
        
        # Description
        desc_label = QLabel(
            "<b>Method:</b> Uses statistical distribution<br>"
            "<b>Formula:</b> Mean ± (σ × Std Dev)<br>"
            "Assumes normal (Gaussian) distribution"
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("padding: 8px; background-color: #e3f2fd; border-radius: 4px; font-size: 10px;")
        auto_layout.addWidget(desc_label)
        
        # Std Dev parameter
        stddev_layout = QHBoxLayout()
        stddev_layout.addWidget(QLabel("Std Dev Range:"))
        self.stddev_spinbox = QDoubleSpinBox()
        self.stddev_spinbox.setRange(1.0, 5.0)
        self.stddev_spinbox.setValue(2.5)
        self.stddev_spinbox.setSingleStep(0.5)
        self.stddev_spinbox.setSuffix(" σ")
        stddev_layout.addWidget(self.stddev_spinbox)
        
        # Coverage info
        self.coverage_label = QLabel("Coverage: 98.8% of data")
        stddev_layout.addWidget(self.coverage_label)
        stddev_layout.addStretch()
        
        self.stddev_spinbox.valueChanged.connect(self.update_coverage_label)
        auto_layout.addLayout(stddev_layout)
        
        self.auto_convert_btn = QPushButton("Auto Levels & Convert to 10-bit")
        self.auto_convert_btn.clicked.connect(self.auto_convert)
        self.auto_convert_btn.setEnabled(False)
        auto_layout.addWidget(self.auto_convert_btn)
        
        left_layout.addWidget(auto_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 5px; background-color: #e8f5e9; border-radius: 3px;")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image viewers
        viewers_group = QGroupBox("Image Preview")
        viewers_layout = QHBoxLayout(viewers_group)
        
        # Original image
        original_container = QVBoxLayout()
        original_container.addWidget(QLabel("<b>Original</b>"))
        self.original_viewer = ImageViewer()
        original_container.addWidget(self.original_viewer)
        viewers_layout.addLayout(original_container)
        
        # Preview/Converted image
        preview_container = QVBoxLayout()
        preview_container.addWidget(QLabel("<b>Preview / Converted</b>"))
        self.preview_viewer = ImageViewer()
        preview_container.addWidget(self.preview_viewer)
        viewers_layout.addLayout(preview_container)
        
        right_layout.addWidget(viewers_group)
        
        # Histogram
        histogram_group = QGroupBox("Histogram")
        histogram_layout = QVBoxLayout(histogram_group)
        self.histogram_canvas = HistogramCanvas()
        histogram_layout.addWidget(self.histogram_canvas)
        right_layout.addWidget(histogram_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        
    def apply_style(self):
        """Apply modern style."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
        """)
    
    def update_coverage_label(self, sigma):
        """Update coverage label based on sigma value."""
        coverage_map = {
            1.0: 68.3, 1.5: 86.6, 2.0: 95.4, 2.5: 98.8, 3.0: 99.7, 
            3.5: 99.95, 4.0: 99.99, 4.5: 99.999, 5.0: 99.9999
        }
        # Find closest value
        closest_sigma = min(coverage_map.keys(), key=lambda x: abs(x - sigma))
        coverage = coverage_map[closest_sigma]
        self.coverage_label.setText(f"Coverage: ~{coverage}% of data")
        
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            self.load_image_from_path(file_path)
            
    def load_image_from_path(self, file_path):
        """Load image from path with Unicode support."""
        try:
            # Load with Unicode support
            if any(ord(char) > 127 for char in file_path):
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                QMessageBox.warning(self, "Error", "Failed to load image.")
                return
            
            self.current_file_path = file_path
            self.original_image = image.copy()
            self.current_image = image.copy()
            self.converted_image = None
            self.preview_image = None
            
            # Display original image
            self.original_viewer.display_image(image)
            self.preview_viewer.setText("No preview")
            
            # Display statistics
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            dtype = image.dtype
            min_val = np.min(image)
            max_val = np.max(image)
            mean_val = np.mean(image)
            std_val = np.std(image)
            
            # Scale to 10-bit for statistics
            if dtype == np.uint8:
                scale_factor = 1023.0 / 255.0
                mean_10bit = mean_val * scale_factor
                std_10bit = std_val * scale_factor
            else:
                mean_10bit = mean_val
                std_10bit = std_val
            
            info_text = f"<b>File:</b> {Path(file_path).name}<br>"
            info_text += f"<b>Size:</b> {width}x{height} px<br>"
            info_text += f"<b>Type:</b> {dtype}<br><br>"
            info_text += f"<b>Mean:</b> {mean_10bit:.1f}<br>"
            info_text += f"<b>Std Dev:</b> {std_10bit:.1f}<br>"
            info_text += f"<b>Mean ± 2.5σ:</b> [{mean_10bit - 2.5*std_10bit:.0f}, {mean_10bit + 2.5*std_10bit:.0f}]"
            self.info_label.setText(info_text)
            
            # Plot histogram
            self.histogram_canvas.plot_histogram(image, self.bit_depth)
            
            # Reset levels
            self.reset_levels()
            
            # Enable buttons
            self.preview_btn.setEnabled(True)
            self.manual_convert_btn.setEnabled(True)
            self.auto_convert_btn.setEnabled(True)
            self.save_10bit_btn.setEnabled(False)
            self.save_8bit_btn.setEnabled(False)
            self.revert_btn.setEnabled(False)
            
            self.status_label.setText(f"Image loaded: {Path(file_path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
            
    def reset_levels(self):
        """Reset levels to default."""
        self.min_slider.setValue(0)
        self.min_spinbox.setValue(0)
        self.max_slider.setValue(1023)
        self.max_spinbox.setValue(1023)
        self.histogram_canvas.update_levels(0, 1023)
        
    def revert_to_original(self):
        """Revert to original image."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.converted_image = None
            self.preview_image = None
            self.preview_viewer.setText("No preview")
            self.original_viewer.display_image(self.original_image)
            self.histogram_canvas.plot_histogram(self.original_image, self.bit_depth)
            self.reset_levels()
            self.save_10bit_btn.setEnabled(False)
            self.save_8bit_btn.setEnabled(False)
            self.revert_btn.setEnabled(False)
            self.status_label.setText("Reverted to original image")
        
    def on_min_changed(self, value):
        """Handle min slider change."""
        self.min_spinbox.setValue(value)
        if value >= self.max_slider.value():
            self.max_slider.setValue(value + 1)
        self.histogram_canvas.update_levels(value, self.max_slider.value())
        
    def on_min_spinbox_changed(self, value):
        """Handle min spinbox change."""
        self.min_slider.setValue(value)
        
    def on_max_changed(self, value):
        """Handle max slider change."""
        self.max_spinbox.setValue(value)
        if value <= self.min_slider.value():
            self.min_slider.setValue(value - 1)
        self.histogram_canvas.update_levels(self.min_slider.value(), value)
        
    def on_max_spinbox_changed(self, value):
        """Handle max spinbox change."""
        self.max_slider.setValue(value)
        
    def apply_levels_adjustment(self, image, input_min, input_max):
        """Apply levels adjustment to image."""
        img_float = image.astype(np.float32)
        
        # Scale 8-bit to 10-bit if needed
        if image.dtype == np.uint8:
            img_float = img_float * (1023.0 / 255.0)
        
        # Apply levels (clip and stretch)
        img_float = np.clip(img_float, input_min, input_max)
        
        if input_max > input_min:
            img_float = (img_float - input_min) / (input_max - input_min) * 1023.0
        
        img_10bit = np.clip(img_float, 0, 1023).astype(np.uint16)
        
        return img_10bit
        
    def preview_levels(self):
        """Preview the levels adjustment."""
        if self.original_image is None:
            return
        
        input_min = self.min_slider.value()
        input_max = self.max_slider.value()
        
        self.preview_image = self.apply_levels_adjustment(self.original_image, input_min, input_max)
        self.preview_viewer.display_image(self.preview_image)
        self.status_label.setText(f"Preview with levels {input_min}-{input_max}")
        
    def manual_convert(self):
        """Convert image with manual settings."""
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return
        
        input_min = self.min_slider.value()
        input_max = self.max_slider.value()
        
        self.converted_image = self.apply_levels_adjustment(self.original_image, input_min, input_max)
        self.preview_viewer.display_image(self.converted_image)
        
        self.status_label.setText(f"✓ Converted with levels {input_min}-{input_max}")
        self.save_10bit_btn.setEnabled(True)
        self.save_8bit_btn.setEnabled(True)
        self.revert_btn.setEnabled(True)
        
        QMessageBox.information(self, "Success", 
            f"Image converted to 10-bit!\n\n"
            f"Input range: {input_min} - {input_max}\n"
            f"Output range: 0 - 1023")
        
    def auto_convert(self):
        """Convert using Mean ± Std Dev method."""
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return
        
        # Convert to grayscale for analysis
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image.copy()
        
        # Scale to 10-bit
        if gray.dtype == np.uint8:
            gray = (gray.astype(np.float32) * (1023.0 / 255.0)).astype(np.uint16)
        
        # Calculate Mean ± Std Dev
        std_range = self.stddev_spinbox.value()
        mean = np.mean(gray)
        std = np.std(gray)
        
        input_min = int(max(0, mean - std_range * std))
        input_max = int(min(1023, mean + std_range * std))
        
        if input_max <= input_min:
            input_max = input_min + 1
        
        # Update sliders
        self.min_slider.setValue(input_min)
        self.max_slider.setValue(input_max)
        
        # Convert
        self.converted_image = self.apply_levels_adjustment(self.original_image, input_min, input_max)
        self.preview_viewer.display_image(self.converted_image)
        
        self.status_label.setText(f"✓ Auto-converted: Mean={mean:.0f}, Std={std:.0f}, Range=[{input_min}-{input_max}]")
        self.save_10bit_btn.setEnabled(True)
        self.save_8bit_btn.setEnabled(True)
        self.revert_btn.setEnabled(True)
        
        QMessageBox.information(self, "Success", 
            f"Auto-converted using Mean ± {std_range}σ!\n\n"
            f"Mean: {mean:.1f}\n"
            f"Std Dev: {std:.1f}\n"
            f"Input range: {input_min} - {input_max}\n"
            f"Output range: 0 - 1023")
        
    def save_image(self, as_10bit=True):
        """Save the converted image."""
        if self.converted_image is None:
            QMessageBox.warning(self, "Error", "No converted image to save.")
            return
        
        # Suggest filename
        if self.current_file_path:
            original_path = Path(self.current_file_path)
            suffix = "_10bit.png" if as_10bit else "_8bit.png"
            suggested_name = original_path.stem + suffix
            suggested_path = str(original_path.parent / suggested_name)
        else:
            suggested_path = "converted_10bit.png" if as_10bit else "converted_8bit.png"
        
        file_filter = "PNG Files (*.png);;TIFF Files (*.tiff *.tif)"
        if not as_10bit:
            file_filter += ";;JPEG Files (*.jpg *.jpeg)"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", suggested_path, file_filter
        )
        
        if file_path:
            try:
                if as_10bit:
                    # Save as 16-bit (10-bit data: 0-1023)
                    success = cv2.imwrite(file_path, self.converted_image)
                    if success:
                        self.status_label.setText(f"✓ Saved 10-bit: {Path(file_path).name}")
                        QMessageBox.information(self, "Success", 
                            f"10-bit image saved!\n\n{Path(file_path).name}\n\nData range: 0-1023")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to save image.")
                else:
                    # Convert to 8-bit for display
                    img_8bit = (self.converted_image / 1023.0 * 255.0).astype(np.uint8)
                    success = cv2.imwrite(file_path, img_8bit)
                    if success:
                        self.status_label.setText(f"✓ Saved 8-bit: {Path(file_path).name}")
                        QMessageBox.information(self, "Success", 
                            f"8-bit image saved!\n\n{Path(file_path).name}\n\nLooks as shown on screen")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to save image.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
                
    def batch_convert(self):
        """Batch convert all images in a folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder for Batch Conversion")
        
        if not folder_path:
            return
        
        std_range = self.stddev_spinbox.value()
        
        # Find all image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(Path(folder_path).glob(ext))
        
        if not image_files:
            QMessageBox.warning(self, "Error", "No image files found in folder.")
            return
        
        # Confirm
        reply = QMessageBox.question(
            self, "Batch Convert",
            f"Convert {len(image_files)} images to 10-bit?\n\n"
            f"Method: Mean ± {std_range}σ\n"
            f"Original files will not be modified.\n"
            f"New files will be saved with '_10bit' suffix.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Process images
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(image_files))
        success_count = 0
        
        for i, img_path in enumerate(image_files):
            try:
                # Load image
                if any(ord(char) > 127 for char in str(img_path)):
                    with open(str(img_path), 'rb') as f:
                        image_data = f.read()
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                else:
                    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                
                if image is None:
                    continue
                
                # Calculate Mean ± Std Dev
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                if gray.dtype == np.uint8:
                    gray = (gray.astype(np.float32) * (1023.0 / 255.0)).astype(np.uint16)
                
                mean = np.mean(gray)
                std = np.std(gray)
                
                input_min = int(max(0, mean - std_range * std))
                input_max = int(min(1023, mean + std_range * std))
                
                if input_max <= input_min:
                    input_max = input_min + 1
                
                # Convert
                converted = self.apply_levels_adjustment(image, input_min, input_max)
                
                # Save
                output_path = img_path.parent / (img_path.stem + "_10bit.png")
                cv2.imwrite(str(output_path), converted)
                
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()
        
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"✓ Batch complete: {success_count}/{len(image_files)} images")
        
        QMessageBox.information(self, "Batch Complete",
            f"Successfully converted {success_count} out of {len(image_files)} images!")
    
    def batch_convert_recursive(self):
        """Batch convert all images in folder and all subfolders."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Root Folder for Recursive Conversion")
        
        if not folder_path:
            return
        
        std_range = self.stddev_spinbox.value()
        
        # Find all image files recursively
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        root_path = Path(folder_path)
        
        for ext in extensions:
            # Use rglob for recursive search
            image_files.extend(root_path.rglob(ext))
            # Also search uppercase extensions
            image_files.extend(root_path.rglob(ext.upper()))
        
        if not image_files:
            QMessageBox.warning(self, "Error", "No image files found in folder tree.")
            return
        
        # Count files per folder for statistics
        folders = set(f.parent for f in image_files)
        
        # Ask for bit depth
        bit_depth_msg = QMessageBox()
        bit_depth_msg.setWindowTitle("Select Output Bit Depth")
        bit_depth_msg.setText(f"Found {len(image_files)} images in {len(folders)} folders.\n\nSelect output format:")
        bit_depth_msg.addButton("10-bit (uint16)", QMessageBox.ButtonRole.YesRole)
        bit_depth_msg.addButton("8-bit (visual)", QMessageBox.ButtonRole.NoRole)
        bit_depth_msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        
        bit_depth_choice = bit_depth_msg.exec()
        
        if bit_depth_choice == 2:  # Cancel
            return
        
        save_as_10bit = (bit_depth_choice == 0)
        suffix = "_10bit" if save_as_10bit else "_8bit"
        
        # Ask for output location
        reply = QMessageBox.question(
            self, "Output Location",
            f"Where to save converted images?\n\n"
            f"• Yes = Same folder as original (adds '{suffix}' suffix)\n"
            f"• No = Choose output folder",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Cancel:
            return
        
        save_in_place = (reply == QMessageBox.StandardButton.Yes)
        output_root = None
        
        if not save_in_place:
            output_root = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if not output_root:
                return
            output_root = Path(output_root)
        
        # Confirm
        bit_depth_str = "10-bit (0-1023 in uint16)" if save_as_10bit else "8-bit (0-255, visual)"
        confirm_msg = f"Convert {len(image_files)} images recursively?\n\n"
        confirm_msg += f"Method: Mean ± {std_range}σ\n"
        confirm_msg += f"Output format: {bit_depth_str}\n"
        confirm_msg += f"Folders to process: {len(folders)}\n"
        if save_in_place:
            confirm_msg += f"Output: Same folders (adds '{suffix}' suffix)\n"
        else:
            confirm_msg += f"Output: {output_root}\n"
        confirm_msg += f"\nOriginal files will not be modified."
        
        reply = QMessageBox.question(
            self, "Confirm Batch Conversion",
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Process images
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(image_files))
        success_count = 0
        error_count = 0
        error_files = []
        
        for i, img_path in enumerate(image_files):
            try:
                # Update status
                relative_path = img_path.relative_to(root_path)
                self.status_label.setText(f"Processing ({i+1}/{len(image_files)}): {relative_path}")
                
                # Load image
                if any(ord(char) > 127 for char in str(img_path)):
                    with open(str(img_path), 'rb') as f:
                        image_data = f.read()
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                else:
                    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                
                if image is None:
                    error_count += 1
                    error_files.append(str(relative_path))
                    continue
                
                # Calculate Mean ± Std Dev
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                if gray.dtype == np.uint8:
                    gray = (gray.astype(np.float32) * (1023.0 / 255.0)).astype(np.uint16)
                
                mean = np.mean(gray)
                std = np.std(gray)
                
                input_min = int(max(0, mean - std_range * std))
                input_max = int(min(1023, mean + std_range * std))
                
                if input_max <= input_min:
                    input_max = input_min + 1
                
                # Convert
                converted = self.apply_levels_adjustment(image, input_min, input_max)
                
                # Convert to 8-bit if needed
                if not save_as_10bit:
                    converted = (converted / 1023.0 * 255.0).astype(np.uint8)
                
                # Determine output path
                if save_in_place:
                    # Save in same folder with suffix
                    output_path = img_path.parent / (img_path.stem + f"{suffix}.png")
                else:
                    # Recreate folder structure in output folder
                    relative_folder = img_path.parent.relative_to(root_path)
                    output_folder = output_root / relative_folder
                    output_folder.mkdir(parents=True, exist_ok=True)
                    output_path = output_folder / (img_path.stem + f"{suffix}.png")
                
                # Save
                success = cv2.imwrite(str(output_path), converted)
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    error_files.append(f"{relative_path} (save failed)")
                
            except Exception as e:
                error_count += 1
                error_files.append(f"{relative_path} ({str(e)})")
                print(f"Error processing {img_path}: {e}")
            
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()
        
        self.progress_bar.setVisible(False)
        
        # Show results
        format_str = "10-bit" if save_as_10bit else "8-bit"
        result_msg = f"Batch conversion complete!\n\n"
        result_msg += f"Format: {format_str}\n"
        result_msg += f"Method: Mean ± {std_range}σ\n\n"
        result_msg += f"✓ Successfully converted: {success_count}\n"
        result_msg += f"✗ Failed: {error_count}\n"
        result_msg += f"Total processed: {len(image_files)}\n"
        result_msg += f"Folders processed: {len(folders)}"
        
        if error_files and len(error_files) <= 10:
            result_msg += f"\n\nFailed files:\n" + "\n".join(error_files[:10])
        elif error_files:
            result_msg += f"\n\nFailed files: {len(error_files)} (too many to list)"
        
        self.status_label.setText(f"✓ Batch complete: {success_count}/{len(image_files)} images ({format_str})")
        
        QMessageBox.information(self, "Batch Conversion Complete", result_msg)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ImageConverter10Bit()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()