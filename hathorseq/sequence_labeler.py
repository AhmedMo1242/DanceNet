import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.animation as animation    
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QSlider, QFileDialog, QComboBox,
                             QStatusBar, QFrame, QCheckBox, QGroupBox, QMessageBox, QSplitter,
                             QListWidget, QListWidgetItem, QToolButton, QRadioButton, QButtonGroup, QDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIntValidator

from data_loader import load_mocap_data     
from skeleton_visualization import generate_edge_indices, get_line_segments, put_lines
from skeleton_def import point_labels

class SequenceLabelerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Motion Capture Sequence Labeler")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.data_filtered = None  # Filtered data (for display)
        self.data_unfiltered = None  # Unfiltered data (for export)
        self.current_sequence = None
        self.original_sequence = None  # Store original unfiltered data
        self.current_frame = 0
        self.start_frame = 0
        self.sequence_length = 0
        self.labels = []
        self.selected_label_index = -1
        self.animation_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create a splitter to separate visualization and controls
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)
        
        # Set up the visualization area
        self.setup_visualization_area()
        
        # Set up the controls area
        self.setup_controls_area()
        
        # Set up status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load a motion capture file to begin.")
        
        # Initialize the edge indices for visualization
        self.edge_indices = generate_edge_indices()

    def setup_visualization_area(self):
        """Set up the 3D visualization area"""
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        
        # Create a figure and canvas for the 3D plot
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        
        # Create a 3D axis for visualization
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Motion Capture Visualization')
        
        # Add the navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        self.splitter.addWidget(viz_container)

    def setup_controls_area(self):
        """Set up the control panel area"""
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        
        # File operations section
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Mocap File")
        self.load_button.clicked.connect(self.load_file)
        
        file_layout.addWidget(self.load_button)
        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)
        
        # Navigation section
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout()
        
        # Frame slider row
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame:")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Will be updated when data is loaded
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        
        self.frame_entry = QLineEdit("0")
        self.frame_entry.setFixedWidth(60)
        self.frame_entry.returnPressed.connect(self.frame_entry_changed)
        
        self.total_frames_label = QLabel("/ 0")
        
        slider_layout.addWidget(self.frame_label)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_entry)
        slider_layout.addWidget(self.total_frames_label)
        nav_layout.addLayout(slider_layout)
        
        # Navigation buttons row
        nav_buttons_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.prev_frame)
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_play)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.next_frame)
        
        self.playback_speed = QComboBox()
        self.playback_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.playback_speed.setCurrentIndex(2)  # Default to 1x
        self.playback_speed.currentIndexChanged.connect(self.update_playback_speed)
        
        nav_buttons_layout.addWidget(self.prev_button)
        nav_buttons_layout.addWidget(self.play_button)
        nav_buttons_layout.addWidget(self.next_button)
        nav_buttons_layout.addWidget(QLabel("Speed:"))
        nav_buttons_layout.addWidget(self.playback_speed)
        nav_layout.addLayout(nav_buttons_layout)
        
        nav_group.setLayout(nav_layout)
        controls_layout.addWidget(nav_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        # Add radio buttons for visualization style
        style_layout = QHBoxLayout()
        self.viz_style_group = QButtonGroup(self)
        
        self.both_style_radio = QRadioButton("Lines & Dots")
        self.both_style_radio.setChecked(True)
        self.viz_style_group.addButton(self.both_style_radio)
        style_layout.addWidget(self.both_style_radio)
        
        self.lines_style_radio = QRadioButton("Lines Only")
        self.viz_style_group.addButton(self.lines_style_radio)
        style_layout.addWidget(self.lines_style_radio)
        
        self.dots_style_radio = QRadioButton("Dots Only")
        self.viz_style_group.addButton(self.dots_style_radio)
        style_layout.addWidget(self.dots_style_radio)
        
        # Connect the buttons to update function
        self.both_style_radio.toggled.connect(self.update_display)
        self.lines_style_radio.toggled.connect(self.update_display)
        self.dots_style_radio.toggled.connect(self.update_display)
        
        display_layout.addLayout(style_layout)
        
        # Existing display options
        options_layout = QHBoxLayout()
        
        self.show_axes_check = QCheckBox("Show Axes/Grid")
        self.show_axes_check.setChecked(True)
        self.show_axes_check.stateChanged.connect(self.update_display)
        
        options_layout.addWidget(self.show_axes_check)
        display_layout.addLayout(options_layout)
        
        display_group.setLayout(display_layout)
        controls_layout.addWidget(display_group)
        
        # Sequence labeling section
        label_group = QGroupBox("Sequence Labeling")
        label_layout = QVBoxLayout()
        
        # Mode selection for defining sequence
        mode_layout = QHBoxLayout()
        self.seq_mode_group = QButtonGroup(self)
        
        self.length_mode_radio = QRadioButton("Define by Length")
        self.length_mode_radio.setChecked(True)
        self.seq_mode_group.addButton(self.length_mode_radio)
        mode_layout.addWidget(self.length_mode_radio)
        
        self.endpoint_mode_radio = QRadioButton("Define by End Frame")
        self.seq_mode_group.addButton(self.endpoint_mode_radio)
        mode_layout.addWidget(self.endpoint_mode_radio)
        
        # Connect mode toggle
        self.length_mode_radio.toggled.connect(self.toggle_sequence_mode)
        
        label_layout.addLayout(mode_layout)
        
        # Start frame and length/end frame row
        range_layout = QHBoxLayout()
        
        range_layout.addWidget(QLabel("Start Frame:"))
        self.start_frame_entry = QLineEdit("0")
        self.start_frame_entry.setFixedWidth(60)
        self.start_frame_entry.textChanged.connect(self.update_sequence_fields)
        range_layout.addWidget(self.start_frame_entry)
        
        self.set_start_button = QPushButton("Set to Current")
        self.set_start_button.clicked.connect(self.set_start_to_current)
        range_layout.addWidget(self.set_start_button)
        
        range_layout.addSpacing(10)
        
        # Sequence length field (active in length mode)
        self.length_widget = QWidget()
        length_layout = QHBoxLayout(self.length_widget)
        length_layout.setContentsMargins(0, 0, 0, 0)
        
        length_layout.addWidget(QLabel("Sequence Length:"))
        self.seq_length_entry = QLineEdit("10")
        self.seq_length_entry.setFixedWidth(60)
        self.seq_length_entry.textChanged.connect(self.update_sequence_fields)
        length_layout.addWidget(self.seq_length_entry)
        
        range_layout.addWidget(self.length_widget)
        
        # End frame field/label (editable in endpoint mode, label in length mode)
        self.end_widget = QWidget()
        end_layout = QHBoxLayout(self.end_widget)
        end_layout.setContentsMargins(0, 0, 0, 0)
        
        end_layout.addWidget(QLabel("End Frame:"))
        
        # Stacked widgets for end frame (label in length mode, entry in endpoint mode)
        self.end_frame_label = QLabel("9")
        self.end_frame_label.setFixedWidth(60)
        
        self.end_frame_entry = QLineEdit("9")
        self.end_frame_entry.setFixedWidth(60)
        self.end_frame_entry.textChanged.connect(self.update_sequence_fields)
        
        # Initially show label (length mode), hide entry
        self.end_frame_entry.hide()
        end_layout.addWidget(self.end_frame_label)
        end_layout.addWidget(self.end_frame_entry)
        
        self.set_end_button = QPushButton("Set to Current")
        self.set_end_button.clicked.connect(self.set_end_to_current)
        end_layout.addWidget(self.set_end_button)
        
        range_layout.addWidget(self.end_widget)
        
        label_layout.addLayout(range_layout)
        
        # Label row
        label_row_layout = QHBoxLayout()
        
        label_row_layout.addWidget(QLabel("Label:"))
        self.label_entry = QLineEdit()
        label_row_layout.addWidget(self.label_entry)
        
        self.add_label_button = QPushButton("Add Labeled Sequence")
        self.add_label_button.clicked.connect(self.add_labeled_sequence)
        label_row_layout.addWidget(self.add_label_button)
        
        label_layout.addLayout(label_row_layout)
        
        label_group.setLayout(label_layout)
        controls_layout.addWidget(label_group)
        
        # Labeled sequences list and management
        sequences_group = QGroupBox("Labeled Sequences")
        sequences_layout = QVBoxLayout()
        
        # List of labeled sequences
        self.sequences_list = QListWidget()
        self.sequences_list.itemClicked.connect(self.select_labeled_sequence)
        sequences_layout.addWidget(self.sequences_list)
        
        # Edit/Delete buttons
        buttons_layout = QHBoxLayout()
        
        self.edit_sequence_button = QPushButton("Edit Selected")
        self.edit_sequence_button.clicked.connect(self.edit_labeled_sequence)
        self.edit_sequence_button.setEnabled(False)
        buttons_layout.addWidget(self.edit_sequence_button)
        
        self.delete_sequence_button = QPushButton("Delete Selected")
        self.delete_sequence_button.clicked.connect(self.delete_labeled_sequence)
        self.delete_sequence_button.setEnabled(False)
        buttons_layout.addWidget(self.delete_sequence_button)
        
        # Add Export to NPY button
        self.export_npy_button = QPushButton("Export to NPY Dataset")
        self.export_npy_button.clicked.connect(self.export_to_npy)
        self.export_npy_button.setEnabled(False)
        buttons_layout.addWidget(self.export_npy_button)
        
        sequences_layout.addLayout(buttons_layout)
        
        sequences_group.setLayout(sequences_layout)
        controls_layout.addWidget(sequences_group)
        
        self.splitter.addWidget(controls_container)

    def toggle_sequence_mode(self):
        """Toggle between length-based and endpoint-based sequence definition"""
        if self.length_mode_radio.isChecked():
            # Length mode - show length field, use end frame label
            self.seq_length_entry.show()
            self.end_frame_entry.hide()
            self.end_frame_label.show()
            
            # Update the end frame label based on current start + length
            self.update_sequence_fields()
        else:
            # Endpoint mode - hide length field, use end frame entry
            self.seq_length_entry.hide()
            self.end_frame_label.hide()
            self.end_frame_entry.show()
            
            # Initialize end frame entry with calculated value from current settings
            try:
                start = int(self.start_frame_entry.text())
                length = int(self.seq_length_entry.text())
                end = start + length - 1
                self.end_frame_entry.setText(str(end))
            except ValueError:
                pass
                
            # Update the length based on current start + end
            self.update_sequence_fields()

    def update_sequence_fields(self):
        """Update sequence fields based on the selected mode"""
        if self.length_mode_radio.isChecked():
            # Length mode: calculate end frame from start + length
            self.update_end_frame()
        else:
            # Endpoint mode: calculate length from start + end
            self.update_sequence_length()

    def update_end_frame(self):
        """Update the end frame based on start frame and sequence length"""
        try:
            start = int(self.start_frame_entry.text())
            length = int(self.seq_length_entry.text())
            
            if length < 1:
                length = 1
                self.seq_length_entry.setText("1")
            
            # Calculate end frame
            end = start + length - 1
            
            # Check if end frame exceeds maximum
            if self.current_sequence is not None and end >= self.frame_slider.maximum() + 1:
                end = self.frame_slider.maximum()
                length = end - start + 1
                self.seq_length_entry.setText(str(length))
            
            self.end_frame_label.setText(str(end))
            
            # Also update the entry field (even if hidden)
            self.end_frame_entry.setText(str(end))
        except ValueError:
            # If input is not valid, don't update
            pass

    def update_sequence_length(self):
        """Update the sequence length based on start and end frames"""
        try:
            start = int(self.start_frame_entry.text())
            end = int(self.end_frame_entry.text())
            
            # Make sure end is not less than start
            if end < start:
                end = start
                self.end_frame_entry.setText(str(end))
            
            # Check if end frame exceeds maximum
            if self.current_sequence is not None and end >= self.frame_slider.maximum() + 1:
                end = self.frame_slider.maximum()
                self.end_frame_entry.setText(str(end))
            
            # Calculate length
            length = end - start + 1
            self.seq_length_entry.setText(str(length))
        except ValueError:
            # If input is not valid, don't update
            pass

    def set_start_to_current(self):
        """Set the start frame to the current frame"""
        self.start_frame = self.current_frame
        self.start_frame_entry.setText(str(self.start_frame))
        self.update_sequence_fields()

    def set_end_to_current(self):
        """Set the end frame to the current frame"""
        self.end_frame = self.current_frame
        
        if self.length_mode_radio.isChecked():
            # In length mode, update the length
            try:
                start = int(self.start_frame_entry.text())
                end = self.end_frame
                
                # Make sure end is not less than start
                if end < start:
                
                    end = start
                
                # Calculate length
                length = end - start + 1
                self.seq_length_entry.setText(str(length))
            except ValueError:
                pass
        else:
            # In endpoint mode, directly update the end frame entry
            self.end_frame_entry.setText(str(self.end_frame))
        
        self.update_sequence_fields()

    def load_file(self):
        """Load a motion capture file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Mocap File", "./examples", "NPY Files (*.npy)"
        )
        
        if file_path:
            try:
                self.statusBar.showMessage(f"Loading file: {file_path}")
                
                # Load two versions of the data:
                # 1. Filtered version (for display) with joints 26 and 53 excluded
                self.data_filtered = load_mocap_data(file_path, exclude_joints=True)
                self.current_sequence = self.data_filtered['raw']
                
                # 2. Unfiltered version (for export) with all joints preserved
                self.data_unfiltered = load_mocap_data(file_path, exclude_joints=False) 
                self.original_sequence = self.data_unfiltered.get('original')
                
                # Update UI elements
                total_frames = self.current_sequence.shape[0]
                self.frame_slider.setMaximum(total_frames - 1)
                self.total_frames_label.setText(f"/ {total_frames-1}")
                
                # Reset frame position
                self.current_frame = 0
                self.frame_slider.setValue(0)
                self.frame_entry.setText("0")
                
                # Set default sequence length to 10% of total frames
                default_length = min(60, int(total_frames * 0.1))
                self.seq_length_entry.setText(str(default_length))
                self.update_end_frame()
                
                # Update visualization
                self.update_visualization()
                
                # Show joint count info
                filtered_joint_count = self.current_sequence.shape[1]
                unfiltered_joint_count = self.original_sequence.shape[1] if self.original_sequence is not None else "unknown"
                
                self.statusBar.showMessage(
                    f"Loaded {os.path.basename(file_path)}: {total_frames} frames, "
                    f"displaying {filtered_joint_count} joints (full data has {unfiltered_joint_count} joints)"
                )
                
                # Enable the export NPY button if data is loaded
                self.export_npy_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
                self.statusBar.showMessage("Error loading file")

    def slider_changed(self):
        """Handle changes to the frame slider"""
        self.current_frame = self.frame_slider.value()
        self.frame_entry.setText(str(self.current_frame))
        self.update_visualization()

    def frame_entry_changed(self):
        """Handle manual entry of frame number"""
        try:
            new_frame = int(self.frame_entry.text())
            if 0 <= new_frame < self.frame_slider.maximum() + 1:
                self.current_frame = new_frame
                self.frame_slider.setValue(new_frame)
                self.update_visualization()
            else:
                self.frame_entry.setText(str(self.current_frame))
        except ValueError:
            self.frame_entry.setText(str(self.current_frame))

    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)
            self.frame_entry.setText(str(self.current_frame))
            self.update_visualization()

    def next_frame(self):
        """Go to next frame"""
        if self.current_frame < self.frame_slider.maximum():
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.frame_entry.setText(str(self.current_frame))
            self.update_visualization()
        elif self.animation_playing:
            # If playing and we reach the end, stop
            self.toggle_play()

    def toggle_play(self):
        """Toggle play/pause of the animation"""
        if not self.animation_playing:
            self.animation_playing = True
            self.play_button.setText("⏸ Pause")
            self.update_playback_speed()  # Start the timer
        else:
            self.animation_playing = False
            self.play_button.setText("▶ Play")
            self.play_timer.stop()

    def update_playback_speed(self):
        """Update the playback speed based on the dropdown selection"""
        if not self.animation_playing:
            return
            
        speed_text = self.playback_speed.currentText()
        speed_factor = float(speed_text.strip('x'))
        
        # Base interval of 40ms (25 fps) adjusted by speed factor
        interval = int(40 / speed_factor)
        
        self.play_timer.stop()
        self.play_timer.start(interval)

    def add_labeled_sequence(self):
        """Add a labeled sequence to the list"""
        try:
            start = int(self.start_frame_entry.text())
            
            if self.length_mode_radio.isChecked():
                length = int(self.seq_length_entry.text())
                end = start + length - 1
            else:
                end = int(self.end_frame_entry.text())
                length = end - start + 1
            
            label = self.label_entry.text().strip()
            
            if not label:
                QMessageBox.warning(self, "Warning", "Please enter a label for the sequence.")
                return
                
            if self.current_sequence is None:
                QMessageBox.warning(self, "Warning", "No motion capture data loaded.")
                return
                
            if end >= self.frame_slider.maximum() + 1:
                QMessageBox.warning(self, "Warning", f"End frame exceeds maximum frame ({self.frame_slider.maximum()}).")
                return
            
            # Add to labels list
            entry = {
                'start': start,
                'end': end,
                'length': length,
                'label': label
            }
            
            if self.selected_label_index >= 0 and self.selected_label_index < len(self.labels):
                # Replace the selected entry
                self.labels[self.selected_label_index] = entry
                self.statusBar.showMessage(f"Updated labeled sequence: {label} ({start}-{end})")
            else:
                # Add a new entry
                self.labels.append(entry)
                self.statusBar.showMessage(f"Added labeled sequence: {label} ({start}-{end})")
            
            # Update the list display
            self.update_sequences_list()
            
            # Clear the label entry and reset selection for the next label
            self.label_entry.clear()
            self.selected_label_index = -1
            self.edit_sequence_button.setEnabled(False)
            self.delete_sequence_button.setEnabled(False)
            
            # Enable export NPY button if we have sequences
            if self.labels:
                self.export_npy_button.setEnabled(True)
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid frame numbers.")

    def update_sequences_list(self):
        """Update the list widget with all labeled sequences"""
        self.sequences_list.clear()
        
        for i, entry in enumerate(self.labels):
            item_text = f"{entry['label']} (Frames {entry['start']}-{entry['end']}, Length: {entry['length']})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store the index
            self.sequences_list.addItem(item)

    def select_labeled_sequence(self, item):
        """Handle selection of a labeled sequence from the list"""
        index = item.data(Qt.UserRole)
        if 0 <= index < len(self.labels):
            self.selected_label_index = index
            self.edit_sequence_button.setEnabled(True)
            self.delete_sequence_button.setEnabled(True)
            
            # Jump to the start frame of the selected sequence
            entry = self.labels[index]
            self.frame_slider.setValue(entry['start'])
            
            self.statusBar.showMessage(f"Selected sequence: {entry['label']}")

    def edit_labeled_sequence(self):
        """Load the selected sequence's details into the editing fields"""
        if 0 <= self.selected_label_index < len(self.labels):
            entry = self.labels[self.selected_label_index]
            
            # Set the fields
            self.start_frame_entry.setText(str(entry['start']))
            self.seq_length_entry.setText(str(entry['length']))
            self.label_entry.setText(entry['label'])
            
            self.statusBar.showMessage(f"Editing sequence: {entry['label']}")

    def delete_labeled_sequence(self):
        """Delete the selected labeled sequence"""
        if 0 <= self.selected_label_index < len(self.labels):
            entry = self.labels[self.selected_label_index]
            
            # Confirm deletion
            reply = QMessageBox.question(
                self, 
                "Confirm Deletion",
                f"Are you sure you want to delete the sequence '{entry['label']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove from the list
                self.labels.pop(self.selected_label_index)
                self.update_sequences_list()
                
                # Reset selection
                self.selected_label_index = -1
                self.edit_sequence_button.setEnabled(False)
                self.delete_sequence_button.setEnabled(False)
                
                self.statusBar.showMessage(f"Deleted sequence: {entry['label']}")

    def export_to_npy(self):
        """Export labeled sequences to NPY dataset with fixed sequence length"""
        if self.current_sequence is None:
            QMessageBox.warning(self, "Warning", "No motion capture data loaded.")
            return
        
        # Ask user for export parameters
        export_dialog = QDialog(self)
        export_dialog.setWindowTitle("Export to NPY Dataset")
        export_dialog.setMinimumWidth(400)
        
        dialog_layout = QVBoxLayout(export_dialog)
        
        # Sequence length input
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Sequence Length (frames):"))
        sequence_length_input = QLineEdit("50")
        sequence_length_input.setValidator(QIntValidator(1, 999999))
        length_layout.addWidget(sequence_length_input)
        dialog_layout.addLayout(length_layout)
        
        # Overlap input
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap between sequences:"))
        overlap_input = QLineEdit("0")
        overlap_input.setValidator(QIntValidator(0, 999999))
        overlap_layout.addWidget(overlap_input)
        dialog_layout.addLayout(overlap_layout)
        
        # Include unlabeled segments checkbox
        include_unlabeled = QCheckBox("Include unlabeled segments")
        include_unlabeled.setChecked(True)
        dialog_layout.addWidget(include_unlabeled)
        
        # Default label for unlabeled segments
        unlabeled_layout = QHBoxLayout()
        unlabeled_layout.addWidget(QLabel("Default label for unlabeled:"))
        default_label_input = QLineEdit("")
        unlabeled_layout.addWidget(default_label_input)
        dialog_layout.addLayout(unlabeled_layout)
        
        # Add option to include or exclude specific joints
        joints_layout = QHBoxLayout()
        include_all_joints = QCheckBox("Include all joints (including 26 and 53)")
        include_all_joints.setChecked(True)
        joints_layout.addWidget(include_all_joints)
        dialog_layout.addLayout(joints_layout)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(export_dialog.reject)
        
        export_button = QPushButton("Export")
        export_button.clicked.connect(export_dialog.accept)
        export_button.setDefault(True)
        
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(export_button)
        dialog_layout.addLayout(buttons_layout)
        
        # Show the dialog and wait for result
        if not export_dialog.exec_():
            return  # User canceled
        
        # Get parameters from dialog
        try:
            seq_length = int(sequence_length_input.text())
            overlap = int(overlap_input.text())
            include_unlabeled_flag = include_unlabeled.isChecked()
            default_label = default_label_input.text()
            use_all_joints = include_all_joints.isChecked()
            
            if seq_length <= 0:
                raise ValueError("Sequence length must be positive")
            
            if overlap >= seq_length:
                raise ValueError("Overlap must be less than sequence length")
                
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return
        
        # Get the file path to save the NPY file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save NPY Dataset", "./output", "NPY Files (*.npy)"
        )
        
        if not file_path:
            return  # User canceled
        
        # Make sure the file has .npy extension
        if not file_path.lower().endswith('.npy'):
            file_path += '.npy'
        
        # Create the dataset
        try:
            # Choose which data to use based on user's joint preference
            if use_all_joints:
                # Use original unfiltered data with all joints
                motion_data = self.original_sequence if self.original_sequence is not None else self.current_sequence
                joint_selection = "all joints included"
            else:
                # Use filtered data (without joints 26 and 53)
                motion_data = self.current_sequence
                joint_selection = "joints 26 and 53 excluded"
                
            joint_count = motion_data.shape[1]
            
            # Create separate lists for unlabeled and labeled sequences
            unlabeled_sequences = []
            labeled_sequences = []
            
            # Calculate the step size for sliding window (accounting for overlap)
            step = seq_length - overlap
            
            # Number of sequences we'll generate
            num_sequences = (motion_data.shape[0] - seq_length) // step + 1
            
            # Process each segment
            for i in range(num_sequences):
                start_idx = i * step
                end_idx = start_idx + seq_length
                
                # Get the motion data for this segment
                segment_data = motion_data[start_idx:end_idx].copy()
                
                # Find if this segment has a label from our labeled sequences
                segment_label = default_label
                segment_has_label = False
                
                for label_entry in self.labels:
                    # Check if there's a significant overlap with this labeled sequence
                    label_start = label_entry['start']
                    label_end = label_entry['end']
                    
                    # Calculate overlap between segment and labeled sequence
                    overlap_start = max(start_idx, label_start)
                    overlap_end = min(end_idx - 1, label_end)
                    
                    if overlap_start <= overlap_end:
                        # There is an overlap, calculate percentage
                        overlap_length = overlap_end - overlap_start + 1
                        overlap_percentage = overlap_length / seq_length
                        
                        # If > 50% of the segment is covered by this label, use this label
                        if overlap_percentage > 0.5:
                            segment_label = label_entry['label']
                            segment_has_label = True
                            break
                
                # Add to the appropriate list based on whether it has a label
                if segment_has_label:
                    labeled_sequences.append((segment_data, segment_label))
                elif include_unlabeled_flag:
                    unlabeled_sequences.append((segment_data, default_label))
            
            # Combine lists - unlabeled first, then labeled
            combined_sequences = unlabeled_sequences + labeled_sequences
            
            # If no segments were included, show an error
            if not combined_sequences:
                QMessageBox.warning(self, "Warning", "No segments were extracted with the current settings.")
                return
            
            # Create a structured array to save
            # We'll save as a list of tuples (data, label) where data is a numpy array and label is a string
            np.save(file_path, combined_sequences)
            
            # Show summary
            QMessageBox.information(
                self, 
                "Export Complete", 
                f"Dataset exported successfully!\n\n"
                f"Total segments: {len(combined_sequences)}\n"
                f"Unlabeled segments: {len(unlabeled_sequences)}\n"
                f"Labeled segments: {len(labeled_sequences)}\n"
                f"Joints per frame: {joint_count} ({joint_selection})\n"
                f"Segment length: {seq_length} frames\n"
                f"Overlap: {overlap} frames\n"
                f"Saved to: {file_path}"
            )
            
            self.statusBar.showMessage(
                f"Exported {len(combined_sequences)} segments with {joint_count} joints each to {file_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            self.statusBar.showMessage("Error exporting dataset")

    def update_display(self):
        """Update display options and refresh visualization"""
        self.update_visualization()

    def update_visualization(self):
        """Update the 3D visualization based on current frame and display options"""
        if self.current_sequence is None:
            return
            
        # Clear the current visualization
        self.ax.clear()
        
        # Get the current frame data
        frame_data = self.current_sequence[self.current_frame]
        
        # Determine visualization style
        show_lines = self.both_style_radio.isChecked() or self.lines_style_radio.isChecked()
        show_dots = self.both_style_radio.isChecked() or self.dots_style_radio.isChecked()
        
        # Setup axis properties
        if self.show_axes_check.isChecked():
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title(f'Frame {self.current_frame}')
            self.ax.grid(True)
        else:
            self.ax.set_axis_off()
            self.ax.grid(False)
        
        # Calculate appropriate axis limits based on the data
        # Use wider limits to ensure the entire figure fits
        data_range = 1.0  # Default for normalized data
        
        # Dynamically determine axis limits based on data
        x_min, x_max = frame_data[:, 0].min(), frame_data[:, 0].max()
        y_min, y_max = frame_data[:, 1].min(), frame_data[:, 1].max()
        z_min, z_max = frame_data[:, 2].min(), frame_data[:, 2].max()
        
        # Add margin to ensure figure fits within view
        margin = 0.2
        x_range = max(data_range, x_max - x_min) * (1 + margin)
        y_range = max(data_range, y_max - y_min) * (1 + margin)
        z_range = max(data_range, z_max - z_min) * (1 + margin)
        
        # Center the figure on the axes
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        self.ax.set_zlim(max(0, z_min - z_range*0.1), z_center + z_range/2)
        
        # Plot markers (dots)
        if show_dots:
            self.ax.scatter(
                frame_data[:, 0],
                frame_data[:, 1],
                frame_data[:, 2],
                c='b',
                s=20,
                alpha=0.8
            )
        
        # Plot skeleton
        if show_lines:
            # Create a dummy sequence with just the current frame
            single_frame_seq = frame_data.reshape(1, *frame_data.shape)
            
            # Get line segments
            xline, _ = get_line_segments(single_frame_seq, self.edge_indices)
            
            # Add skeleton lines
            skeleton_idxs = self.edge_indices['skeleton']
            for i in range(len(skeleton_idxs)):
                self.ax.plot(
                    np.linspace(xline[0, i, 0, 0], xline[0, i, 0, 1], 2),
                    np.linspace(xline[0, i, 1, 0], xline[0, i, 1, 1], 2),
                    np.linspace(xline[0, i, 2, 0], xline[0, i, 2, 1], 2),
                    color='r',
                    alpha=0.8,
                    lw=2
                )
        
        # Highlight if current frame is within a selected range
        try:
            # Check if we're in any of the labeled sequences
            in_sequence = False
            sequence_label = None
            
            for entry in self.labels:
                if entry['start'] <= self.current_frame <= entry['end']:
                    in_sequence = True
                    sequence_label = entry['label']
                    break
            
            if in_sequence:
                self.ax.text2D(0.05, 0.95, f"In Sequence: {sequence_label}", 
                              transform=self.ax.transAxes, color='green')
                
            # Check if we're in the currently defined range
            start = int(self.start_frame_entry.text())
            length = int(self.seq_length_entry.text())
            end = start + length - 1
            
            if start <= self.current_frame <= end:
                self.ax.text2D(0.05, 0.90, "In Current Selection", 
                              transform=self.ax.transAxes, color='blue')
                
        except ValueError:
            pass
        
        # Refresh the canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SequenceLabelerApp()
    window.show()
    sys.exit(app.exec_())
