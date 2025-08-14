import json  # for retrieving video original audio fs
import os
import subprocess  # for retrieving video original audio fs
import sys

import cv2  # via opencv-python
import numpy as np
import pandas as pd  # for exporting the trial times
import pyqtgraph as pg  # for graphing the audio
from PySide6.QtCore import QThread, Signal, Slot, Qt, QEvent, QCoreApplication, QMetaObject, QSize, QProcess
from PySide6.QtGui import QColor, QBrush, QPixmap, QImage, QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QGridLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy,
                               QSlider, QWidget, QLineEdit, QTableWidget, QHeaderView, QTableWidgetItem,
                               QAbstractItemView, QStatusBar, QSpinBox, QAbstractSpinBox, QFrame, QMessageBox,
                               QProgressBar, QFileDialog, QDialog, QVBoxLayout)

# import re  # parsing ffmpeg output for progress

__version__ = '2.5'

# Note: to build the exe, pyinstaller is required. Once installed, go to Windows terminal, navigate to folder with
# target script, and enter:
# > python -m PyInstaller main_trim.py -n TrialTrim
# where -n specifies the resulting exe name


def get_seconds_from_time(time_str):
    time = time_str.split(":")
    if len(time) > 2:
        # timestamp includes hours
        nSec = float(time[0]) * 60 * 60 + float(time[1]) * 60 + float(time[2])
    else:
        # timestamp is just minutes and seconds
        nSec = float(time[0]) * 60 + float(time[1])

    return nSec


def get_time_from_seconds(seconds):
    nMin, nSec = divmod(seconds, 60)
    nHour, nMin = divmod(nMin, 60)
    # nMicro = round((nSec % 1)*10**6)
    timeStr = f"{round(nHour)}:{round(nMin):02}:{nSec:07.4f}"  # seconds with 07 for len(nSec)
    # ts = datetime.time(hour=int(nHour), minute=int(nMin), second=int(nSec), microsecond=int(nMicro))
    # timeStr = ts.strftime('%H:%M:%S.%f')
    return timeStr


def check_ffmpeg_installed():
    """
    Checks if FFmpeg is installed and accessible by trying to run 'ffmpeg -version'.
    Returns True if FFmpeg is found, False otherwise.
    """
    try:
        # Run the ffmpeg -version command
        # stdout=subprocess.PIPE captures the output
        # stderr=subprocess.PIPE captures error messages
        # check=True raises CalledProcessError if the command returns a non-zero exit code
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        message = "FFmpeg is installed and accessible."
        return True, message
    except FileNotFoundError:
        message = "FFmpeg is not found in the system's PATH."
        return False, message
    except subprocess.CalledProcessError as e:
        message = f"FFmpeg command failed with error: {e} Stderr: {e.stderr.decode()}"
        return False, message
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        return False, message


# def get_audio_fs(video_path):
#     """Moviepy does not properly return the audio sample rate - instead it returns the resampled rate, which
#     usually defaults to 44.1kHz.  This process gets the actual sample rate, although it is somewhat complex"""
#     result = subprocess.run(
#         [
#             "ffprobe", "-v", "error",
#             "-select_streams", "a:0",
#             "-show_entries", "stream=sample_rate",
#             "-of", "json",
#             video_path
#         ],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True
#     )
#     info = json.loads(result.stdout)
#     return int(info['streams'][0]['sample_rate'])


def extract_audio(video_path):
    """Extract the audio from the video file without moviepy, thereby reducing the package requirement for the script"""

    # Get stream info
    mapping = {
        's16': np.int16,
        's32': np.int32,
        'flt': np.float32,
        'dbl': np.float64,
        'f32': np.float32
    }

    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,sample_fmt",
        "-of", "json", video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    info = json.loads(result.stdout)
    stream = info['streams'][0]
    nChan = stream['channels']
    fs = int(stream['sample_rate'])
    fmt = stream['sample_fmt']
    if fmt == 'fltp':
        fmt = 'f32'

    fmtdtype = mapping[fmt]

    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", fmt + "le",
        "-acodec", f"pcm_{fmt}le",
        "-vn", "-hide_banner", "-loglevel", "error",
        "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read()
    audio = np.frombuffer(raw, dtype=fmtdtype)

    # Reshape to (n_samples, n_channels)
    audio = audio.reshape((-1, nChan))
    return audio, fs


class VideoThread(QThread):
    # How to display opencv video in pyqt apps: https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
    change_pixmap_signal = Signal(np.ndarray, int)

    def __init__(self, cap):
        super().__init__()
        self.run_flag = False
        self.cap = cap

    def run(self):
        self.run_flag = True
        while self.run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self.run_flag = False
        self.wait()

    def close(self):
        # shut down capture system
        self.cap.release()


class ProgressDialog(QDialog):

    cancel_requested = Signal()

    def __init__(self, nClips):
        super().__init__()
        self.setWindowTitle("Splitting Progress")
        layout = QVBoxLayout(self)

        self.setFixedWidth(400)

        self.clipOverallText = QLabel()
        self.clipOverallText.setFixedHeight(24)
        self.clipOverallText.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.clipOverallText)

        self.clipOverallBar = QProgressBar()
        self.clipOverallBar.setMinimum(0)
        self.clipOverallBar.setMaximum(nClips)
        self.clipOverallBar.setValue(0)  # Initial value
        layout.addWidget(self.clipOverallBar)

        self.clipProgressText = QLabel()
        self.clipProgressText.setFixedHeight(24)
        self.clipProgressText.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.clipProgressText)

        # self.clipProgressBar = QProgressBar()
        # self.clipProgressBar.setMinimum(0)
        # self.clipProgressBar.setMaximum(100)
        # self.clipProgressBar.setValue(0)  # Initial value
        # layout.addWidget(self.clipProgressBar, 3, 0, 1, 2)

        self.cancelButton = QPushButton()
        self.cancelButton.setObjectName(u"cancelButton")
        self.cancelButton.setFixedSize(QSize(84, 24))
        self.cancelButton.setText("Cancel")
        layout.addWidget(self.cancelButton, alignment=Qt.AlignmentFlag.AlignCenter)

        self.cancelButton.clicked.connect(self.cancel)

    @Slot(str)
    def update_clip_text(self, val):
        self.clipProgressText.setText(val)

    @Slot(int)
    def update_overall_progress(self, val):
        self.clipOverallBar.setValue(val)

    @Slot(str)
    def update_overall_text(self, val):
        self.clipOverallText.setText(val)

    def cancel(self):
        self.cancel_requested.emit()
        self.close()

    def closeEvent(self, event):
        self.cancel_requested.emit()
        event.accept()


class UiMainWindow(object):

    def __init__(self):
        self.centralwidget = None
        self.progressBar = None
        self.versionLabel = None
        self.controlGridLayout = None

        self.fileLayout = None
        self.pathLabel = None
        self.loadVideoButton = None
        # self.pathText = None

        self.videoFrame = None
        self.audioFrame = None

        self.boundingGridFrame = None
        self.boundingGridLayout = None
        self.boundingLeftLabel = None
        self.boundingLeftSpinBox = None
        self.boundingRightLabel = None
        self.boundingRightSpinBox = None
        self.boundingTopLabel = None
        self.boundingTopSpinBox = None
        self.boundingBottomLabel = None
        self.boundingBottomSpinBox = None
        self.boundingBoxButton = None

        self.trialMarkerTable = None

        self.trackingLayout = None
        self.timeStartTextEdit = None
        self.timeEndLabel = None
        self.trackingSlider = None

        self.settingLayout = None
        self.playVideoButton = None
        self.setStartButton = None
        self.setEndButton = None

        self.addTrialButton = None
        self.remTrialButton = None
        self.loadParamButton = None
        self.saveTraceButton = None

        self.statusbar = None

    def setup_ui(self, mainwindow):
        # region Formatting templates
        # Text formatting
        # font10 = QFont()
        # font10.setPointSize(10)
        # font11 = QFont()
        # font11.setPointSize(11)
        # font11Bold = QFont()
        # font11Bold.setPointSize(11)
        # font11Bold.setBold(True)
        # font12Bold = QFont()
        # font12Bold.setPointSize(12)
        # font12Bold.setBold(True)
        # font11Under = QFont()
        # font11Under.setPointSize(11)
        # font11Under.setUnderline(True)

        # Size policies
        sizePolicy_Fixed = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy_Fixed.setHorizontalStretch(0)
        sizePolicy_Fixed.setVerticalStretch(0)

        sizePolicy_Ex = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy_Ex.setHorizontalStretch(0)
        sizePolicy_Ex.setVerticalStretch(0)

        sizePolicy_minEx_max = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum)
        sizePolicy_minEx_max.setHorizontalStretch(0)
        sizePolicy_minEx_max.setVerticalStretch(0)

        sizePolicy_max = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy_max.setHorizontalStretch(0)
        sizePolicy_max.setVerticalStretch(0)

        sizePolicy_preferred = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy_preferred.setHorizontalStretch(0)
        sizePolicy_preferred.setVerticalStretch(0)
        # endregion Formatting templates

        if not mainwindow.objectName():
            mainwindow.setObjectName(u"Main Window")
        mainwindow.resize(800, 600)

        """
        # Layout schematic
             0                                                           1                    2                 
            ┏━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┱──────────────────────────────────────┐
        0   ┃ loadVideoButton    │ pathLabel                            ┃ loadParamsButton                     │
            ┡━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━┯━━━━━━━━┯━━━━━━━━━┯━━━━━━━━┪
        1   │ videoFrame                                                ┃ leftLbl  │ leftBx │ rtLbl   │ rtBx   ┃
            │                                                           ┠──────────┼────────┼─────────┼────────┨
            │                                                           ┃ topLbl   │ topBx  │ btmLbl  │ btmBx  ┃
            │                                                           ┠──────────┴────────┴─────────┴────────┨
            │                                                           ┃ boundingBoxButton                    ┃
            │                                                           ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        2   │                                                           │ trialMarkerTable                     │
            │                                                           │                                      │
            │                                                           │                                      │
            ├───────────────────────────────────────────────────────────┤                                      │
        3   │ waveformView                                              │                                      │
            ┢━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━╅───────────────────┬──────────────────┤
        4   ┃ timeStartTextEdit │ trackingSlider       │ timeEndLabel   ┃ newTrialButton    │ remTrialButton   │
            ┣━━━━━━━━━━━━━━━━━━━┷━┯━━━━━━━━━━━━━━━━━━┯━┷━━━━━━━━━━━━━━━━╉───────────────────┴──────────────────┤
        5   ┃ playButton          │ setStartButton   │ setEndButton     ┃ exportButton                         │   
            ┗━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┹──────────────────────────────────────┘
        
        

        ─   ━   │   ┃
        ┌   ┍   ┎   ┏   ┐   ┑   ┒   ┓   └   ┕   ┖   ┗   ┘   ┙   ┚   ┛
        ├   ┝   ┞   ┟   ┠   ┡   ┢   ┣   ┤   ┥   ┦   ┧   ┨   ┩   ┪   ┫
        ┬   ┭   ┮   ┯   ┰   ┱   ┲   ┳   ┴   ┵   ┶   ┷   ┸   ┹   ┺   ┻
        ┼   ┽   ┾   ┿   ╀   ╁   ╂   ╃   ╄   ╅   ╆   ╇   ╈   ╉   ╊   ╋
        
        ═   ║   
        ╒   ╓   ╔   ╕   ╖   ╗   ╘   ╙   ╚   ╛   ╜   ╝   
        ╞   ╟   ╠   ╡   ╢   ╣   ╤   ╥   ╦   ╧   ╨   ╩   
        ╪   ╫   ╬
        """
        self.centralwidget = QWidget(mainwindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.statusbar = QStatusBar()

        self.versionLabel = QLabel(self.centralwidget)
        self.versionLabel.setFixedSize(70, 24)
        self.versionLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.statusbar.addPermanentWidget(self.versionLabel)

        # self.progressBar = QProgressBar(self.centralwidget)
        # self.progressBar.setMinimum(0)
        # self.progressBar.setMaximum(100)
        # self.progressBar.setValue(0)  # Initial value
        # self.statusbar.addPermanentWidget(self.progressBar)

        mainwindow.setStatusBar(self.statusbar)

        self.controlGridLayout = QGridLayout(self.centralwidget)
        self.controlGridLayout.setObjectName(u"controlGridLayout")

        self.fileLayout = QHBoxLayout()

        self.loadVideoButton = QPushButton(self.centralwidget)
        self.loadVideoButton.setObjectName(u"loadButton")
        self.loadVideoButton.setSizePolicy(sizePolicy_Fixed)
        self.loadVideoButton.setFixedSize(QSize(84, 24))
        self.fileLayout.addWidget(self.loadVideoButton)

        self.pathLabel = QLabel(self.centralwidget)
        self.pathLabel.setObjectName(u"pathLabel")
        self.pathLabel.setSizePolicy(sizePolicy_minEx_max)
        self.pathLabel.setMinimumWidth(50)
        self.pathLabel.setFixedHeight(24)
        self.pathLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fileLayout.addWidget(self.pathLabel)

        # self.pathText = QLineEdit(self.centralwidget)
        # self.pathText.setObjectName(u"pathText")
        # self.pathText.setSizePolicy(sizePolicy_Ex)
        # self.pathText.setMinimumSize(QSize(84, 24))
        # self.pathText.setMaximumHeight(24)
        # # self.pathText.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.fileLayout.addWidget(self.pathText)

        self.controlGridLayout.addLayout(self.fileLayout, 0, 0, 1, 1)

        self.loadParamButton = QPushButton(self.centralwidget)
        self.loadParamButton.setObjectName(u"pathLabel")
        self.loadParamButton.setSizePolicy(sizePolicy_minEx_max)
        self.loadParamButton.setMinimumWidth(50)
        self.loadParamButton.setFixedHeight(24)
        self.controlGridLayout.addWidget(self.loadParamButton, 0, 1, 1, 2)

        self.videoFrame = QLabel(self.centralwidget)
        self.videoFrame.setObjectName(u"videoFrame")
        self.videoFrame.setSizePolicy(sizePolicy_Ex)
        # self.videoFrame.setFrameShape(QFrame.StyledPanel)  # for whatever reason the frame seems to screw with the
        # image scaling!
        # https://stackoverflow.com/questions/42833511/qt-how-to-create-image-that-scale-with-window-and-keeps-aspect-ratio
        # self.videoFrame.setFrameShadow(QFrame.Raised)
        self.videoFrame.setScaledContents(False)
        self.videoFrame.setMinimumSize(QSize(600, 600))
        self.videoFrame.setMaximumSize(QSize(1800, 1800))
        self.videoFrame.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.controlGridLayout.addWidget(self.videoFrame, 1, 0, 2, 1)

        self.audioFrame = pg.PlotWidget()
        self.audioFrame.setObjectName(u"audioFrame")
        self.audioFrame.setSizePolicy(sizePolicy_minEx_max)
        # self.audioFrame.setScaledContents(False)
        self.audioFrame.setFixedHeight(100)
        self.audioFrame.hideAxis('bottom')
        self.audioFrame.hideAxis('left')
        self.audioFrame.setMouseEnabled(x=False, y=False)
        self.controlGridLayout.addWidget(self.audioFrame, 3, 0, 1, 1)

        self.boundingGridFrame = QFrame(self.centralwidget)
        self.boundingGridFrame.setObjectName(u"boundingGridFrame")
        self.boundingGridFrame.setSizePolicy(sizePolicy_Fixed)
        self.boundingGridFrame.setFrameShape(QFrame.Shape.StyledPanel)

        self.boundingGridLayout = QGridLayout(self.boundingGridFrame)
        self.boundingGridLayout.setObjectName(u"boundingGridLayout")

        self.boundingLeftLabel = QLabel(self.centralwidget)
        self.boundingLeftLabel.setObjectName(u"boundingLeftLabel")
        self.boundingLeftLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingLeftLabel.setFixedSize(QSize(50, 24))
        self.boundingLeftLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingLeftLabel, 0, 0, 1, 1)

        self.boundingLeftSpinBox = QSpinBox(self.centralwidget)
        self.boundingLeftSpinBox.setObjectName(u"boundingLeftSpinBox")
        self.boundingLeftSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingLeftSpinBox.setFixedSize(QSize(60, 24))
        self.boundingLeftSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingLeftSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingLeftSpinBox, 0, 1, 1, 1)

        self.boundingRightLabel = QLabel(self.centralwidget)
        self.boundingRightLabel.setObjectName(u"boundingRightLabel")
        self.boundingRightLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingRightLabel.setFixedSize(QSize(50, 24))
        self.boundingRightLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingRightLabel, 0, 2, 1, 1)

        self.boundingRightSpinBox = QSpinBox(self.centralwidget)
        self.boundingRightSpinBox.setObjectName(u"boundingRightSpinBox")
        self.boundingRightSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingRightSpinBox.setFixedSize(QSize(60, 24))
        self.boundingRightSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingRightSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingRightSpinBox, 0, 3, 1, 1)

        self.boundingTopLabel = QLabel(self.centralwidget)
        self.boundingTopLabel.setObjectName(u"boundingTopLabel")
        self.boundingTopLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingTopLabel.setFixedSize(QSize(50, 24))
        self.boundingTopLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingTopLabel, 1, 0, 1, 1)

        self.boundingTopSpinBox = QSpinBox(self.centralwidget)
        self.boundingTopSpinBox.setObjectName(u"boundingTopSpinBox")
        self.boundingTopSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingTopSpinBox.setFixedSize(QSize(60, 24))
        self.boundingTopSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingTopSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingTopSpinBox, 1, 1, 1, 1)

        self.boundingBottomLabel = QLabel(self.centralwidget)
        self.boundingBottomLabel.setObjectName(u"boundingBottomLabel")
        self.boundingBottomLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingBottomLabel.setFixedSize(QSize(50, 24))
        self.boundingBottomLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingBottomLabel, 1, 2, 1, 1)

        self.boundingBottomSpinBox = QSpinBox(self.centralwidget)
        self.boundingBottomSpinBox.setObjectName(u"boundingBottomSpinBox")
        self.boundingBottomSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingBottomSpinBox.setFixedSize(QSize(60, 24))
        self.boundingBottomSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingBottomSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingBottomSpinBox, 1, 3, 1, 1)

        self.boundingBoxButton = QPushButton(self.centralwidget)
        self.boundingBoxButton.setObjectName(u"boundingBoxButton")
        self.boundingBoxButton.setSizePolicy(sizePolicy_minEx_max)
        self.boundingBoxButton.setFixedHeight(24)
        self.boundingGridLayout.addWidget(self.boundingBoxButton, 2, 0, 1, 4)

        self.controlGridLayout.addWidget(self.boundingGridFrame, 1, 1, 1, 2)

        self.trialMarkerTable = QTableWidget(self.centralwidget)
        self.trialMarkerTable.setObjectName(u"trialMarkerTable")
        self.trialMarkerTable.setSizePolicy(sizePolicy_Ex)
        self.trialMarkerTable.setMinimumSize(QSize(200, 100))
        self.trialMarkerTable.setMaximumSize(QSize(1000, 1000))
        self.controlGridLayout.addWidget(self.trialMarkerTable, 2, 1, 2, 2)

        self.trackingLayout = QHBoxLayout()

        self.timeStartTextEdit = QLineEdit(self.centralwidget)
        self.timeStartTextEdit.setObjectName(u"startTimestampLabel")
        self.timeStartTextEdit.setSizePolicy(sizePolicy_Fixed)
        self.timeStartTextEdit.setFixedSize(QSize(84, 30))
        self.timeStartTextEdit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trackingLayout.addWidget(self.timeStartTextEdit)

        self.trackingSlider = QSlider(self.centralwidget)
        self.trackingSlider.setObjectName(u"trackingSlider")
        self.trackingSlider.setOrientation(Qt.Orientation.Horizontal)
        self.trackingSlider.setSizePolicy(sizePolicy_minEx_max)
        self.trackingSlider.setMaximumHeight(30)
        self.trackingLayout.addWidget(self.trackingSlider)

        self.timeEndLabel = QLabel(self.centralwidget)
        self.timeEndLabel.setObjectName(u"endTimestampLabel")
        self.timeEndLabel.setSizePolicy(sizePolicy_Fixed)
        self.timeEndLabel.setFixedSize(QSize(84, 30))
        self.timeEndLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trackingLayout.addWidget(self.timeEndLabel)

        self.controlGridLayout.addLayout(self.trackingLayout, 4, 0, 1, 1)

        self.settingLayout = QHBoxLayout()

        self.playVideoButton = QPushButton(self.centralwidget)
        self.playVideoButton.setObjectName(u"playVideoButton")
        self.playVideoButton.setSizePolicy(sizePolicy_minEx_max)
        self.playVideoButton.setMaximumHeight(30)
        self.settingLayout.addWidget(self.playVideoButton)

        self.setStartButton = QPushButton(self.centralwidget)
        self.setStartButton.setObjectName(u"setStartButton")
        self.setStartButton.setSizePolicy(sizePolicy_minEx_max)
        self.setStartButton.setMaximumHeight(30)
        self.settingLayout.addWidget(self.setStartButton)

        self.setEndButton = QPushButton(self.centralwidget)
        self.setEndButton.setObjectName(u"setEndButton")
        self.setEndButton.setSizePolicy(sizePolicy_minEx_max)
        self.setEndButton.setMaximumHeight(30)
        self.settingLayout.addWidget(self.setEndButton)

        self.controlGridLayout.addLayout(self.settingLayout, 5, 0, 1, 1)

        self.addTrialButton = QPushButton(self.centralwidget)
        self.addTrialButton.setObjectName(u"addTrialButton")
        self.addTrialButton.setSizePolicy(sizePolicy_minEx_max)
        self.addTrialButton.setFixedHeight(30)
        self.addTrialButton.setMaximumWidth(120)
        self.controlGridLayout.addWidget(self.addTrialButton, 4, 1, 1, 1)

        self.remTrialButton = QPushButton(self.centralwidget)
        self.remTrialButton.setObjectName(u"remTrialButton")
        self.remTrialButton.setSizePolicy(sizePolicy_minEx_max)
        self.remTrialButton.setFixedHeight(30)
        self.remTrialButton.setMaximumWidth(120)
        self.controlGridLayout.addWidget(self.remTrialButton, 4, 2, 1, 1)

        self.saveTraceButton = QPushButton(self.centralwidget)
        self.saveTraceButton.setObjectName(u"saveTraceButton")
        self.saveTraceButton.setSizePolicy(sizePolicy_minEx_max)
        self.saveTraceButton.setFixedHeight(30)
        self.controlGridLayout.addWidget(self.saveTraceButton, 5, 1, 1, 2)

        mainwindow.setCentralWidget(self.centralwidget)

        self.retranslate_ui(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)

    # setupUi

    def retranslate_ui(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("mainwindow", u"mainwindow", None))
        self.pathLabel.setText(QCoreApplication.translate("mainwindow", u"", None))
        self.loadVideoButton.setText(QCoreApplication.translate("mainwindow", u"Load Video", None))
        self.boundingLeftLabel.setText(QCoreApplication.translate("mainwindow", u"Left", None))
        self.boundingLeftSpinBox.setValue(0)
        self.boundingRightLabel.setText(QCoreApplication.translate("mainwindow", u"Right", None))
        self.boundingRightSpinBox.setValue(0)
        self.boundingTopLabel.setText(QCoreApplication.translate("mainwindow", u"Top", None))
        self.boundingTopSpinBox.setValue(0)
        self.boundingBottomLabel.setText(QCoreApplication.translate("mainwindow", u"Bottom", None))
        self.boundingBottomSpinBox.setValue(0)
        self.loadParamButton.setText(QCoreApplication.translate("mainwindow", u"Load Param", None))
        self.playVideoButton.setText(QCoreApplication.translate("mainwindow", u"Play", None))
        self.setStartButton.setText(QCoreApplication.translate("mainwindow", u"Set Start", None))
        self.setEndButton.setText(QCoreApplication.translate("mainwindow", u"Set End", None))
        self.addTrialButton.setText(QCoreApplication.translate("mainwindow", u"Add Trial", None))
        self.remTrialButton.setText(QCoreApplication.translate("mainwindow", u"Delete Trial", None))
        self.saveTraceButton.setText(QCoreApplication.translate("mainwindow", u"Split Video", None))
        self.boundingBoxButton.setText(QCoreApplication.translate("mainwindow", u"Crop Video", None))
        self.timeStartTextEdit.setText(QCoreApplication.translate("mainwindow", u"0:00:00.0000", None))
        self.timeEndLabel.setText(QCoreApplication.translate("mainwindow", u"0:00:00.0000", None))
        self.versionLabel.setText(f"Version {__version__}")
    # retranslateUi


# class FFmpegClipExporter(QThread):
#     """With assistance from ChatGPT. Using filter_complex in ffmpeg to split multiple, cropped clips from a single
#     video without reinitializing ffmpeg each time"""
#     progress = Signal(str)  # To send status updates to UI
#     finished = Signal(bool)  # True if success, False if error
#
#     def __init__(self, input_path, clip_times, crop, output_paths, parent=None):
#         super().__init__(parent)
#         self.input_path = input_path
#         self.clip_times = clip_times  # list of (start, end) in seconds
#         self.crop = crop              # dict: {'w':640, 'h':480, 'x':100, 'y':50}
#         self.output_paths = output_paths  # list of output filenames
#         self.process = None
#
#         # get framerate because otherwise we can have frameate drift (e.g. 30.2 fps)
#         cmd = [
#             "ffprobe",
#             "-v", "error",
#             "-select_streams", "v:0",  # first video stream
#             "-show_entries", "stream=r_frame_rate",
#             "-of", "json",
#             self.input_path
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True)
#
#         info = json.loads(result.stdout)
#         rate_str = info['streams'][0]['r_frame_rate']
#         num, den = map(int, rate_str.split('/'))
#         self.framerate = num / den
#
#     # noinspection PyUnresolvedReferences
#     def run(self):
#         self.process = QProcess()
#         self.process.setProcessChannelMode(QProcess.MergedChannels)
#         self.process.readyReadStandardError.connect(self.handle_stderr)
#         self.process.finished.connect(self.handle_finished)
#
#         args = self.build_ffmpeg_args()
#         self.process.start("ffmpeg", args)
#         self.exec()  # Enters a nested event loop, keeps thread alive
#
#     def handle_stderr(self):
#         data = self.process.readAllStandardError().data().decode()
#         for line in data.splitlines():
#             if "frame=" in line or "time=" in line:
#                 self.progress.emit(line.strip())
#
#     # noinspection PyUnresolvedReferences
#     def handle_finished(self):
#         success = (self.process.exitStatus() == QProcess.NormalExit and self.process.exitCode() == 0)
#         self.finished.emit(success)
#         self.quit()  # Exit thread event loop
#
#     def build_ffmpeg_args(self):
#         crop_str = f"crop={self.crop['w']}:{self.crop['h']}:{self.crop['x']}:{self.crop['y']}"
#         num_clips = len(self.clip_times)
#
#         # Build filter_complex string
#         filter_parts = [
#             f"[0:v]{crop_str},split={num_clips}" + ''.join(f"[v{i}]" for i in range(num_clips)),
#             f"[0:a]asplit={num_clips}" + ''.join(f"[a{i}]" for i in range(num_clips)),
#         ]
#
#         for i, (start, end) in enumerate(self.clip_times):
#             filter_parts += [
#                 f"[v{i}]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}out]",
#                 f"[a{i}]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}out]"
#             ]
#
#         filter_complex = "; ".join(filter_parts)
#
#         args = [
#             "-y",  # Overwrite output
#             "-i", self.input_path,
#             "-filter_complex", filter_complex,
#         ]
#
#         for i, out in enumerate(self.output_paths):
#             args += ["-map", f"[v{i}out]", "-map", f"[a{i}out]", out]
#
#         args += [
#             "-r", self.framerate
#         ]
#
#         return args
#

class MainWindow(QMainWindow, UiMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_ui(self)
        self.setWindowTitle("Bobbing Trial Splitting")

        print(f"Trial Splitter by Andrew Rouse (08/2025). Version {__version__}")

        self.process = QProcess(self)  # for running ffmpeg without blocking the ui input
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setProgram('ffmpeg')

        self.loadVideoButton.clicked.connect(lambda: self.load_video())
        self.boundingLeftSpinBox.editingFinished.connect(self.update_bounding_box)
        self.boundingRightSpinBox.editingFinished.connect(self.update_bounding_box)
        self.boundingTopSpinBox.editingFinished.connect(self.update_bounding_box)
        self.boundingBottomSpinBox.editingFinished.connect(self.update_bounding_box)
        self.boundingBoxButton.clicked.connect(lambda: self.set_box())
        self.trialMarkerTable.cellClicked.connect(self.select_trial)
        self.trackingSlider.setTracking(True)
        self.trackingSlider.sliderPressed.connect(self.drag_on)
        self.trackingSlider.sliderMoved.connect(self.user_move_slider)
        self.trackingSlider.sliderReleased.connect(self.drag_off)
        self.trackingSlider.valueChanged.connect(self.adjust_trackingslider)
        self.addTrialButton.clicked.connect(self.trial_add)
        self.remTrialButton.clicked.connect(self.trial_rem)
        self.timeStartTextEdit.editingFinished.connect(self.user_set_time)
        self.playVideoButton.clicked.connect(self.video_play)
        self.setStartButton.clicked.connect(lambda: self.set_trial_start())
        self.setEndButton.clicked.connect(lambda: self.set_trial_end())
        self.loadParamButton.clicked.connect(lambda: self.load_settings())
        self.saveTraceButton.clicked.connect(lambda: self.split_video())

        # table setup
        self.trialMarkerTable.setColumnCount(3)
        self.trialMarkerTable.setHorizontalHeaderLabels(["Trial", "Start", "End"])
        self.trialMarkerTable.verticalHeader().setVisible(False)
        self.trialMarkerTable.setColumnWidth(0, 40)
        self.trialMarkerTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trialMarkerTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # self.trialMarkerTable.horizontalHeader().setStretchLastSection(True)

        # disable buttons until video is loaded
        self.trackingSlider.setEnabled(False)
        self.playVideoButton.setEnabled(False)
        self.boundingLeftSpinBox.setEnabled(False)
        self.boundingRightSpinBox.setEnabled(False)
        self.boundingTopSpinBox.setEnabled(False)
        self.boundingBottomSpinBox.setEnabled(False)
        self.boundingBoxButton.setEnabled(False)
        self.timeStartTextEdit.setEnabled(False)
        self.saveTraceButton.setEnabled(False)
        self.addTrialButton.setEnabled(False)
        self.remTrialButton.setEnabled(False)
        self.setStartButton.setEnabled(False)
        self.setEndButton.setEnabled(False)

        # self.videoFrame.installEventFilter(self)
        self.thread = None

        # video stats
        self.videoFrameRate = None
        self.cap = None  # capture stream
        self.vidWidth = None
        self.vidHeight = None
        self.frameCurrent = None
        self.frameCurrentNumber = None

        self.bbox = [0, 0, 0, 0]  # current position of tracker
        # self.bboxOriginal = None  # original position of tracker
        self.bboxImage = None  # image in the original tracker selection for matching
        self.cropWidth = None
        self.cropHeight = None
        self.cropX = None
        self.cropY = None
        self.bboxPainter = QPainter(self.videoFrame)

        self.trial = 0  # current trial number
        self.trialCount = 1  # number of trials

        self.trialTable = None

        self.videoPath = None
        self.videoName = None
        self.videoExt = None

        self.audioWaveform = None
        self.audioWavePlot = None
        self.audioTrackerLine = None

        self.user_dragging = False

        self.trialdf = None
        self.folderPath = None
        # self.outputTemp = None
        self.clipDur = None
        self.currRow = None
        self.fullStart = None
        self.fullEnd = None

        # self.timeSearch = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

        # if self.progressBar:
        #     self.progressBar.hide()
        self.progressDialog = None

        self.exporter = None

        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')

        self.ffmpegInstalled, message = check_ffmpeg_installed()
        self.update_status(message)
        if not self.ffmpegInstalled:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            message = (
                "ffmpeg is not detected."
                "Trials can be identified but cannot be split into individual files."
            )
            msg_box.setText(message)
            msg_box.setWindowTitle("ffmpeg missing")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()  # Displays the message box and waits for user interaction

        # get video codec
        self.videoCodec = None

        # self.center_on_screen()

    def eventFilter(self, widget, event):
        """Resize the current frame with the window.
        Activates on every event, but filtered so it only affects resize events of the video frame"""

        # https://stackoverflow.com/questions/21041941/how-to-autoresize-qlabel-pixmap-keeping-ratio-without-using-classes/21053898#21053898
        if event.type() == QEvent.Type.Resize and widget is self.videoFrame:
            self.videoFrame.setPixmap(self.videoFrame.pixmap.scaled(self.videoFrame.width(), self.videoFrame.height(),
                                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation))

            # # resize the window
            # self.resize(self.sizeHint().width(), self.sizeHint().height())
            return True
        return QMainWindow.eventFilter(self, widget, event)

    def closeEvent(self, event):
        """I think this triggers when the window is closed to gracefully close the playback thread"""
        # if thread hasn't initialized yet, then the stop and close methods don't exist yet so we need to check
        if hasattr(self.thread, 'stop'):
            self.thread.stop()
        if hasattr(self.thread, 'close'):
            self.thread.close()

    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        screen_center = screen.availableGeometry().center()

        # Get the window's frame geometry
        frame_geo = self.frameGeometry()

        # Move the frame's center to the screen's center
        frame_geo.moveCenter(screen_center)

        # Apply the new position to the window
        self.move(frame_geo.topLeft())

    def load_video(self, filepath=None):
        """Load the selected video, if provided, or prompt user to select a video"""
        if filepath is None:
            fileName = QFileDialog.getOpenFileName(self, 'Open Video')
            if fileName[0]:
                filepath = fileName[0]

        self.videoPath = filepath
        self.pathLabel.setText(self.videoPath)
        self.videoName = os.path.splitext(os.path.split(self.videoPath)[1])[0]
        self.videoExt = os.path.splitext(os.path.split(self.videoPath)[1])[1]
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not read video file",
                                 QMessageBox.StandardButton.Ok)
        else:
            # load first frame
            ok, frame = self.cap.read()
            if not ok:
                QMessageBox.critical(self, "Error", "Could not read video file",
                                     QMessageBox.StandardButton.Ok)
                return False
            else:
                # get stats - framerate, length
                self.videoFrameRate = self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                endStamp = self.get_time_from_frame(frame_count)
                self.timeEndLabel.setText(endStamp)
                self.trackingSlider.setMaximum(frame_count)

                # create blank trial table
                self.trialTable = list(range(frame_count))

                # enable buttons
                self.trackingSlider.setEnabled(True)
                self.boundingLeftSpinBox.setEnabled(True)
                self.boundingRightSpinBox.setEnabled(True)
                self.boundingTopSpinBox.setEnabled(True)
                self.boundingBottomSpinBox.setEnabled(True)
                self.boundingBoxButton.setEnabled(True)
                self.timeStartTextEdit.setEnabled(True)
                self.saveTraceButton.setEnabled(True)
                self.addTrialButton.setEnabled(True)
                self.remTrialButton.setEnabled(True)
                self.setStartButton.setEnabled(True)
                self.setEndButton.setEnabled(True)
                self.playVideoButton.setEnabled(True)

                # display first frame
                self.frameCurrent = frame
                self.frameCurrentNumber = 1
                self.update_image(self.frameCurrent, self.frameCurrentNumber)
                # resize the window
                self.resize(self.sizeHint().width(), self.sizeHint().height())
                # reupdate the image to make it happy with the aspect ratio (otherwise it constantly resizes
                # itself to try to meet the aspect ratio)

                self.vidHeight = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.vidWidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

                self.boundingLeftSpinBox.setMaximum(self.vidWidth)
                self.boundingRightSpinBox.setMaximum(self.vidWidth)
                self.boundingTopSpinBox.setMaximum(self.vidHeight)
                self.boundingBottomSpinBox.setMaximum(self.vidHeight)

                # display audio
                # fs = get_audio_fs(self.videoPath)

                self.videoCodec = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        self.videoPath
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                ).stdout.strip()

                # ffmpeg
                waveform, fs = extract_audio(self.videoPath)
                print(f'Audio sample rate (Hz): {fs}')
                waveform = waveform.mean(axis=1)  # convert from stereo to mono
                dsFactor = 10
                self.audioWaveform = waveform[::dsFactor]  # downsample for plotting efficacy

                nSamp = len(self.audioWaveform)
                time = np.linspace(0, nSamp / (fs / dsFactor), num=nSamp, endpoint=False)
                self.audioWavePlot = self.audioFrame.plot(time, self.audioWaveform, pen=pg.mkPen('w', width=1))

                self.audioTrackerLine = pg.InfiniteLine(0, pen=pg.mkPen('y', width=1))
                self.audioFrame.addItem(self.audioTrackerLine)

                # EventFilter resizes the frame as the window resizes
                self.videoFrame.installEventFilter(self)

                return True

    # noinspection PyUnresolvedReferences
    def load_settings(self):
        """Load a previously saved settings file"""
        fileName = QFileDialog.getOpenFileName(self, 'Open Settings File', filter="*.csv")
        if fileName[0]:
            fileName = fileName[0]
            settingsdf = pd.read_csv(fileName)
            # videoName = os.path.splitext(os.path.split(fileName)[1])[0]
            videoPath = settingsdf[['Video']].tail(1).values[0][0]

            success = self.load_video(filepath=videoPath)
            if success:
                # set bbox, crop, bounding values
                self.bbox = settingsdf[['cropX', 'cropY', 'cropWidth', 'cropHeight']].tail(1).values[0]
                self.update_from_bbox()

                # set trials
                trialdf = settingsdf[['Trial', 'Start', 'End']].iloc[:-1]
                # clear table
                for i in range(1, self.trialCount):
                    self.trialMarkerTable.removeRow(i)
                    self.trialCount -= 1

                for i in range(len(trialdf)):
                    # lastTrial = self.trialCount - 1
                    self.trialMarkerTable.insertRow(i)
                    trialNumItem = QTableWidgetItem(Qt.DisplayRole, str(int(trialdf[['Trial']].iloc[i].values[0])))
                    trialNumItem.setData(Qt.DisplayRole, str(int(trialdf[['Trial']].iloc[i].values[0])))
                    self.trialMarkerTable.setItem(i, 0, trialNumItem)
                    # fill remaining cells with an actual item
                    self.trialMarkerTable.setItem(i, 1, QTableWidgetItem(trialdf[['Start']].iloc[i].values[0]))
                    self.trialMarkerTable.setItem(i, 2, QTableWidgetItem(trialdf[['End']].iloc[i].values[0]))
                    self.trialCount += 1

                    # startTime = QTableWidgetItem(self.timeStartTextEdit.text())
                    # self.trialMarkerTable.setItem(self.trial, 1, startTime)

    def set_box(self):
        """Set the bounding/crop box by selecting on the frame itself"""
        # self.bbox = (1261, 586, 60, 72)
        frameCopy = self.frameCurrent  # make a copy of the frame and add the rectangle to it rather than the original

        # set up instruction text
        boxInstr = ["Select object to track with", "left mouse button, press", "Enter when finished"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(boxInstr):
            textSize = cv2.getTextSize(line, font, .8, 2)[0]
            textX = textSize[1]  # (frameCopy.shape[1] - textSize[0]) / 2
            textY = textSize[1] * 2 * (i + 1)
            textOrg = (textX, textY)
            cv2.putText(frameCopy, line, textOrg,
                        font, 1, (0, 0, 255), 2)

        # original frame size
        frameWidth = self.videoFrame.width()
        frameHeight = self.videoFrame.height()

        # actually select box
        windowName = 'Press enter to finish'
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, frameWidth, frameHeight)
        self.bbox = cv2.selectROI(windowName, frameCopy, False, printNotice=False)

        cv2.destroyWindow('Press enter to finish')
        if all(x > 0 for x in self.bbox):
            self.update_from_bbox()
        else:
            print("No ROI selected")

    def update_from_bbox(self):
        """update crop/bounding values after bbox is updated"""
        self.cropWidth = int(self.bbox[2])
        self.cropHeight = int(self.bbox[3])
        self.cropX = int(self.bbox[0])
        self.cropY = int(self.bbox[1])

        # blockSignals so that updating the text boxes doesn't trigger updating self.bbox in an infinite loop
        self.boundingLeftSpinBox.blockSignals(True)
        self.boundingRightSpinBox.blockSignals(True)
        self.boundingTopSpinBox.blockSignals(True)
        self.boundingBottomSpinBox.blockSignals(True)
        self.boundingLeftSpinBox.setValue(self.bbox[0])
        self.boundingRightSpinBox.setValue(self.vidWidth - (self.bbox[0] + self.bbox[2]))
        self.boundingTopSpinBox.setValue(self.bbox[1])
        self.boundingBottomSpinBox.setValue(self.vidHeight - (self.bbox[1] + self.bbox[3]))
        self.boundingLeftSpinBox.blockSignals(False)
        self.boundingRightSpinBox.blockSignals(False)
        self.boundingTopSpinBox.blockSignals(False)
        self.boundingBottomSpinBox.blockSignals(False)

    def update_bounding_box(self):
        """After the bounding box has been changed via the spinboxes, update all the relevant variables"""
        # boundLeft = self.boundingLeftSpinBox.text()
        # boundRight = self.boundingRightSpinBox.text()  # .setText(self.vidWidth - (self.bbox[0] + self.bbox[2])))
        # boundTop = self.boundingTopSpinBox.text() # .setText(self.bbox[1])
        # boundBottom = self.boundingBottomSpinBox.text()  #.setText(self.vidHeight - (self.bbox[1] + self.bbox[3]))

        cropLeft = self.boundingLeftSpinBox.value()
        cropRight = self.boundingRightSpinBox.value()
        cropTop = self.boundingTopSpinBox.value()
        cropBottom = self.boundingBottomSpinBox.value()

        self.cropX = int(cropLeft)
        self.cropY = int(cropTop)
        self.cropWidth = int(self.vidWidth - (cropLeft + cropRight))
        self.cropHeight = int(self.vidHeight - (cropTop + cropBottom))

        self.bbox = [self.cropX, self.cropY, self.cropWidth, self.cropHeight]

    def set_trial_start(self):
        startTime = QTableWidgetItem(self.timeStartTextEdit.text())
        self.trialMarkerTable.setItem(self.trial, 1, startTime)

    def set_trial_end(self):
        endTime = QTableWidgetItem(self.timeStartTextEdit.text())
        self.trialMarkerTable.setItem(self.trial, 2, endTime)

    def select_trial(self, row):
        """Update which trial is selected in the table so the set_trial_start and set_trial_end operate on the
        correct row"""
        self.trial = row

    # noinspection PyUnresolvedReferences
    def trial_add(self):
        """Add blank row to the trial table"""
        lastTrial = self.trialCount - 1
        self.trialMarkerTable.insertRow(lastTrial)
        trialNumItem = QTableWidgetItem()
        trialNumItem.setData(Qt.DisplayRole, self.trialCount)
        self.trialMarkerTable.setItem(lastTrial, 0, trialNumItem)
        # fill remaining cells with an actual item
        self.trialMarkerTable.setItem(lastTrial, 1, QTableWidgetItem())
        self.trialMarkerTable.setItem(lastTrial, 2, QTableWidgetItem())
        self.trialCount += 1

    # noinspection PyUnresolvedReferences
    def trial_rem(self):
        """Remove a trial from the table, as long as there are enough trials"""
        if self.trialCount > 1:
            self.trialMarkerTable.removeRow(self.trial)
            self.trialCount -= 1
            # renumber trials after deletion
            for row in range(self.trialCount - 1):  # trialCount - 1 because trialCount is not 0-indexed
                self.trialMarkerTable.item(row, 0).setData(Qt.DisplayRole, row + 1)

    def video_play(self):
        """Start playing the video. This won't play at realtime speed because this is a hacky way of loading and
        displaying the video, plus all the other processes that happen"""
        # update the play button
        self.playVideoButton.setText("Pause")
        self.playVideoButton.clicked.disconnect(self.video_play)
        self.playVideoButton.clicked.connect(self.video_stop)
        # disable the buttons that could change the frame
        self.timeStartTextEdit.setEnabled(False)
        self.trackingSlider.setEnabled(False)

        # create the video capture thread
        self.thread = VideoThread(self.cap)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_tracker)
        # start the thread
        self.thread.start()

    def video_stop(self):
        """Stop the automatic playback"""
        self.playVideoButton.clicked.disconnect(self.video_stop)
        self.playVideoButton.clicked.connect(self.video_play)
        self.playVideoButton.setText("Play")
        self.thread.stop()
        self.timeStartTextEdit.setEnabled(True)
        self.trackingSlider.setEnabled(True)
        # self.thread.change_pixmap_signal.disconnect(self.update_tracker)

    # def load_frame(self, targetFrame):
    #     # set the tracking slider to the specified frame, including loading the new image
    #     self.trackingSlider.setValue(target_frame)
    #     newTimeStamp = self.get_time_from_frame(target_frame)
    #     self.timeStartTextEdit.setText(newTimeStamp)
    #     self.frameCurrentNumber = target_frame
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
    #         ret, cv_img = self.cap.read()
    #         if ret:
    #             self.update_image(cv_img, self.frameCurrentNumber)

    def drag_on(self):
        """User started dragging the slider. Changing the slider triggers adjust_trackingslider, so user_dragging is a
        flag that keeps frames from loading while you drag - otherwise dragging would be very slow"""
        self.user_dragging = True

    def drag_off(self):
        """User let go of the slider, disable the flag and update the frame"""
        self.user_dragging = False
        self.adjust_trackingslider()

    def update_frame_number(self, targetFrame):
        """Update tracking slider and audio tracker while video is playing (so don't trigger a new image load)"""
        self.trackingSlider.blockSignals(True)
        self.trackingSlider.setValue(targetFrame)
        self.trackingSlider.blockSignals(False)

        self.update_timestamp(targetFrame)
        self.frameCurrentNumber = targetFrame
        self.update_audio_tracker(targetFrame)

    def adjust_trackingslider(self):
        """after dragging, user releases the tracking slider. Load the frame at the slider position"""
        if not self.user_dragging:
            targetFrame = self.trackingSlider.value()
            self.frameCurrentNumber = targetFrame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
            ret, cv_img = self.cap.read()
            if ret:
                self.update_image(cv_img, self.frameCurrentNumber)

    def user_move_slider(self, targetFrame):
        """user is dragging the slider - update the timestamp based on slider position without loading a new frame"""
        self.update_timestamp(targetFrame)
        self.update_audio_tracker(targetFrame)

    def user_set_time(self):
        """User set the timestamp - set the slider, load the new frame"""
        # convert time to nearest frame number
        userTS = self.timeStartTextEdit.text()
        userTS = userTS.split(":")
        if len(userTS) > 2:
            # timestamp includes hours
            nSec = float(userTS[0]) * 60 * 60 + float(userTS[1]) * 60 + float(userTS[2])
        else:
            # timestamp is just minutes and seconds
            nSec = float(userTS[0]) * 60 + float(userTS[1])
        rawFrame = nSec * self.videoFrameRate
        targetFrame = int(round(rawFrame, 0))

        # now load the frame at the slider position
        self.frameCurrentNumber = targetFrame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
        ret, cv_img = self.cap.read()
        if ret:
            self.update_image(cv_img, self.frameCurrentNumber)

    def update_tracker(self, frame, frame_number):
        """Triggered by playing the video - updates the video frame """
        if frame_number <= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            # Display result
            self.update_image(frame, frame_number)

    def update_audio_tracker(self, targetFrame):
        """Update the position of the tracking line on the audio waveform"""
        if self.audioTrackerLine:
            newTS = targetFrame / self.videoFrameRate
            # self.audioTrackerLine.setData([newTS, newTS], [-10, 10])
            self.audioTrackerLine.setPos(newTS)

    def update_timestamp(self, targetFrame):
        """Update value of timeStartTextEdit based on frame number, without triggering the signal"""
        # targetFrame = self.trackingSlider.value()
        newTimeStamp = self.get_time_from_frame(targetFrame)
        self.timeStartTextEdit.blockSignals(True)
        self.timeStartTextEdit.setText(newTimeStamp)
        self.timeStartTextEdit.blockSignals(False)

    @Slot(np.ndarray)
    def update_image(self, cv_img, frame_number):
        """Updates the image_label with a new opencv image"""
        if all(x > 0 for x in self.bbox):
            # self.bboxPainter.drawRect(self.cropX, self.cropY, self.cropWidth, self.cropHeight)
            cv2.rectangle(cv_img, (self.cropX, self.cropY), (self.cropX + self.cropWidth, self.cropY + self.cropHeight),
                          (255, 0, 0), 3)

        self.frameCurrent = cv_img
        frame = self.convert_cv_qt(cv_img)
        self.videoFrame.setPixmap(frame)
        self.videoFrame.pixmap = QPixmap(frame)
        self.update_frame_number(frame_number)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.videoFrame.width(), self.videoFrame.height(),
                                        Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def get_time_from_frame(self, framenumber):
        """return time as string (with commented lines for returning as datetime)"""
        nSecondsRaw = framenumber / self.videoFrameRate
        nMin, nSec = divmod(nSecondsRaw, 60)
        nHour, nMin = divmod(nMin, 60)
        # nMicro = round((nSec % 1)*10**6)
        timeStr = f"{round(nHour)}:{round(nMin):02}:{nSec:07.4f}"  # seconds with 07 for len(nSec)
        # ts = datetime.time(hour=int(nHour), minute=int(nMin), second=int(nSec), microsecond=int(nMicro))
        # timeStr = ts.strftime('%H:%M:%S.%f')
        return timeStr

    def split_video(self):
        """Split Video button pressed - start the whole process of cropping the input video and clipping into trials"""
        brushHighlight = QBrush(QColor('yellow'))
        brushReg = QBrush(QColor('white'))

        self.trialCount = self.trialMarkerTable.rowCount()
        # Validate that all trials have start and end before trying to clip/crop them
        trialsValid = True
        for row in range(self.trialCount):
            trialNItem = self.trialMarkerTable.item(row, 0)
            startItem = self.trialMarkerTable.item(row, 1)
            endItem = self.trialMarkerTable.item(row, 2)
            if not trialNItem.text():
                trialsValid = False
                trialNItem.setBackground(brushHighlight)
            else:
                trialNItem.setBackground(brushReg)
            if not startItem.text():
                trialsValid = False
                startItem.setBackground(brushHighlight)
            else:
                startItem.setBackground(brushReg)
            if not endItem.text():
                trialsValid = False
                endItem.setBackground(brushHighlight)
            else:
                endItem.setBackground(brushReg)

        if trialsValid:
            self.folderPath = QFileDialog.getExistingDirectory(self, 'Save to Folder')
            if self.folderPath:
                # loop through trials and export cropped clips
                columnHeaders = ['Trial', 'Start', 'End']
                self.trialdf = pd.DataFrame(columns=columnHeaders, index=range(self.trialCount))
                # First export settings for future reference (and in case of a crash before it finishes)
                bboxdf = pd.DataFrame({'cropX': [self.cropX], 'cropY': [self.cropY],
                                       'cropWidth': [self.cropWidth], 'cropHeight': [self.cropHeight],
                                       'Video': [self.videoPath]})
                for row in range(self.trialCount):
                    self.trialdf.loc[row, 'Trial'] = self.trialMarkerTable.item(row, 0).text()
                    # # shift times back by 1 frame because seems like code gets frame n+1
                    # start = self.trialMarkerTable.item(row, 1).text()
                    # start = get_seconds_from_time(start) - 1/self.videoFrameRate
                    # start = get_time_from_seconds(start)
                    self.trialdf.loc[row, 'Start'] = self.trialMarkerTable.item(row, 1).text()
                    # end = self.trialMarkerTable.item(row, 2).text()
                    # end = get_seconds_from_time(end) - 1 / self.videoFrameRate
                    # end = get_time_from_seconds(end)
                    self.trialdf.loc[row, 'End'] = self.trialMarkerTable.item(row, 2).text()

                # add the bbox values as the last row, in separate columns
                outdf = pd.concat([self.trialdf, bboxdf], ignore_index=True)

                # export trial times and names
                textFileName = self.videoName + "_trialTimes.csv"
                outputPath = os.path.join(self.folderPath, textFileName)
                outdf.to_csv(outputPath, index=False)

                # again, if ffmpeg is missing, alert user but inform that parameter file was saved
                if not self.ffmpegInstalled:
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Icon.Warning)
                    message = (
                        f"Trials have been saved to a settings file ({outputPath}), "
                        "but splitting requires ffmpeg. Once ffmpeg is installed you can reload the settings file "
                        "and try again."
                    )
                    msg_box.setText(message)
                    msg_box.setWindowTitle("ffmpeg missing")
                    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msg_box.exec()  # Displays the message box and waits for user interaction
                    return
                self.block_ui(False)
                
                # start clipping
                if self.progressBar:
                    self.progressBar.show()

                # self.outputTemp = os.path.join(self.folderPath, 'tempVideo.MP4')

                self.start_clipping()

        else:
            self.update_status(f'Missing trial data')

    # def crop_video(self):
    #     """Crop the original video, preserving bit rate, frame rate, etc. into a temporary file that will then get
    #     split into clips"""
    #
    #     # first get the input video information
    #     vCodec, fps_str = subprocess.run(
    #         [
    #             "ffprobe", "-v", "error",
    #             "-select_streams", "v:0",
    #             "-show_entries", "stream=codec_name,r_frame_rate",
    #             "-of", "default=noprint_wrappers=1:nokey=1",
    #             self.videoPath
    #         ],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         text=True
    #     ).stdout.strip().split("\n")
    #     # fps = eval(fps_str)
    #     # stream_info = json.loads(result.stdout)["streams"][0]
    #
    #     # codec = stream_info.get("codec_name", "libx265")  # HEVC should report as 'hevc'
    #     # framerate = stream_info.get("r_frame_rate")
    #     if fps_str and fps_str != "0/0":
    #         framerate = str(eval(fps_str))  # Convert "30000/1001" -> 29.97
    #     else:
    #         framerate = None
    #
    #     # get start and end of areas of interest
    #     self.fullStart = get_seconds_from_time(self.trialdf['Start'].iloc[0])
    #     self.fullEnd = get_seconds_from_time(self.trialdf['End'].iloc[-1])
    #     self.clipDur = self.fullEnd - self.fullStart
    #
    #     # Step 2: Build FFmpeg command
    #     ffmpeg_cmd = [
    #         "-y", "-i", self.videoPath,
    #         "-ss", str(self.fullStart - 0.5),  # add an extra half second on either end
    #         "-t", str(self.clipDur + 0.5),
    #         "-vf", f"crop={self.cropWidth}:{self.cropHeight}:{self.cropX}:{self.cropY}"
    #     ]
    #
    #     # Video codec and settings
    #     if vCodec == "hevc":
    #         ffmpeg_cmd += ["-c:v", "libx265"]
    #     else:
    #         ffmpeg_cmd += ["-c:v", "libx264"]  # fallback
    #
    #     if framerate:
    #         ffmpeg_cmd += ["-r", framerate]
    #
    #     # set keyframe freq
    #     ffmpeg_cmd += ["-g", "15"]
    #
    #     # Audio copy
    #     ffmpeg_cmd += ["-c:a", "copy"]
    #
    #     # now specify the clip points as keyframes
    #     boundaries = sorted(set(self.trialdf['Start'].tolist() + self.trialdf['End'].tolist()))
    #     boundaries = list(map(get_seconds_from_time, boundaries))
    #     boundaries = [x - self.fullStart for x in boundaries]
    #     keyframe_str = ",".join(str(round(t, 4)) for t in boundaries)
    #     # keyframe_str = ",".join(boundaries)
    #     # ffmpeg_cmd += ["-force_key_frames", f'{keyframe_str}']
    #     ffmpeg_cmd += ["-force_key_frames", keyframe_str]
    #
    #     # self.outputTemp = os.path.join(self.folderPath, 'tempVideo.MP4')
    #     ffmpeg_cmd += ["-reset_timestamps", "1"]
    #     ffmpeg_cmd += [self.outputTemp]
    #
    #     # get video duration for progress bar
    #     # result = subprocess.run(
    #     #     [
    #     #         "ffprobe",
    #     #         "-v", "error",
    #     #         "-show_entries", "format=duration",
    #     #         "-of", "default=noprint_wrappers=1:nokey=1",
    #     #         self.videoPath
    #     #     ],
    #     #     stdout=subprocess.PIPE,
    #     #     stderr=subprocess.PIPE,
    #     #     text=True
    #     # )
    #     # self.clipDur = float(result.stdout.strip())
    #
    #     self.update_status(f"Initializing. This may take a while, and can be resource-intensive!")
    #     self.progressBar.setValue(0)
    #     self.process.setArguments(ffmpeg_cmd)
    #     self.process.readyReadStandardError.connect(self.crop_stderr)
    #     self.process.finished.connect(self.cropping_finished)
    #     self.process.start()
    #
    # def clip_stderr(self):
    #     """Reading the output of ffmpeg process and extracting the current video time to drive progressBar"""
    #     output = self.process.readAllStandardOutput().data().decode()
    #     for line in output.splitlines():
    #         match = self.timeSearch.search(line)
    #         if match:
    #             hours, minutes, seconds = match.groups()
    #             currentTime = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    #             self.update_status("Step 1: Cropping input video.", False)
    #
    #             progress = 100 * (currentTime / self.clipDur)
    #             self.progressDialog.update_clip_progress(progress)
    #             if self.progressBar:
    #                 self.progressBar.setValue(progress)
    #
    # def cropping_finished(self):
    #     """Cropping has finished, start clipping"""
    #     self.update_status(f"Finished cropping input!")
    #     self.process.readyReadStandardError.disconnect(self.crop_stderr)
    #     self.process.finished.disconnect(self.cropping_finished)
    #     # self.process.readyReadStandardOutput.connect(self.crop_stdout)
    #     self.process.readyReadStandardError.connect(self.clip_stderr)
    #     self.process.finished.connect(self.clipping_finished)
    #     self.currRow = 0
    #     self.start_clip_video()

    def update_status(self, message, repeat=True):
        """Short separate function for writing messages to the status bar (and printing to console)"""
        self.statusbar.showMessage(message)
        if repeat:
            print(message)

    # def clip_stderr(self):
    #     """Only necessary if we want to parse the output of the clipping process, but unlikely because it's so fast"""
    #     # output = self.process.readAllStandardError().data().decode()
    #     # print(output)
    #     pass

    def clip_output_parse(self):
        """Parse the clipping process outout to get progress"""
        while self.process.canReadLine():
            line = self.process.readLine().data().decode("utf-8").strip()
            if line.startswith("out_time="):
                self.progressDialog.update_clip_text("Encoding clip...")
                # timeStr = line.split("=")[1]
                # if timeStr != "N/A":
                #     t = get_seconds_from_time(timeStr)
                #     percent = int((t / self.clipDur) * 100)
                #
                #     # self.progressDialog.update_clip_progress(percent)

    def start_clipping(self):
        """Create progress bar window and start the clipping process"""
        self.process.readyReadStandardOutput.connect(self.clip_output_parse)
        self.process.finished.connect(self.clipping_finished)
        self.currRow = 0
        self.progressDialog = ProgressDialog(self.trialCount)
        self.progressDialog.cancel_requested.connect(self.stop_process)
        self.progressDialog.show()
        self.start_clip_video()

    def stop_process(self):
        """Interrupt the QProcess if cancel button clicked on dialog"""
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.block_ui(True)

    def start_clip_video(self):
        """Start clipping the next video"""
        trialN = self.trialdf.loc[self.currRow, 'Trial']
        start = self.trialdf.loc[self.currRow, 'Start']
        end = self.trialdf.loc[self.currRow, 'End']

        # update the progress dialog with clip info
        self.progressDialog.update_overall_text(
            f'Clipping trial {trialN} of {self.trialCount} (Clip {self.currRow})')
        self.progressDialog.update_overall_progress(self.currRow)
        self.progressDialog.update_clip_text("Decoding clip... (This may take awhile)")
        # self.progressDialog.update_clip_progress(0)

        # self.currRow = trialN
        currFileName = self.videoName + "_t" + trialN + self.videoExt
        output_path = os.path.join(self.folderPath, currFileName)

        # Step 2: Build FFmpeg command
        ffmpeg_cmd = [
            "-y", "-i", self.videoPath,
            "-ss", start,
            "-to", end,
            "-vf", f"crop={self.cropWidth}:{self.cropHeight}:{self.cropX}:{self.cropY}",
            "-map", "0:v",  # To remove any extra streams
            "-map", "0:a?"
        ]

        # Video codec and settings
        if self.videoCodec == "hevc":
            ffmpeg_cmd += ["-c:v", "libx265"]
        else:
            ffmpeg_cmd += ["-c:v", "libx264"]  # fallback

        if self.videoFrameRate:
            ffmpeg_cmd += ["-r", str(self.videoFrameRate)]

        # # set keyframe freq
        # ffmpeg_cmd += ["-g", "15"]

        # Audio copy
        ffmpeg_cmd += ["-c:a", "copy"]

        ffmpeg_cmd += ["-progress", "pipe:1",
                       "-nostats"]

        ffmpeg_cmd += [output_path]

        # # get number of frames for progress bar
        startSec = get_seconds_from_time(start)
        endSec = get_seconds_from_time(end)
        self.clipDur = endSec - startSec

        # self.update_status(f'Step 2: Clipping trial {trialN} of {self.trialCount} (Clip {self.currRow})')
        # if self.progressBar:
        #     progress = 100 * (int(trialN) / self.trialCount)
        #     self.progressBar.setValue(progress)
        # self.process.setProgram('ffmpeg')
        self.process.setArguments(ffmpeg_cmd)
        self.process.start()

    def clipping_finished(self):
        """Current clipping process finished. Either move on to next trial or reenable the interface"""
        if self.currRow >= self.trialCount - 1:

            msg = "Export complete!"
            self.update_status(msg)

            self.progressDialog.update_overall_progress(self.trialCount)
            self.progressDialog.update_overall_text("All clips exported!")
            self.progressDialog.cancelButton.setText("Done")
            self.block_ui(True)

            # # remove the temporary clipped file
            # if os.path.exists(self.outputTemp):
            #     os.remove(self.outputTemp)
            #     print('Temp file removed successfully')
            # else:
            #     print('Temp file does not exist')

        else:
            self.currRow += 1
            self.start_clip_video()

    def block_ui(self, editable=True):
        self.trackingSlider.setEnabled(editable)
        self.playVideoButton.setEnabled(editable)
        self.boundingLeftSpinBox.setEnabled(editable)
        self.boundingRightSpinBox.setEnabled(editable)
        self.boundingTopSpinBox.setEnabled(editable)
        self.boundingBottomSpinBox.setEnabled(editable)
        self.boundingBoxButton.setEnabled(editable)
        self.timeStartTextEdit.setEnabled(editable)
        self.saveTraceButton.setEnabled(editable)
        self.addTrialButton.setEnabled(editable)
        self.remTrialButton.setEnabled(editable)
        self.setStartButton.setEnabled(editable)
        self.setEndButton.setEnabled(editable)
        self.loadVideoButton.setEnabled(editable)
        self.loadParamButton.setEnabled(editable)

    # def start_clip_video(self):
    #     """Start clipping the next video"""
    #     trialN = self.trialdf.loc[self.currRow, 'Trial']
    #     start = self.trialdf.loc[self.currRow, 'Start']
    #     end = self.trialdf.loc[self.currRow, 'End']
    #
    #     # self.currRow = trialN
    #     currFileName = self.videoName + "_t" + trialN + self.videoExt
    #     output_path = os.path.join(self.folderPath, currFileName)
    #
    #     # # get number of frames for progress bar
    #     # startTS = start.split(":")
    #     startSec = get_seconds_from_time(start)
    #     # float(startTS[0]) * 60 * 60 + float(startTS[1]) * 60 + float(startTS[2])
    #     # endTS = end.split(":")
    #     endSec = get_seconds_from_time(end)
    #     # float(endTS[0]) * 60 * 60 + float(endTS[1]) * 60 + float(endTS[2])
    #
    #     self.clipDur = endSec - startSec
    #
    #     if self.fullStart is None:
    #         self.fullStart = get_seconds_from_time(self.trialdf['Start'].iloc[0])
    #
    #     command = [
    #         "-y",
    #         "-i", self.outputTemp,
    #         "-ss", str(round(startSec - self.fullStart, 4)),
    #         "-to", str(round(endSec - self.fullStart, 4)),
    #         "-map", "0:v",  # To remove any extra streams
    #         "-map", "0:a?",
    #         "-c", "copy",
    #         # "-avoid_negative_ts", "make_zero",
    #         # "-reset_timestamps", "1",
    #         output_path
    #     ]
    #     self.update_status(f'Step 2: Clipping trial {trialN} of {self.trialCount} (Clip {self.currRow})')
    #     if self.progressBar:
    #         progress = 100 * (int(trialN) / self.trialCount)
    #         self.progressBar.setValue(progress)
    #     self.process.setProgram('ffmpeg')
    #     self.process.setArguments(command)
    #     self.process.start()


app = QApplication(sys.argv)

window = MainWindow()
window.show()
# Center window on screen
frame_geom = window.frameGeometry()
center_point = app.primaryScreen().availableGeometry().center()
frame_geom.moveCenter(center_point)
window.move(frame_geom.topLeft())

app.exec()
