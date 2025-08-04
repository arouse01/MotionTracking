import sys
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import QThread, Signal, Slot, Qt, QEvent, QCoreApplication, QMetaObject, QSize
from PySide6.QtGui import QFont, QColor, QBrush
from PySide6.QtWidgets import (QGridLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QSlider, QWidget, QLineEdit,
                               QTableWidget, QHeaderView, QTableWidgetItem, QAbstractItemView, QStatusBar, QSpinBox,
                               QAbstractSpinBox, QFrame)
# from videoAnalysis_ui import UiMainWindow
import cv2  # via opencv-python AND opencv-contrib-python (for other trackers)
import numpy as np
import pandas as pd  # for exporting the trial times
import subprocess
import os
# import av  # for extracting audio from video - TOO SLOW, optimized for reading small pieces but not bulk
from moviepy import VideoFileClip
import pyqtgraph as pg  # for graphing the audio


# Note: to build the exe, pyinstaller is required. Once installed, go to Windows terminal, navigate to folder with
# target script, and enter:
# > python -m PyInstaller main.py -n BobbingAnalysis
# where -n specifies the resulting exe name

# # Sources:
# https://github.com/Google-Developer-Student-Clubs-Guelph/GDSCHacksOpenCVWorkshop
# https://learnopencv.com/object-tracking-using-opencv-cpp-python/
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


class UiMainWindow(object):

    def __init__(self):
        self.centralwidget = None
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
        font10 = QFont()
        font10.setPointSize(10)
        font11 = QFont()
        font11.setPointSize(11)
        font11Bold = QFont()
        font11Bold.setPointSize(11)
        font11Bold.setBold(True)
        font12Bold = QFont()
        font12Bold.setPointSize(12)
        font12Bold.setBold(True)
        font11Under = QFont()
        font11Under.setPointSize(11)
        font11Under.setUnderline(True)

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
        mainwindow.setStatusBar(self.statusbar)

        self.controlGridLayout = QGridLayout(self.centralwidget)
        self.controlGridLayout.setObjectName(u"controlGridLayout")

        self.fileLayout = QHBoxLayout(self.centralwidget)

        self.loadVideoButton = QPushButton(self.centralwidget)
        self.loadVideoButton.setObjectName(u"loadButton")
        self.loadVideoButton.setSizePolicy(sizePolicy_Fixed)
        self.loadVideoButton.setMinimumSize(QSize(84, 24))
        self.loadVideoButton.setMaximumSize(QSize(84, 24))
        self.fileLayout.addWidget(self.loadVideoButton)

        self.pathLabel = QLabel(self.centralwidget)
        self.pathLabel.setObjectName(u"pathLabel")
        self.pathLabel.setSizePolicy(sizePolicy_minEx_max)
        self.pathLabel.setMinimumSize(QSize(50, 24))
        self.pathLabel.setMaximumHeight(24)
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
        self.loadParamButton.setMinimumSize(QSize(50, 24))
        self.loadParamButton.setMaximumHeight(24)
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
        self.controlGridLayout.addWidget(self.videoFrame, 1, 0, 2, 1)

        self.audioFrame = pg.PlotWidget()
        self.audioFrame.setObjectName(u"audioFrame")
        self.audioFrame.setSizePolicy(sizePolicy_minEx_max)
        # self.audioFrame.setScaledContents(False)
        self.audioFrame.setMinimumHeight(100)
        self.audioFrame.setMaximumHeight(100)
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
        self.boundingLeftLabel.setMinimumSize(QSize(50, 24))
        self.boundingLeftLabel.setMaximumSize(QSize(50, 24))
        self.boundingLeftLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingLeftLabel, 0, 0, 1, 1)

        self.boundingLeftSpinBox = QSpinBox(self.centralwidget)
        self.boundingLeftSpinBox.setObjectName(u"boundingLeftSpinBox")
        self.boundingLeftSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingLeftSpinBox.setMinimumSize(QSize(60, 24))
        self.boundingLeftSpinBox.setMaximumSize(QSize(60, 24))
        self.boundingLeftSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingLeftSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingLeftSpinBox, 0, 1, 1, 1)

        self.boundingRightLabel = QLabel(self.centralwidget)
        self.boundingRightLabel.setObjectName(u"boundingRightLabel")
        self.boundingRightLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingRightLabel.setMinimumSize(QSize(50, 24))
        self.boundingRightLabel.setMaximumSize(QSize(50, 24))
        self.boundingRightLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingRightLabel, 0, 2, 1, 1)

        self.boundingRightSpinBox = QSpinBox(self.centralwidget)
        self.boundingRightSpinBox.setObjectName(u"boundingRightSpinBox")
        self.boundingRightSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingRightSpinBox.setMinimumSize(QSize(60, 24))
        self.boundingRightSpinBox.setMaximumSize(QSize(60, 24))
        self.boundingRightSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingRightSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingRightSpinBox, 0, 3, 1, 1)

        self.boundingTopLabel = QLabel(self.centralwidget)
        self.boundingTopLabel.setObjectName(u"boundingTopLabel")
        self.boundingTopLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingTopLabel.setMinimumSize(QSize(50, 24))
        self.boundingTopLabel.setMaximumSize(QSize(50, 24))
        self.boundingTopLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingTopLabel, 1, 0, 1, 1)

        self.boundingTopSpinBox = QSpinBox(self.centralwidget)
        self.boundingTopSpinBox.setObjectName(u"boundingTopSpinBox")
        self.boundingTopSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingTopSpinBox.setMinimumSize(QSize(60, 24))
        self.boundingTopSpinBox.setMaximumSize(QSize(60, 24))
        self.boundingTopSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingTopSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingTopSpinBox, 1, 1, 1, 1)

        self.boundingBottomLabel = QLabel(self.centralwidget)
        self.boundingBottomLabel.setObjectName(u"boundingBottomLabel")
        self.boundingBottomLabel.setSizePolicy(sizePolicy_Fixed)
        self.boundingBottomLabel.setMinimumSize(QSize(50, 24))
        self.boundingBottomLabel.setMaximumSize(QSize(50, 24))
        self.boundingBottomLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.boundingGridLayout.addWidget(self.boundingBottomLabel, 1, 2, 1, 1)

        self.boundingBottomSpinBox = QSpinBox(self.centralwidget)
        self.boundingBottomSpinBox.setObjectName(u"boundingBottomSpinBox")
        self.boundingBottomSpinBox.setSizePolicy(sizePolicy_Fixed)
        self.boundingBottomSpinBox.setMinimumSize(QSize(60, 24))
        self.boundingBottomSpinBox.setMaximumSize(QSize(60, 24))
        self.boundingBottomSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundingBottomSpinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.boundingGridLayout.addWidget(self.boundingBottomSpinBox, 1, 3, 1, 1)

        self.boundingBoxButton = QPushButton(self.centralwidget)
        self.boundingBoxButton.setObjectName(u"boundingBoxButton")
        self.boundingBoxButton.setSizePolicy(sizePolicy_minEx_max)
        self.boundingBoxButton.setMinimumHeight(24)
        self.boundingBoxButton.setMaximumHeight(24)
        self.boundingGridLayout.addWidget(self.boundingBoxButton, 2, 0, 1, 4)

        self.controlGridLayout.addWidget(self.boundingGridFrame, 1, 1, 1, 2)

        self.trialMarkerTable = QTableWidget(self.centralwidget)
        self.trialMarkerTable.setObjectName(u"trialMarkerTable")
        self.trialMarkerTable.setSizePolicy(sizePolicy_Ex)
        self.trialMarkerTable.setMinimumSize(QSize(200, 100))
        self.trialMarkerTable.setMaximumSize(QSize(1000, 1000))
        self.controlGridLayout.addWidget(self.trialMarkerTable, 2, 1, 2, 2)

        self.trackingLayout = QHBoxLayout(self.centralwidget)

        self.timeStartTextEdit = QLineEdit(self.centralwidget)
        self.timeStartTextEdit.setObjectName(u"startTimestampLabel")
        self.timeStartTextEdit.setSizePolicy(sizePolicy_Fixed)
        self.timeStartTextEdit.setMinimumSize(QSize(84, 30))
        self.timeStartTextEdit.setMaximumSize(QSize(84, 30))
        self.timeStartTextEdit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trackingLayout.addWidget(self.timeStartTextEdit)

        self.trackingSlider = QSlider(self.centralwidget)
        self.trackingSlider.setObjectName(u"trackingSlider")
        self.trackingSlider.setOrientation(Qt.Orientation.Horizontal)
        self.trackingSlider.setSizePolicy(sizePolicy_minEx_max)
        self.trackingSlider.setMaximumSize(QSize(16777215, 30))
        self.trackingLayout.addWidget(self.trackingSlider)

        self.timeEndLabel = QLabel(self.centralwidget)
        self.timeEndLabel.setObjectName(u"endTimestampLabel")
        self.timeEndLabel.setSizePolicy(sizePolicy_Fixed)
        self.timeEndLabel.setMinimumSize(QSize(84, 30))
        self.timeEndLabel.setMaximumSize(QSize(84, 30))
        self.timeEndLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trackingLayout.addWidget(self.timeEndLabel)

        self.controlGridLayout.addLayout(self.trackingLayout, 4, 0, 1, 1)

        self.settingLayout = QHBoxLayout(self.centralwidget)

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
        self.addTrialButton.setMinimumHeight(30)
        self.addTrialButton.setMaximumSize(QSize(120, 30))
        self.controlGridLayout.addWidget(self.addTrialButton, 4, 1, 1, 1)

        self.remTrialButton = QPushButton(self.centralwidget)
        self.remTrialButton.setObjectName(u"remTrialButton")
        self.remTrialButton.setSizePolicy(sizePolicy_minEx_max)
        self.remTrialButton.setMinimumHeight(30)
        self.remTrialButton.setMaximumSize(QSize(120, 30))
        self.controlGridLayout.addWidget(self.remTrialButton, 4, 2, 1, 1)

        self.saveTraceButton = QPushButton(self.centralwidget)
        self.saveTraceButton.setObjectName(u"saveTraceButton")
        self.saveTraceButton.setSizePolicy(sizePolicy_minEx_max)
        self.saveTraceButton.setMinimumHeight(30)
        self.saveTraceButton.setMaximumHeight(30)
        self.controlGridLayout.addWidget(self.saveTraceButton, 5, 1, 1, 2)

        # self.traceGraph = pg.PlotWidget()  # QtCharts.QChartView(self.centralwidget)
        # self.traceGraph.setObjectName(u"traceGraph")
        # self.traceGraph.setMinimumHeight(50)
        # self.traceGraph.setMaximumHeight(150)
        # self.traceGraph.setSizePolicy(sizePolicy_minEx_max)
        # self.traceGraph.setBackground("w")
        # self.traceGraph.getPlotItem().hideAxis('bottom')
        # self.traceGraph.getPlotItem().hideAxis('left')
        # self.traceGraph.setMouseEnabled(x=False, y=False)  # Disable mouse panning & zooming
        # self.traceGraph.hideButtons()  # Disable corner auto-scale button
        # self.traceGraph.getPlotItem().setMenuEnabled(False)  # Disable right-click context menu
        #
        # self.controlGridLayout.addWidget(self.traceGraph, 3, 0, 2, 5)

        mainwindow.setCentralWidget(self.centralwidget)

        # self.statusbar = QStatusBar(mainwindow)
        # self.statusbar.setObjectName(u"statusbar")
        # mainwindow.setStatusBar(self.statusbar)

        self.retranslate_ui(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)

    # setupUi

    def retranslate_ui(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("mainwindow", u"mainwindow", None))
        self.pathLabel.setText(QCoreApplication.translate("mainwindow", u"File", None))
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
    # retranslateUi


class MainWindow(QtWidgets.QMainWindow, UiMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_ui(self)
        self.setWindowTitle("Bobbing Trial Splitting")

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
        self.bboxPainter = QtGui.QPainter(self.videoFrame)

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

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def eventFilter(self, widget, event):
        # bug: after loading a frame, it resizes slightly, which fires the event, which resizes it,
        # which fires the event, and it gets stuck in an expanding loop
        # RESOLVED: seems to have to do with the frame:
        # https://stackoverflow.com/questions/42833511/qt-how-to-create-image-that-scale-with-window-and-keeps-aspect-ratio

        # https://stackoverflow.com/questions/21041941/how-to-autoresize-qlabel-pixmap-keeping-ratio-without-using-classes/21053898#21053898
        if event.type() == QEvent.Type.Resize and widget is self.videoFrame:
            self.videoFrame.setPixmap(self.videoFrame.pixmap.scaled(self.videoFrame.width(), self.videoFrame.height(),
                                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation))

            # resize the window
            self.resize(self.sizeHint().width(), self.sizeHint().height())
            return True
        return QtWidgets.QMainWindow.eventFilter(self, widget, event)

    def closeEvent(self, event):
        # if thread hasn't initialized yet, then the stop and close methods don't exist yet so we need to check
        if hasattr(self.thread, 'stop'):
            self.thread.stop()
        if hasattr(self.thread, 'close'):
            self.thread.close()

    def load_video(self, filepath=None):
        if filepath is None:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video')
            if fileName[0]:
                filepath = fileName[0]

        self.videoPath = filepath
        self.pathLabel.setText(self.videoPath)
        self.videoName = os.path.splitext(os.path.split(self.videoPath)[1])[0]
        self.videoExt = os.path.splitext(os.path.split(self.videoPath)[1])[1]
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Could not read video file",
                                           QtWidgets.QMessageBox.StandardButton.Ok)
        else:
            # load first frame
            ok, frame = self.cap.read()
            if not ok:
                QtWidgets.QMessageBox.critical(self, "Error", "Could not read video file",
                                               QtWidgets.QMessageBox.StandardButton.Ok)
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
                # OLD METHOD - pyav is too slow because it's optimized for reading audio packets rather than bulk import
                # with av.open(self.videoPath) as container:
                #     audio_stream = next(s for s in container.streams if s.type == 'audio')
                #     audio_frames = []
                #
                #     fs = audio_stream.sample_rate
                #
                #     for packet in container.demux(audio_stream):
                #         for frame in packet.decode():
                #             audio_frames.append(frame.to_ndarray())
                # waveform = np.concatenate(audio_frames, axis=1)
                videoClip = VideoFileClip(self.videoPath)
                fs = videoClip.audio.fps
                waveform = videoClip.audio.to_soundarray(fps=fs)
                waveform = waveform.mean(axis=1)  # convert from stereo to mono
                dsFactor = 10
                self.audioWaveform = waveform[::dsFactor]  # downsample for plotting efficacy

                nSamp = len(self.audioWaveform)
                time = np.linspace(0, nSamp/(fs/dsFactor), num=nSamp, endpoint=False)
                self.audioWavePlot = self.audioFrame.plot(time, self.audioWaveform)

                self.audioTrackerLine = pg.InfiniteLine(0, pen=pg.mkPen('y', width=1))
                self.audioFrame.addItem(self.audioTrackerLine)

                # self.audioTrackerLine = self.audioFrame.plot([0, 0], [-10, 10], pen=pg.mkPen('y', width=1))

                self.videoFrame.installEventFilter(self)

                return True

    def load_settings(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Settings File')
        if fileName[0]:
            fileName = fileName[0]
            settingsdf = pd.read_csv(fileName)
            # videoName = os.path.splitext(os.path.split(fileName)[1])[0]
            videoPath = settingsdf[['Video']].tail(1).values[0][0]

            # # load video
            # videoFolder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Video Folder')
            # if videoFolder:
            #     videoPath = os.path.join(videoFolder, videoName)

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
                    # trialNumItem = QTableWidgetItem(Qt.DisplayRole, trialdf[['Trial']].iloc[i])
                    # trialNumItem.setData(Qt.DisplayRole, trialdf[['Trial']].iloc[i])
                    self.trialMarkerTable.setItem(i, 0, QTableWidgetItem(
                        Qt.DisplayRole, trialdf[['Trial']].iloc[i].values[0]))
                    # fill remaining cells with an actual item
                    self.trialMarkerTable.setItem(i, 1, QTableWidgetItem(trialdf[['Start']].iloc[i].values[0]))
                    self.trialMarkerTable.setItem(i, 2, QTableWidgetItem(trialdf[['End']].iloc[i].values[0]))
                    self.trialCount += 1

                    # startTime = QTableWidgetItem(self.timeStartTextEdit.text())
                    # self.trialMarkerTable.setItem(self.trial, 1, startTime)

    def set_box(self):
        # self.bbox = (1261, 586, 60, 72)
        frameCopy = self.frameCurrent  # make a copy of the frame and add the rectangle to it rather than the original

        # set up instruction text
        boxInstr = ["Select object to track with", "left mouse button, press", "Enter when finished"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(boxInstr):
            textSize = cv2.getTextSize(line, font, .8, 2)[0]
            textX = textSize[1]  # (frameCopy.shape[1] - textSize[0]) / 2
            textY = textSize[1] * 2 * (i+1)
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
        # update crop/bounding values after bbox is updated
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
        self.trial = row
        # print(row)

    def trial_add(self):
        lastTrial = self.trialCount-1
        self.trialMarkerTable.insertRow(lastTrial)
        trialNumItem = QTableWidgetItem()
        trialNumItem.setData(Qt.DisplayRole, self.trialCount)
        self.trialMarkerTable.setItem(lastTrial, 0, trialNumItem)
        # fill remaining cells with an actual item
        self.trialMarkerTable.setItem(lastTrial, 1, QTableWidgetItem())
        self.trialMarkerTable.setItem(lastTrial, 2, QTableWidgetItem())
        self.trialCount += 1

    def trial_rem(self):
        if self.trialCount > 1:
            self.trialMarkerTable.removeRow(self.trial)
            self.trialCount -= 1
            # renumber trials after deletion
            for row in range(self.trialCount-1):  # trialCount - 1 because trialCount is not 0-indexed
                self.trialMarkerTable.item(row, 0).setData(Qt.DisplayRole, row+1)

    def video_play(self):
        # update the play button
        self.playVideoButton.setText("Pause")
        self.playVideoButton.clicked.disconnect(self.video_play)
        self.playVideoButton.clicked.connect(self.video_stop)
        # disable the buttons that could change the frame
        self.trackingSlider.setEnabled(False)

        # create the video capture thread
        self.thread = VideoThread(self.cap)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_tracker)
        # start the thread
        self.thread.start()

    def video_stop(self):
        self.playVideoButton.clicked.disconnect(self.video_stop)
        self.playVideoButton.clicked.connect(self.video_play)
        self.playVideoButton.setText("Play")
        self.thread.stop()

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
        self.user_dragging = True

    def drag_off(self):
        self.user_dragging = False
        self.adjust_trackingslider()

    def update_frame_number(self, targetFrame):
        """Update tracking slider and audio tracker while video is playing (so don't trigger a new image load)"""
        self.trackingSlider.blockSignals(True)
        self.trackingSlider.setValue(targetFrame)
        self.trackingSlider.blockSignals(False)

        self.update_timestamp(targetFrame)
        # newTimeStamp = self.get_time_from_frame(targetFrame)
        # self.timeStartTextEdit.setText(newTimeStamp)
        self.frameCurrentNumber = targetFrame
        self.update_audio_tracker(targetFrame)

    def adjust_trackingslider(self):
        """after dragging, user releases the tracking slider. Load the frame at the slider position"""
        if not self.user_dragging:
            targetFrame = self.trackingSlider.value()
            # newTimeStamp = self.get_time_from_frame(targetFrame)
            # self.timeStartTextEdit.setText(newTimeStamp)
            self.frameCurrentNumber = targetFrame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
            ret, cv_img = self.cap.read()
            if ret:
                self.update_image(cv_img, self.frameCurrentNumber)

    def user_move_slider(self, targetFrame):
        """user is dragging the slider - update the timestamp based on slider position without loading a new frame"""
        # targetFrame = self.trackingSlider.value()
        self.update_timestamp(targetFrame)
        # newTimeStamp = self.get_time_from_frame(targetFrame)
        # self.timeStartTextEdit.setText(newTimeStamp)
        self.update_audio_tracker(targetFrame)

    def user_set_time(self):
        """user set the timestamp - set the slider, load the new frame"""
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

        # # changing trackingSlider value triggers user_move_slider, which updates the text and audio tracker
        # self.trackingSlider.setValue(targetFrame)

        # now load the frame at the slider position
        self.frameCurrentNumber = targetFrame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
        ret, cv_img = self.cap.read()
        if ret:
            self.update_image(cv_img, self.frameCurrentNumber)

    def update_tracker(self, frame, frame_number):
        #
        if frame_number <= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            # Display result
            self.update_image(frame, frame_number)

    def update_audio_tracker(self, targetFrame):
        if self.audioTrackerLine:
            newTS = targetFrame / self.videoFrameRate
            # self.audioTrackerLine.setData([newTS, newTS], [-10, 10])
            self.audioTrackerLine.setPos(newTS)

    def update_timestamp(self, targetFrame):
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
            cv2.rectangle(cv_img, (self.cropX, self.cropY), (self.cropX + self.cropWidth, self.cropY+self.cropHeight),
                          (255, 0, 0), 3)

        self.frameCurrent = cv_img
        frame = self.convert_cv_qt(cv_img)
        self.videoFrame.setPixmap(frame)
        self.videoFrame.pixmap = QtGui.QPixmap(frame)
        self.update_frame_number(frame_number)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.videoFrame.width(), self.videoFrame.height(),
                                        Qt.AspectRatioMode.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def get_time_from_frame(self, framenumber):
        # return time as string (with commented lines for returning as datetime)
        nSecondsRaw = framenumber/self.videoFrameRate
        nMin, nSec = divmod(nSecondsRaw, 60)
        nHour, nMin = divmod(nMin, 60)
        # nMicro = round((nSec % 1)*10**6)
        timeStr = f"{round(nHour)}:{round(nMin):02}:{nSec:07.4f}"  # seconds with 07 for len(nSec)
        # ts = datetime.time(hour=int(nHour), minute=int(nMin), second=int(nSec), microsecond=int(nMicro))
        # timeStr = ts.strftime('%H:%M:%S.%f')
        return timeStr

    def split_video(self):

        brushHighlight = QBrush(QColor('yellow'))
        brushReg = QBrush(QColor('white'))

        trialCount = self.trialMarkerTable.rowCount()
        # Validate that all trials have start and end before trying to clip/crop them
        trialsValid = True
        for row in range(trialCount):
            trialNItem = self.trialMarkerTable.item(row, 0)
            startItem = self.trialMarkerTable.item(row, 1)
            endItem = self.trialMarkerTable.item(row, 2)
            if not trialNItem.text():
                # self.statusbar.showMessage(f'Row {row + 1} missing trial number')
                trialsValid = False
                trialNItem.setBackground(brushHighlight)
            else:
                trialNItem.setBackground(brushReg)
            if not startItem.text():
                # self.statusbar.showMessage(f'Row {row+1} missing start time')
                trialsValid = False
                startItem.setBackground(brushHighlight)
            else:
                startItem.setBackground(brushReg)
            if not endItem.text():
                # self.statusbar.showMessage(f'Row {row+1} missing end time')
                trialsValid = False
                endItem.setBackground(brushHighlight)
            else:
                endItem.setBackground(brushReg)

        if trialsValid:
            folderPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Save to Folder')
            if folderPath:
                # loop through trials and export cropped clips
                columnHeaders = ['Trial', 'Start', 'End']
                trialdf = pd.DataFrame(columns=columnHeaders, index=range(trialCount))
                for row in range(trialCount):
                    trialN = self.trialMarkerTable.item(row, 0).text()
                    start = self.trialMarkerTable.item(row, 1).text()
                    end = self.trialMarkerTable.item(row, 2).text()
                    trialdf.loc[row, 'Trial'] = trialN
                    trialdf[row, 'Start'] = start
                    trialdf[row, 'End'] = end

                    currFileName = self.videoName + "_t" + trialN + self.videoExt
                    output_path = os.path.join(folderPath, currFileName)

                    if all(x > 0 for x in self.bbox):
                        # video is cropped
                        # TODO: add verification that ffmpeg is installed
                        command = (
                            f'ffmpeg -n -i "{self.videoPath}" -ss {start} -to {end} '
                            f"-filter:v crop={self.cropWidth}:{self.cropHeight}:{self.cropX}:{self.cropY} "
                            f'-c:a copy "{output_path}"'
                        )
                    else:
                        command = (
                            f'ffmpeg -n -i "{self.videoPath}" -ss {start} -to {end} '
                            f'-c:a copy "{output_path}"'
                        )
                    self.statusbar.showMessage(f'Clipping trial {trialN} of {trialCount}')
                    print(f'Clipping trial {trialN} of {trialCount}')
                    subprocess.call(command, shell=True)
                self.statusbar.showMessage(f'')

                # cropping box
                bboxdf = pd.DataFrame({'cropX': [self.cropX], 'cropY': [self.cropY],
                                       'cropWidth': [self.cropWidth], 'cropHeight': [self.cropHeight]})
                outdf = pd.concat([trialdf, bboxdf], ignore_index=True)

                # export trial times and names
                textFileName = self.videoName + "_trialTimes.csv"
                output_path = os.path.join(folderPath, textFileName)
                outdf.to_csv(output_path, index=False)

        else:
            self.statusbar.showMessage(f'Missing trial data')


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
