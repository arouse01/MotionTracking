import sys
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import QThread, Signal, Slot, Qt, QEvent, QCoreApplication, QMetaObject, QSize
from PySide6.QtGui import QFont, QColor, QBrush
from PySide6.QtWidgets import (QGridLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QSlider, QWidget, QLineEdit,
                               QTableWidget, QHeaderView, QTableWidgetItem, QAbstractItemView, QStatusBar)
# from videoAnalysis_ui import UiMainWindow
import cv2  # via opencv-python AND opencv-contrib-python (for other trackers)
import numpy as np
import pandas as pd  # for exporting the trial times
import subprocess
import os
# import pyqtgraph as pg


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
        self.gridLayout = None

        self.fileLayout = None
        self.pathLabel = None
        self.pathText = None
        self.loadButton = None

        self.videoFrame = None

        self.trialMarkerTable = None

        self.trackingLayout = None
        self.timeStartLabel = None
        self.timeEndLabel = None
        self.trackingSlider = None

        self.settingLayout = None
        self.playVideoButton = None
        self.setStartButton = None
        self.setEndButton = None

        self.addTrialButton = None
        self.remTrialButton = None
        self.boundingBoxButton = None
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
        ┏━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━┓
        ┃ pathLabel    │ pathTextEdit                                │ loadButton          │ loadButton ┃
        ┡━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━┷━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━┩
        │ videoFrame                                             │ trialMarkerTable                     │
        │                                                        │                                      │
        ┢━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━╅───────────────────┬──────────────────┤
        ┃ timeStartLabel │ trackingSlider       │ timeEndLabel   ┃ newTrialButton    │ remTrialButton   │
        ┣━━━━━━━━━━━━━━━━┷━┯━━━━━━━━━━━━━━━━━━┯━┷━━━━━━━━━━━━━━━━╉───────────────────┼──────────────────┤
        ┃ playButton       │ setStartButton   │ setEndButton     ┃ cropButton        │ exportButton     │   
        ┗━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┹───────────────────┴──────────────────┘
        
        

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

        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")

        self.fileLayout = QHBoxLayout(self.centralwidget)

        self.pathLabel = QLabel(self.centralwidget)
        self.pathLabel.setObjectName(u"pathLabel")
        self.pathLabel.setSizePolicy(sizePolicy_Fixed)
        self.pathLabel.setMinimumSize(QSize(50, 24))
        self.pathLabel.setMaximumSize(QSize(50, 24))
        self.pathLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fileLayout.addWidget(self.pathLabel)

        self.pathText = QLineEdit(self.centralwidget)
        self.pathText.setObjectName(u"pathText")
        self.pathText.setSizePolicy(sizePolicy_Ex)
        self.pathText.setMinimumSize(QSize(84, 24))
        self.pathText.setMaximumHeight(24)
        # self.pathText.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fileLayout.addWidget(self.pathText)

        self.loadButton = QPushButton(self.centralwidget)
        self.loadButton.setObjectName(u"pathLabel")
        self.loadButton.setSizePolicy(sizePolicy_Fixed)
        self.loadButton.setMinimumSize(QSize(84, 24))
        self.loadButton.setMaximumSize(QSize(84, 24))
        self.fileLayout.addWidget(self.loadButton)

        self.gridLayout.addLayout(self.fileLayout, 0, 0, 1, 3)

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
        self.gridLayout.addWidget(self.videoFrame, 1, 0, 1, 1)

        self.trialMarkerTable = QTableWidget(self.centralwidget)
        self.trialMarkerTable.setObjectName(u"trialMarkerTable")
        self.trialMarkerTable.setSizePolicy(sizePolicy_Ex)
        self.trialMarkerTable.setMinimumSize(QSize(200, 100))
        self.trialMarkerTable.setMaximumSize(QSize(1000, 1000))
        self.gridLayout.addWidget(self.trialMarkerTable, 1, 1, 1, 2)

        self.trackingLayout = QHBoxLayout(self.centralwidget)

        self.timeStartLabel = QLabel(self.centralwidget)
        self.timeStartLabel.setObjectName(u"startTimestampLabel")
        self.timeStartLabel.setSizePolicy(sizePolicy_Fixed)
        self.timeStartLabel.setMinimumSize(QSize(84, 30))
        self.timeStartLabel.setMaximumSize(QSize(84, 30))
        self.timeStartLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trackingLayout.addWidget(self.timeStartLabel)

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

        self.gridLayout.addLayout(self.trackingLayout, 2, 0, 1, 1)

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

        self.gridLayout.addLayout(self.settingLayout, 3, 0, 1, 1)

        self.addTrialButton = QPushButton(self.centralwidget)
        self.addTrialButton.setObjectName(u"addTrialButton")
        self.addTrialButton.setSizePolicy(sizePolicy_minEx_max)
        self.addTrialButton.setMinimumHeight(30)
        self.addTrialButton.setMaximumSize(QSize(120, 30))
        self.gridLayout.addWidget(self.addTrialButton, 2, 1, 1, 1)

        self.remTrialButton = QPushButton(self.centralwidget)
        self.remTrialButton.setObjectName(u"remTrialButton")
        self.remTrialButton.setSizePolicy(sizePolicy_minEx_max)
        self.remTrialButton.setMinimumHeight(30)
        self.remTrialButton.setMaximumSize(QSize(120, 30))
        self.gridLayout.addWidget(self.remTrialButton, 2, 2, 1, 1)

        self.boundingBoxButton = QPushButton(self.centralwidget)
        self.boundingBoxButton.setObjectName(u"boundingBoxButton")
        self.boundingBoxButton.setSizePolicy(sizePolicy_minEx_max)
        self.boundingBoxButton.setMinimumHeight(30)
        self.boundingBoxButton.setMaximumSize(QSize(120, 30))
        self.gridLayout.addWidget(self.boundingBoxButton, 3, 1, 1, 1)

        self.saveTraceButton = QPushButton(self.centralwidget)
        self.saveTraceButton.setObjectName(u"saveTraceButton")
        self.saveTraceButton.setSizePolicy(sizePolicy_minEx_max)
        self.saveTraceButton.setMinimumHeight(30)
        self.saveTraceButton.setMaximumSize(QSize(120, 30))
        self.gridLayout.addWidget(self.saveTraceButton, 3, 2, 1, 1)

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
        # self.gridLayout.addWidget(self.traceGraph, 3, 0, 2, 5)

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
        self.loadButton.setText(QCoreApplication.translate("mainwindow", u"Load", None))
        self.playVideoButton.setText(QCoreApplication.translate("mainwindow", u"Play", None))
        self.setStartButton.setText(QCoreApplication.translate("mainwindow", u"Set Start", None))
        self.setEndButton.setText(QCoreApplication.translate("mainwindow", u"Set End", None))
        self.addTrialButton.setText(QCoreApplication.translate("mainwindow", u"Add Trial", None))
        self.remTrialButton.setText(QCoreApplication.translate("mainwindow", u"Delete Trial", None))
        self.saveTraceButton.setText(QCoreApplication.translate("mainwindow", u"Split Video", None))
        self.boundingBoxButton.setText(QCoreApplication.translate("mainwindow", u"Crop Video", None))
        self.timeStartLabel.setText(QCoreApplication.translate("mainwindow", u"0:00:00.0000", None))
        self.timeEndLabel.setText(QCoreApplication.translate("mainwindow", u"0:00:00.0000", None))
    # retranslateUi


class MainWindow(QtWidgets.QMainWindow, UiMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_ui(self)
        self.setWindowTitle("Bobbing Trial Splitting")

        self.loadButton.clicked.connect(self.load_video)
        self.trialMarkerTable.cellClicked.connect(self.select_trial)
        self.trackingSlider.setTracking(True)
        self.trackingSlider.valueChanged.connect(self.user_move_slider)
        self.trackingSlider.sliderReleased.connect(self.adjust_trackingslider)
        self.addTrialButton.clicked.connect(self.trial_add)
        self.remTrialButton.clicked.connect(self.trial_rem)
        self.playVideoButton.clicked.connect(self.video_play)
        self.setStartButton.clicked.connect(lambda: self.set_trial_start())
        self.setEndButton.clicked.connect(lambda: self.set_trial_end())
        self.boundingBoxButton.clicked.connect(lambda: self.set_box())
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
        self.boundingBoxButton.setEnabled(False)
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

        self.bbox = None  # current position of tracker
        self.bboxOriginal = None  # original position of tracker
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

    def load_video(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video')
        if fileName[0]:
            fileName = fileName[0]
            self.videoPath = fileName
            self.pathText.setText(self.videoPath)
            self.videoName = os.path.splitext(os.path.split(self.videoPath)[1])[0]
            self.videoExt = os.path.splitext(os.path.split(self.videoPath)[1])[1]
            self.cap = cv2.VideoCapture(fileName)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.critical(self, "Error", "Could not read video file",
                                               QtWidgets.QMessageBox.StandardButton.Ok)
            else:
                # load first frame
                ok, frame = self.cap.read()
                if not ok:
                    QtWidgets.QMessageBox.critical(self, "Error", "Could not read video file",
                                                   QtWidgets.QMessageBox.StandardButton.Ok)
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
                    self.boundingBoxButton.setEnabled(True)
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

                    self.videoFrame.installEventFilter(self)

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
            self.cropWidth = self.bbox[2]
            self.cropHeight = self.bbox[3]
            self.cropX = self.bbox[0]
            self.cropY = self.bbox[1]
            # self.playVideoButton.setEnabled(True)
            #
            # p1 = (int(self.bbox[0]), int(self.bbox[1]))
            # p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
            # cv2.rectangle(frameCopy, p1, p2, (255, 0, 0), 2, 1)
            # self.update_image(frameCopy, self.frameCurrentNumber)
            # self.tlx1[self.frameCurrentNumber] = int(self.bbox[0])
            # self.tlx2[self.frameCurrentNumber] = int(self.bbox[0] + self.bbox[2])
            # self.tly1[self.frameCurrentNumber] = self.vidHeight - int(self.bbox[1])
            # self.tly2[self.frameCurrentNumber] = self.vidHeight - int(self.bbox[1] + self.bbox[3])
            # self.tlxMid[self.frameCurrentNumber] = int(self.bbox[1] + self.bbox[3] / 2)
            # self.tlyMid[self.frameCurrentNumber] = int(self.bbox[0] + self.bbox[2] / 2)
            #
            # self.bboxOriginal = self.bbox  # capture ROI location
            # self.bboxImage = self.frameCurrent[self.bbox[1]:self.bbox[1]+self.bbox[3],
            #                                    self.bbox[0]:self.bbox[0]+self.bbox[2]]
        else:
            print("No ROI selected")

    def set_trial_start(self):
        startTime = QTableWidgetItem(self.timeStartLabel.text())
        self.trialMarkerTable.setItem(self.trial, 1, startTime)

    def set_trial_end(self):
        endTime = QTableWidgetItem(self.timeStartLabel.text())
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

    # def update_table(self):
    #     self.trialMarkerTable
    #     pass

    # def load_frame(self, target_frame):
    #     # set the tracking slider to the specified frame, including loading the new image
    #     self.trackingSlider.setValue(target_frame)
    #     newTimeStamp = self.get_time_from_frame(target_frame)
    #     self.timeStartLabel.setText(newTimeStamp)
    #     self.frameCurrentNumber = target_frame
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
    #         ret, cv_img = self.cap.read()
    #         if ret:
    #             self.update_image(cv_img, self.frameCurrentNumber)

    def update_frame_number(self, target_frame):
        # set the tracking slider to the specified frame without triggering a new image load (i.e. when video is
        # playing)
        self.trackingSlider.blockSignals(True)
        self.trackingSlider.setValue(target_frame)
        self.trackingSlider.blockSignals(False)

        newTimeStamp = self.get_time_from_frame(target_frame)
        self.timeStartLabel.setText(newTimeStamp)
        self.frameCurrentNumber = target_frame

    def adjust_trackingslider(self):
        # move the tracking slider directly, load the frame at the slider position
        targetFrame = self.trackingSlider.value()
        newTimeStamp = self.get_time_from_frame(targetFrame)
        self.timeStartLabel.setText(newTimeStamp)
        self.frameCurrentNumber = targetFrame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, targetFrame - 1)
        ret, cv_img = self.cap.read()
        if ret:
            self.update_image(cv_img, self.frameCurrentNumber)
        # self.thread.select(self.frameCurrentNumber)
        # self.load_frame(self.frameCurrentNumber)

    def user_move_slider(self):
        # update the timestamp based on slider position without loading a new frame (while dragging the slider)
        targetFrame = self.trackingSlider.value()
        newTimeStamp = self.get_time_from_frame(targetFrame)
        self.timeStartLabel.setText(newTimeStamp)

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

                    if self.bbox and all(x > 0 for x in self.bbox):
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

    def video_play(self):
        # update the play button
        self.playVideoButton.setText("Pause")
        self.playVideoButton.clicked.disconnect(self.video_play)
        self.playVideoButton.clicked.connect(self.video_stop)
        # disable the buttons that could change the frame
        self.trackingSlider.setEnabled(False)

        # # create the tracker
        # self.select_tracker("MIL")
        # _ = self.tracker.init(self.frameCurrent, self.bbox)

        # # create the video capture thread
        self.thread = VideoThread(self.cap)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_tracker)
        # start the thread
        self.thread.start()

    def video_stop(self):
        self.playVideoButton.clicked.disconnect(self.video_stop)
        self.playVideoButton.clicked.connect(self.video_play)
        self.playVideoButton.setText("Analyze")
        self.thread.stop()

        self.trackingSlider.setEnabled(True)
        # self.thread.change_pixmap_signal.disconnect(self.update_tracker)

    @Slot(np.ndarray)
    def update_image(self, cv_img, frame_number):
        """Updates the image_label with a new opencv image"""
        if self.bbox and all(x > 0 for x in self.bbox):
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

    def update_tracker(self, frame, frame_number):

        if frame_number <= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):

            # Display result
            self.update_image(frame, frame_number)

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


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
