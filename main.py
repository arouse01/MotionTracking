import sys
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import QThread, Signal, Slot, Qt, QEvent, QCoreApplication, QMetaObject, QSize
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGridLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QSlider, QWidget
# from videoAnalysis_ui import UiMainWindow
import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg


# Note: to build the exe, pyinstaller is required. Once installed, go to Windows terminal, navigate to folder with
# target script, and enter:
# > python -m PyInstaller main.py -n BobbingAnalysis
# where -n specifies the resulting exe name

class VideoThread(QThread):
    # https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
    change_pixmap_signal = Signal(np.ndarray, int)

    def __init__(self, cap, tracker):
        super().__init__()
        self.run_flag = False
        self.cap = cap
        self.tracker = tracker

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
        self.videoFrame = None

        self.loadVideoButton = None
        self.playVideoButton = None
        self.boundingBoxButton = None
        self.saveTraceButton = None

        self.trackingLayout = None
        self.frameBackButton = None
        self.timeStartLabel = None
        self.frameForwardButton = None
        self.timeEndLabel = None
        self.trackingSlider = None

        self.traceGraph = None

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
        sizePolicy_Fixed = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy_Fixed.setHorizontalStretch(0)
        sizePolicy_Fixed.setVerticalStretch(0)

        sizePolicy_Ex = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy_Ex.setHorizontalStretch(0)
        sizePolicy_Ex.setVerticalStretch(0)

        sizePolicy_minEx_max = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        sizePolicy_minEx_max.setHorizontalStretch(0)
        sizePolicy_minEx_max.setVerticalStretch(0)

        sizePolicy_max = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy_max.setHorizontalStretch(0)
        sizePolicy_max.setVerticalStretch(0)

        sizePolicy_preferred = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy_preferred.setHorizontalStretch(0)
        sizePolicy_preferred.setVerticalStretch(0)
        # endregion Formatting templates

        if not mainwindow.objectName():
            mainwindow.setObjectName(u"Main Window")
        mainwindow.resize(800, 600)

        """
        # Layout schematic

         ─│└┴┘├┼┤┌┬┐
         ═║╟╫╢╚╧╝╔╤╗
        """
        self.centralwidget = QWidget(mainwindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")

        self.videoFrame = QLabel(self.centralwidget)
        self.videoFrame.setObjectName(u"videoFrame")
        self.videoFrame.setSizePolicy(sizePolicy_Ex)
        # self.videoFrame.setFrameShape(QFrame.StyledPanel)  # for whatever reason the frame seems to screw with the
        # image scaling!
        # https://stackoverflow.com/questions/42833511/qt-how-to-create-image-that-scale-with-window-and-keeps-aspect-ratio
        # self.videoFrame.setFrameShadow(QFrame.Raised)
        self.videoFrame.setScaledContents(False)
        self.videoFrame.setMaximumSize(QSize(80, 80))
        self.videoFrame.setMaximumSize(QSize(1000, 1000))
        self.gridLayout.addWidget(self.videoFrame, 0, 0, 1, 5)

        self.trackingLayout = QHBoxLayout(self.centralwidget)

        self.frameBackButton = QPushButton(self.centralwidget)
        self.frameBackButton.setObjectName(u"frameBackButton")
        self.frameBackButton.setSizePolicy(sizePolicy_Fixed)
        self.frameBackButton.setMaximumSize(QSize(30, 30))
        self.trackingLayout.addWidget(self.frameBackButton)

        self.timeStartLabel = QLabel(self.centralwidget)
        self.timeStartLabel.setObjectName(u"timestampLabel")
        self.timeStartLabel.setSizePolicy(sizePolicy_Fixed)
        self.timeStartLabel.setMinimumSize(QSize(84, 30))
        self.timeStartLabel.setMaximumSize(QSize(84, 30))
        self.timeStartLabel.setAlignment(Qt.AlignCenter)
        self.trackingLayout.addWidget(self.timeStartLabel)

        self.trackingSlider = QSlider(self.centralwidget)
        self.trackingSlider.setObjectName(u"trackingSlider")
        self.trackingSlider.setOrientation(Qt.Horizontal)
        self.trackingSlider.setSizePolicy(sizePolicy_minEx_max)
        self.trackingSlider.setMaximumSize(QSize(16777215, 30))
        self.trackingLayout.addWidget(self.trackingSlider)

        self.timeEndLabel = QLabel(self.centralwidget)
        self.timeEndLabel.setObjectName(u"timestampLabel")
        self.timeEndLabel.setSizePolicy(sizePolicy_Fixed)
        self.timeEndLabel.setMinimumSize(QSize(84, 30))
        self.timeEndLabel.setMaximumSize(QSize(84, 30))
        self.timeEndLabel.setAlignment(Qt.AlignCenter)
        self.trackingLayout.addWidget(self.timeEndLabel)

        self.frameForwardButton = QPushButton(self.centralwidget)
        self.frameForwardButton.setObjectName(u"frameForwardButton")
        self.frameForwardButton.setSizePolicy(sizePolicy_Fixed)
        self.frameForwardButton.setMaximumSize(QSize(30, 30))
        self.trackingLayout.addWidget(self.frameForwardButton)

        self.gridLayout.addLayout(self.trackingLayout, 1, 0, 1, 5)

        self.loadVideoButton = QPushButton(self.centralwidget)
        self.loadVideoButton.setObjectName(u"loadVideoButton")
        self.loadVideoButton.setSizePolicy(sizePolicy_minEx_max)
        self.loadVideoButton.setMaximumHeight(30)
        self.gridLayout.addWidget(self.loadVideoButton, 2, 0, 1, 1)

        self.playVideoButton = QPushButton(self.centralwidget)
        self.playVideoButton.setObjectName(u"playVideoButton")
        self.playVideoButton.setSizePolicy(sizePolicy_minEx_max)
        self.playVideoButton.setMaximumHeight(30)
        self.gridLayout.addWidget(self.playVideoButton, 2, 2, 1, 1)

        self.boundingBoxButton = QPushButton(self.centralwidget)
        self.boundingBoxButton.setObjectName(u"boundingBoxButton")
        self.boundingBoxButton.setSizePolicy(sizePolicy_minEx_max)
        self.boundingBoxButton.setMaximumHeight(30)
        self.gridLayout.addWidget(self.boundingBoxButton, 2, 3, 1, 1)

        self.saveTraceButton = QPushButton(self.centralwidget)
        self.saveTraceButton.setObjectName(u"saveTraceButton")
        self.saveTraceButton.setSizePolicy(sizePolicy_minEx_max)
        # self.saveTraceButton.setMinimumHeight(30)
        self.saveTraceButton.setMaximumHeight(30)
        self.gridLayout.addWidget(self.saveTraceButton, 2, 4, 1, 1)

        self.traceGraph = pg.PlotWidget()  # QtCharts.QChartView(self.centralwidget)
        self.traceGraph.setObjectName(u"traceGraph")
        self.traceGraph.setMinimumHeight(50)
        self.traceGraph.setMaximumHeight(150)
        self.traceGraph.setSizePolicy(sizePolicy_minEx_max)
        self.traceGraph.setBackground("w")
        self.traceGraph.getPlotItem().hideAxis('bottom')
        self.traceGraph.getPlotItem().hideAxis('left')
        self.traceGraph.setMouseEnabled(x=False, y=False)  # Disable mouse panning & zooming
        self.traceGraph.hideButtons()  # Disable corner auto-scale button
        self.traceGraph.getPlotItem().setMenuEnabled(False)  # Disable right-click context menu

        self.gridLayout.addWidget(self.traceGraph, 3, 0, 2, 5)

        mainwindow.setCentralWidget(self.centralwidget)

        # self.statusbar = QStatusBar(mainwindow)
        # self.statusbar.setObjectName(u"statusbar")
        # mainwindow.setStatusBar(self.statusbar)

        self.retranslate_ui(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)

    # setupUi

    def retranslate_ui(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("mainwindow", u"mainwindow", None))
        self.playVideoButton.setText(QCoreApplication.translate("mainwindow", u"Analyze", None))
        self.frameBackButton.setText(QCoreApplication.translate("mainwindow", u"<", None))
        self.saveTraceButton.setText(QCoreApplication.translate("mainwindow", u"Save Trace", None))
        self.boundingBoxButton.setText(QCoreApplication.translate("mainwindow", u"Set Bounding Box", None))
        self.loadVideoButton.setText(QCoreApplication.translate("mainwindow", u"Load Video", None))
        self.frameForwardButton.setText(QCoreApplication.translate("mainwindow", u">", None))
        self.timeStartLabel.setText(QCoreApplication.translate("mainwindow", u"0:00:00.0000", None))
        self.timeEndLabel.setText(QCoreApplication.translate("mainwindow", u"0:00:00.0000", None))
    # retranslateUi


class MainWindow(QtWidgets.QMainWindow, UiMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_ui(self)
        self.setWindowTitle("Entrainment Analysis v1.0")

        self.trackingSlider.setTracking(False)
        self.frameBackButton.clicked.connect(lambda: self.frame_jump(-1))
        self.trackingSlider.valueChanged.connect(lambda: self.adjust_trackingslider())
        self.frameForwardButton.clicked.connect(lambda: self.frame_jump(1))
        self.loadVideoButton.clicked.connect(self.load_video)
        self.playVideoButton.clicked.connect(self.analyze_start)
        self.boundingBoxButton.clicked.connect(lambda: self.set_box())
        self.saveTraceButton.clicked.connect(lambda: self.save_trace())

        # disable buttons until video is loaded
        self.trackingSlider.setEnabled(False)
        self.frameBackButton.setEnabled(False)
        self.frameForwardButton.setEnabled(False)
        self.playVideoButton.setEnabled(False)
        self.boundingBoxButton.setEnabled(False)
        self.saveTraceButton.setEnabled(False)

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

        self.tracker = None  # selection of tracker type

        # tracker trace variables
        # tl is trackerLog
        self.tlFrame = None
        self.tlx1 = None
        self.tlx2 = None
        self.tly1 = None
        self.tly2 = None
        self.tlxMid = None
        self.tlyMid = None

        self.tlxLine = None
        self.tlyLine = None
        # self.tlChart = None
        # self.tlChartXAxis = None
        # self.tlChartYAxis = None

    def eventFilter(self, widget, event):
        # bug: after loading a frame, it resizes slightly, which fires the event, which resizes it,
        # which fires the event, and it gets stuck in an expanding loop
        # RESOLVED: seems to have to do with the frame:
        # https://stackoverflow.com/questions/42833511/qt-how-to-create-image-that-scale-with-window-and-keeps-aspect-ratio

        # https://stackoverflow.com/questions/21041941/how-to-autoresize-qlabel-pixmap-keeping-ratio-without-using-classes/21053898#21053898
        if event.type() == QEvent.Resize and widget is self.videoFrame:
            self.videoFrame.setPixmap(self.videoFrame.pixmap.scaled(self.videoFrame.width(), self.videoFrame.height(),
                                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.SmoothTransformation))

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

                    # create blank trace vars
                    self.tlFrame = list(range(frame_count))
                    self.tlx1 = frame_count * [0]
                    self.tlx2 = frame_count * [0]
                    self.tly1 = frame_count * [0]
                    self.tly2 = frame_count * [0]
                    self.tlxMid = frame_count * [0]
                    self.tlyMid = frame_count * [0]

                    # enable buttons
                    self.trackingSlider.setEnabled(True)
                    self.frameBackButton.setEnabled(True)
                    self.frameForwardButton.setEnabled(True)
                    self.boundingBoxButton.setEnabled(True)
                    self.saveTraceButton.setEnabled(True)

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

                    self.traceGraph.setXRange(0, frame_count)
                    xPen = pg.mkPen(color=(60, 100, 160))
                    yPen = pg.mkPen(color=(160, 60, 60))
                    self.tlxLine = self.traceGraph.plot(self.tlFrame, self.tlxMid, pen=xPen)
                    self.tlyLine = self.traceGraph.plot(self.tlFrame, self.tlyMid, pen=yPen)
                    # self.tlChart = QtCharts.QChart()
                    # self.tlxLine = QtCharts.QLineSeries()
                    # self.tlyLine = QtCharts.QLineSeries()
                    #
                    # self.tlChart.addSeries(self.tlxLine)
                    # self.tlChart.addSeries(self.tlyLine)
                    #
                    # self.tlChartXAxis = QtCharts.QValueAxis()
                    # self.tlChartXAxis.setRange(0, frame_count)
                    #
                    # self.tlChartYAxis = QtCharts.QValueAxis()
                    # self.tlChartYAxis.setRange(0, self.vidHeight)
                    #
                    # self.tlChart.addAxis(self.tlChartXAxis, Qt.AlignmentFlag.AlignBottom)
                    # self.tlChart.addAxis(self.tlChartYAxis, Qt.AlignmentFlag.AlignLeft)
                    # self.traceGraph.setChart(self.tlChart)
                    #
                    # # test = QtCharts.QLineSeries()
                    # # test.append(0, 6)
                    # # test.append(1, 4)
                    # # test.append(3, 2)
                    # # test.append(4, 5)
                    # #
                    # self.tlChart.legend().hide()
                    #
                    # # self.tlChart.setTitle('test chart')
                    # # self.tlChart.addSeries(test)
                    # # self.traceGraph.setChart(self.tlChart)

    def select_tracker(self, tracker_type):

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.legacy.TrackerBoosting.create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL.create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF.create()
        if tracker_type == 'TLD':
            self.tracker = cv2.legacy.TrackerTLD.create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.legacy.TrackerMedianFlow.create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN.create()
        if tracker_type == 'MOSSE':
            self.tracker = cv2.legacy.TrackerMOSSE.create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT.create()

    def set_box(self):
        # self.bbox = (1261, 586, 60, 72)
        frameCopy = self.frameCurrent  # make a copy of the frame and add the rectangle to it rather than the original

        # set up instruction text
        boxInstr = "Select object to track with left mouse button, press Enter when finished"
        font = cv2.FONT_HERSHEY_SIMPLEX
        textSize = cv2.getTextSize(boxInstr, font, 1, 2)[0]
        textX = textSize[1]  # (frameCopy.shape[1] - textSize[0]) / 2
        textY = textSize[1] * 2
        textOrg = (textX, textY)
        cv2.putText(frameCopy, boxInstr, textOrg,
                    font, 1, (0, 0, 255), 2)

        # actually select box
        self.bbox = cv2.selectROI('Press enter to finish', frameCopy, False, printNotice=False)
        cv2.destroyWindow('Press enter to finish')
        if self.bbox > (0, 0, 0, 0):
            self.playVideoButton.setEnabled(True)

            p1 = (int(self.bbox[0]), int(self.bbox[1]))
            p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
            cv2.rectangle(frameCopy, p1, p2, (255, 0, 0), 2, 1)
            self.update_image(frameCopy, self.frameCurrentNumber)
            self.tlx1[self.frameCurrentNumber] = int(self.bbox[0])
            self.tlx2[self.frameCurrentNumber] = int(self.bbox[0] + self.bbox[2])
            self.tly1[self.frameCurrentNumber] = self.vidHeight - int(self.bbox[1])
            self.tly2[self.frameCurrentNumber] = self.vidHeight - int(self.bbox[1] + self.bbox[3])
            self.tlxMid[self.frameCurrentNumber] = int(self.bbox[1] + self.bbox[3] / 2)
            self.tlyMid[self.frameCurrentNumber] = int(self.bbox[0] + self.bbox[2] / 2)

            self.bboxOriginal = self.bbox  # capture ROI location
            self.bboxImage = self.frameCurrent[self.bbox[1]:self.bbox[1]+self.bbox[3],
                                               self.bbox[0]:self.bbox[0]+self.bbox[2]]
        else:
            print("No ROI selected")

    def frame_jump(self, num_frames):
        # button clicked to set frame number forward or back a certain number
        targetFrame = self.trackingSlider.value() + self.trackingSlider.singleStep() * num_frames
        self.load_frame(targetFrame)

    def load_frame(self, target_frame):
        # set the tracking slider to the specified frame, including loading the new image
        self.trackingSlider.setValue(target_frame)
        newTimeStamp = self.get_time_from_frame(target_frame)
        self.timeStartLabel.setText(newTimeStamp)
        self.frameCurrentNumber = target_frame
        # TODO: load selected frame

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
        # move the tracking slider directly
        targetFrame = self.trackingSlider.value()
        newTimeStamp = self.get_time_from_frame(targetFrame)
        self.timeStartLabel.setText(newTimeStamp)
        self.frameCurrentNumber = targetFrame

    def save_trace(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Trace', '',
                                                         "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All files ("
                                                         "*.*)")
        if fileName[0]:
            data = {
                "x1": self.tlx1,
                "x2": self.tlx2,
                "y1": self.tly1,
                "y2": self.tly2,
                "xMid": self.tlxMid,
                "yMid": self.tlyMid
            }
            outputData = pd.DataFrame(data, index=self.tlFrame)
            # outputData["y2"] = self.vidHeight - outputData["y2"]
            # outputData["y1"] = self.vidHeight - outputData["y1"]
            # outputData["xMid"] = (outputData["x1"] + outputData["x2"]) / 2
            # outputData["yMid"] = (outputData["y1"] + outputData["y2"]) / 2

            outputData.to_csv(fileName[0])

    def analyze_start(self):
        # update the play button
        self.playVideoButton.setText("Pause")
        self.playVideoButton.clicked.disconnect(self.analyze_start)
        self.playVideoButton.clicked.connect(self.analyze_stop)
        # disable the buttons that could change the frame
        self.frameBackButton.setEnabled(False)
        self.frameForwardButton.setEnabled(False)
        self.trackingSlider.setEnabled(False)

        # create the tracker
        self.select_tracker("MOSSE")
        _ = self.tracker.init(self.frameCurrent, self.bbox)

        # create the video capture thread
        self.thread = VideoThread(self.cap, self.tracker)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_tracker)
        # start the thread
        self.thread.start()

    def analyze_stop(self):
        self.playVideoButton.clicked.disconnect(self.analyze_stop)
        self.playVideoButton.clicked.connect(self.analyze_start)
        self.playVideoButton.setText("Analyze")
        self.thread.stop()

        self.frameBackButton.setEnabled(True)
        self.frameForwardButton.setEnabled(True)
        self.trackingSlider.setEnabled(True)
        self.thread.change_pixmap_signal.disconnect(self.update_tracker)

    @Slot(np.ndarray)
    def update_image(self, cv_img, frame_number):
        """Updates the image_label with a new opencv image"""
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
        # now update the tracker
        # frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # blur = cv2.GaussianBlur(videoGray, (5, 5), 0)
        # ret, thresh = cv2.threshold(frameGray, 0, 255, cv2.THRESH_BINARY)
        if frame_number <= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):

            # # Start timer
            # timer = cv2.getTickCount()

            # Update tracker
            ok, self.bbox = self.tracker.update(frame)

            # # Calculate Frames per second (FPS)
            # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(self.bbox[0]), int(self.bbox[1]))
                p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                self.tlx1[frame_number] = int(self.bbox[0])
                self.tlx2[frame_number] = int(self.bbox[0] + self.bbox[2])
                self.tly1[frame_number] = self.vidHeight - int(self.bbox[1])
                self.tly2[frame_number] = self.vidHeight - int(self.bbox[1] + self.bbox[3])
                self.tlxMid[frame_number] = int(self.bbox[1] + self.bbox[3] / 2)
                self.tlyMid[frame_number] = int(self.bbox[0] + self.bbox[2] / 2)
                # row = [frameCount, bbox[0], bbox[1], bbox[2], bbox[3]]
                # boxLog.append(row)
                # xMid = int(self.bbox[0] + self.bbox[2] / 2)
                # yMid = int(self.bbox[1] + self.bbox[3] / 2)
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                self.tlxLine.setData(self.tlFrame, self.tlxMid)
                self.tlyLine.setData(self.tlFrame, self.tlyMid)

                # self.traceGraph.update()

            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)

            # Display tracker type on frame
            cv2.putText(frame, "Frame: " + str(int(frame_number)), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (50, 170, 50), 2)

            # Display result
            self.update_image(frame, frame_number)

            # Update chart

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
