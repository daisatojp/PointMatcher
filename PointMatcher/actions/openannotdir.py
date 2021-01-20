import os.path as osp
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PointMatcher import __appname__
from PointMatcher.data import Matching
from PointMatcher.utils.filesystem import icon_path


QFD = QFileDialog


class OpenAnnotDirAction(QAction):

    def __init__(self, parent):
        super(OpenAnnotDirAction, self).__init__('Open Annot Dir', parent)
        self.p = parent

        self.setIcon(QIcon(icon_path('open')))
        self.setShortcut('Ctrl+O')
        self.triggered.connect(self.openAnnotDir)
        self.setEnabled(False)

    def openAnnotDir(self, _value=False):
        if not self.p.mayContinue():
            return
        if (self.p.annotDir is not None) and osp.exists(self.p.annotDir):
            defaultDir = self.p.annotDir
        elif (self.p.imageDir is not None) and osp.exists(self.p.imageDir):
            defaultDir = self.p.imageDir
        else:
            defaultDir = '.'
        annotDir = QFileDialog.getExistingDirectory(
            self.p, '{} - Open Annotation Directory'.format(__appname__), defaultDir,
            QFD.ShowDirsOnly | QFD.DontResolveSymlinks)
        if osp.isdir(annotDir):
            views_dir = osp.join(annotDir, 'views')
            groups_dir = osp.join(annotDir, 'groups')
            if not osp.isdir(views_dir):
                QMessageBox.warning(self.p, 'Error', '{} not found'.format(views_dir), QMessageBox.Ok)
                return
            if not osp.isdir(groups_dir):
                QMessageBox.warning(self.p, 'Error', '{} not found'.format(groups_dir), QMessageBox.Ok)
                return
            self.p.annotDir = annotDir
            self.p.loadMatching()
            self.p.updateTitle()
