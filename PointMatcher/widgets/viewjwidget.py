import os.path as osp
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QDockWidget
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget


class ViewJWidget(QDockWidget):

    ITEM_COLOR_OK = QBrush(QColor(255, 255, 255))
    ITEM_COLOR_NG = QBrush(QColor(255, 255, 0))

    def __init__(self, parent):
        super(ViewJWidget, self).__init__('View J List', parent)
        self.p = parent

        self.viewListWidget = QListWidget()
        self.viewlistLayout = QVBoxLayout()
        self.viewlistLayout.setContentsMargins(0, 0, 0, 0)
        self.viewlistLayout.addWidget(self.viewListWidget)
        self.viewWidgetContainer = QWidget()
        self.viewWidgetContainer.setLayout(self.viewlistLayout)
        self.setObjectName('View')
        self.setWidget(self.viewWidgetContainer)
        self.setFeatures(QDockWidget.DockWidgetFloatable)

    def itemClicked_connect(self, f):
        self.viewListWidget.itemClicked.connect(f)

    def count(self):
        return self.viewListWidget.count()

    def get_current_idx(self):
        return self.viewListWidget.currentIndex().row()

    def set_current_idx(self, idx):
        self.viewListWidget.setCurrentRow(idx)

    def initialize(self):
        self.viewListWidget.clear()
        list_of_view_id = self.p.matching.get_list_of_view_id()
        for view_id in list_of_view_id:
            self.viewListWidget.addItem(self.item_text(view_id))

    def update_text(self):
        list_of_view_id = self.p.matching.get_list_of_view_id()
        assert self.count() == len(list_of_view_id)
        for i in range(self.count()):
            self.viewListWidget.item(i).setText(self.item_text(list_of_view_id[i]))

    def item_text(self, view_id):
        view_id_i = self.p.matching.get_view_id_i()
        return '(ID={}, K={}, KN={}, M={}) {}'.format(
            view_id,
            self.p.matching.get_keypoint_count(view_id),
            self.p.matching.get_keypoint_none_count(view_id),
            self.p.matching.get_match_count(view_id_i, view_id),
            self.p.matching.get_filename(view_id))
