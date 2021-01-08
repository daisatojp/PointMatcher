from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class MatchingPainter:

    keypoint_default_fill_color = QtGui.QColor(255, 0, 0, 128)
    keypoint_highlighted_fill_color = QtGui.QColor(255, 0, 0, 255)
    keypoint_selected_fill_color = QtGui.QColor(0, 128, 255, 155)
    keypoint_size = 8
    match_line_colors = [
        (255, 0, 0, 128),
        (0, 255, 0, 128),
        (0, 0, 255, 128),
        (255, 255, 0, 128),
        (255, 0, 255, 128),
        (0, 255, 255, 128),
        (255, 255, 255, 128)]
    match_line_width = 2
    match_highlighted_line_width = 3
    match_selected_line_width = 3
    match_line_alpha = 128
    match_highlighted_line_alpha = 255
    match_selected_line_alpha = 255

    def __init__(self):
        self.draw_offset_i_x = 0
        self.draw_offset_i_y = 0
        self.draw_offset_j_x = 0
        self.draw_offset_j_y = 0

    def paint(self, painter, matching, scale):
        pen = QtGui.QPen(self.keypoint_default_fill_color)
        pen.setWidth(max(1, int(round(2.0 / scale))))
        painter.setPen(pen)
        for idx, keypoint in enumerate(matching.get_view_i()['keypoints']):
            point_path = QtGui.QPainterPath()
            point_path.addEllipse(
                keypoint[0] + self.draw_offset_i_x - (self.keypoint_size / scale) / 2.0,
                keypoint[1] + self.draw_offset_i_y - (self.keypoint_size / scale) / 2.0,
                (self.keypoint_size / scale),
                (self.keypoint_size / scale))
            painter.drawPath(point_path)
            if idx == matching.selected_idx_i:
                painter.fillPath(point_path, self.keypoint_selected_fill_color)
            elif idx == matching.highlighted_idx_i:
                painter.fillPath(point_path, self.keypoint_highlighted_fill_color)
            else:
                painter.fillPath(point_path, self.keypoint_default_fill_color)
        for idx, keypoint in enumerate(matching.get_view_j()['keypoints']):
            point_path = QtGui.QPainterPath()
            point_path.addEllipse(
                keypoint[0] + self.draw_offset_j_x - (self.keypoint_size / scale) / 2.0,
                keypoint[1] + self.draw_offset_j_y - (self.keypoint_size / scale) / 2.0,
                (self.keypoint_size / scale),
                (self.keypoint_size / scale))
            painter.drawPath(point_path)
            if idx == matching.selected_idx_j:
                painter.fillPath(point_path, self.keypoint_selected_fill_color)
            elif idx == matching.highlighted_idx_j:
                painter.fillPath(point_path, self.keypoint_highlighted_fill_color)
            else:
                painter.fillPath(point_path, self.keypoint_default_fill_color)
        if matching.get_pair_idx() is not None:
            highlighted_idx_in_view_i = matching.find_match_idx_in_view_i(matching.highlighted_idx_i)
            highlighted_idx_in_view_j = matching.find_match_idx_in_view_j(matching.highlighted_idx_j)
            selected_idx_in_view_i = matching.find_match_idx_in_view_i(matching.selected_idx_i)
            selected_idx_in_view_j = matching.find_match_idx_in_view_j(matching.selected_idx_j)
            for idx, match in enumerate(matching.get_pair()['matches']):
                keypoint_i = matching.get_view_i()['keypoints'][match[0]]
                keypoint_j = matching.get_view_j()['keypoints'][match[1]]
                match_path = QtGui.QPainterPath()
                match_path.moveTo(keypoint_i[0] + self.draw_offset_i_x, keypoint_i[1] + self.draw_offset_i_y)
                match_path.lineTo(keypoint_j[0] + self.draw_offset_j_x, keypoint_j[1] + self.draw_offset_j_y)
                color = self.match_line_colors[idx % len(self.match_line_colors)]
                if idx in (highlighted_idx_in_view_i, highlighted_idx_in_view_j):
                    pen = QtGui.QPen(QtGui.QColor(color[0], color[1], color[2], self.match_highlighted_line_alpha))
                    pen.setWidth(self.match_highlighted_line_width / scale)
                elif idx in (selected_idx_in_view_i, selected_idx_in_view_j):
                    pen = QtGui.QPen(QtGui.QColor(color[0], color[1], color[2], self.match_selected_line_alpha))
                    pen.setWidth(self.match_selected_line_width / scale)
                else:
                    pen = QtGui.QPen(QtGui.QColor(color[0], color[1], color[2], self.match_line_alpha))
                    pen.setWidth(self.match_line_width / scale)
                painter.setPen(pen)
                painter.drawPath(match_path)