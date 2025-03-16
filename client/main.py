from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySideUI import Ui_MainWindow  # 从由ui文件转换而来的Py文件中导入主要函数
import cv2


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cap = cv2.VideoCapture()
        self.button_set()
        '''
        图像显示主要用到了QTimer这个Qt中的计时器组件:
        instance = QTimer(widget)  # 初始化生成计时器实例
        instance.start(timeout)  # 每 timeout 时间后在父组件的基础上发出一个计时信号
        instance.timeout.connect(self.show_picture)  # 我们可以设计一个槽函数用于不断接收这个计时信号，从而实现图像帧的不断显示
        instance.stop  # 关闭计时器
        '''
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_picture)

    '''
    槽函数，该槽函数用于接收及处理 “打开摄像头” 按钮发来的按钮信号
    '''

    def open_camera(self):
        number = self.comboBox.currentIndex()  # 获取当前复选框的索引值，其是依次排列的0、1、2、...
        flag = self.cap.open(number)  # 打开指定的摄像头，若成功打开，该函数会返回 Ture、否之则返回False
        if flag is False:
            QMessageBox.information(self, "警告", "该设备未正常工作", QMessageBox.Ok)
        else:
            self.label.setEnabled(True)  # 此句可删
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(True)
            self.timer.start()  # Qt计时器开始运行，不断的发出计时信号，不断的跳入到show_pic槽函数中，不断的显示图像

    '''
    槽函数，该槽函数用于接收及处理 “关闭摄像头” 按钮发来的按钮信号
    '''

    def close_camera(self):
        self.cap.release()  # 释放摄像头对象
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.timer.stop()  # 停止计时器，不再显示图像帧
        self.label.setText(" ")  # 清空Label，使之重回黑屏状态

    def button_set(self):
        self.pushButton.clicked.connect(self.open_camera)
        self.pushButton_2.clicked.connect(self.close_camera)
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)

    def show_picture(self):
        ret, frame = self.cap.read()
        if ret:
            if frame is not None:
                cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = cur_frame.shape[:2]  # cur_frame=会返回图像的高、宽与颜色通道数，截前2
                '''
                QImage用于访问、转化图像格式操作图像等,其返回值是一个已经转化好格式的QImage对象。
                QImage支持格式枚举描述的几种图像格式，包括单色、8位、32位和alpha混合图像。当然也包括Opencv的 mat 类型数组枚举形式
                格式: QImage(枚举对象图像帧,宽,高,转化成的颜色格式)
                '''
                pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
                '''
                QPixmap用于在屏幕上显示图像, 其返回值是一个QPixmap对象
                QPixmap.fromImage函数用于将QImage对象转化为QPixmap对象，注意QPixmap并非一个图像帧，而是Qt中用于图像展示的一个类对象实例
                其本身也是一种抽象的封装，可被Qt中其他的类对象进行图像显示操作。
                '''
                pixmap = QPixmap.fromImage(pixmap)
                # 获取是视频流和 label 窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                ratio = max(width / self.label.width(), height / self.height())
                pixmap.setDevicePixelRatio(ratio)  # 以适应比例将图像帧置入 Label 中进行播放
                # 视频流置于label中间部分播放
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication([])
    main = MainWindow()
    main.show()
    app.exec()
