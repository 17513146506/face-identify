from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPainter,QImage,QPalette,QBrush
import cv2
import os
import time
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1060, 578)
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap('background.jpg')))
        MainWindow.setPalette(palette)
        MainWindow.setWindowIcon(QIcon('logo.jpg'))
        self.actionOpen_camera = QAction(MainWindow)
        self.actionOpen_camera.setObjectName(u"actionOpen_camera")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 50, 300, 100))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(750, 175, 300, 100))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(750, 300, 300, 100))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(750, 425, 300, 100))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 60, 161, 81))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 565, 100))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人脸识别系统"))
        self.pushButton.setText(_translate("MainWindow", "采集人脸信息"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "开始训练"))
        self.pushButton_4.setText(_translate("MainWindow", "退出系统"))
        # self.label.setText(_translate("MainWindow", "结果："))
        self.pushButton.clicked.connect(self.b)
        self.pushButton_2.clicked.connect(self.final)
        self.pushButton_3.clicked.connect(self.train)
        self.pushButton_4.clicked.connect(self.exit)
    def b(self):
        print('正在调用摄像头！')
        print("输入'esc'为退出！！！")
        cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_id = input('\n 输入录入人脸的id ：')
        print("\n 开启摄像头，请看摄像头")
        count = 0
        while (True):
            self.ret, self.img = cam.read()
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite('./out/' + str(face_id) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])
                show = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.label.setGeometry(-5, -35, 900, 650)
                self.label.setPixmap(QPixmap.fromImage(showImage))
            if count == 50:
                print("\n........................................ 录入人脸完毕....................................................")
                break
        self.label.setPixmap(QPixmap(""))
    def final(self):
        from PIL import Image, ImageDraw, ImageFont
        import sys
        import cv2
        import numpy as np
        import os
        from PIL import Image, ImageDraw, ImageFont
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = 'haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        # iniciate id counter
        id = 0
        names = ['周何彬']
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height
        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
            if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            fontStyle = ImageFont.truetype(
                "simsun.ttc", textSize, encoding="utf-8")
            # 绘制文本
            draw.text(position, text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        while True:
            ret, img =cam.read()
            # img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w + 10, y + h + 50), (0, 200, 255), 5)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 65):
                    id = names[id-1]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                print('标签id:', id, '置信评分：', confidence)
                img = cv2AddChineseText(img, str(id), (250, 50), (255, 0, 0), 80)
            show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label.setGeometry(-5, -35,900, 650)
            self.label.setPixmap(QPixmap.fromImage(showImage))
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
    def train(self):
        import cv2
        import numpy as np
        from PIL import Image  # 导入pillow库里的image模块
        import os
        path = './face/'
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # 设置人脸识别器
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");  # 设置人脸检测器
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # 给出dataset里每个图片的目录并保存在数组imagePaths
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[1].split('.')[0])
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
            return faceSamples, ids
        print("\n 正在训练人脸，请等待")
        faces, ids = getImagesAndLabels(path)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        print("\n 训练完毕".format(len(np.unique(ids))))
        recognizer.save('trainer/trainer.yml')
        print(" -------------------成功将人脸全部写进数据集---------------------")
    def exit (self):
        reply = QMessageBox.question(self, 'Warning', '确认退出？', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            exit(0)
        else:
            self.close()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())