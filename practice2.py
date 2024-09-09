import cv2
import numpy as np

new_cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video-tracking.mp4")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Загрузка классов
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
point_old = []

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter('output.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 20.0, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение диапазонов для красного цвета
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    people_helmet = []
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                people_helmet.append([frame.shape[0] - y - h // 2, x + w // 2])
                # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Подготовка изображения для нейросети
    blob = cv2.dnn.blobFromImage(frame, 0.0005, (800, 700), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Обработка результатов
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == "person":  # Фильтруем только людей
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Прямоугольник вокруг обнаруженного объекта
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Нанесение прямоугольников на изображение
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    cv2.putText(frame, "Number of people: "+str(len(indexes)), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    print("Number of people: ", len(indexes))
    helmet = 0
    without_helmet = 0
    for i in range(len(boxes)):
        if i in indexes:
            color = (0, 255, 0)
            x, y, w, h = boxes[i]
            flag = True
            label = str(classes[class_ids[i]])
            for coord_h in people_helmet:
                if frame.shape[0] - y - h // 2 < coord_h[0] < frame.shape[0] - y and x < coord_h[1] < x + w:
                    color = (0, 255, 255)
                    helmet += 1
            # Добавление точек в ломаную траекторий
            if color == (0, 255, 0):
                if len(point_old) == 0:
                    point_old.append([])
                    point_old[len(point_old) - 1].append([x + w // 2, y + h // 2])
                else:
                    for person in point_old:

                        if len(person) == 0:
                            person.append([x + w // 2, y + h // 2])
                        else:
                            if y < person[len(person)-1][1] < y+h and x < person[len(person)-1][0] < x+w:
                                person.append([x + w // 2, y + h // 2])
                                flag = False
                    if flag:
                        point_old.append([])
                        point_old[len(point_old)-1].append([x + w // 2, y + h // 2])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, str(round(confidences[i], 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(frame, "Number of people with helmet:  "+str(helmet), (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(frame, "Number of people without helmet: " + str(len(indexes) - helmet), (0, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    print("Человек в каске: ", helmet)
    print("Человек без каски: ", len(indexes) - helmet)
    cv2.putText(frame, str(len(indexes)), (0, 0), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # Отрисовка траекторий
    for person in point_old:
        for i in range(1, len(person)-1):
            cv2.line(frame, person[i], person[i-1], (0, 0, 0), 2)
    # Показать изображение
    output.write(frame)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Video", frame)
    cv2.waitKey(1)
cap.release()
output.release()
