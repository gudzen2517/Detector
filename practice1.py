import cv2
import numpy as np
# Поиск людей в группе
def bubble_sort(arr) -> list:
    n = len(arr)
    group = []
    flag = True
    for i in range(n):
        if flag:
            group = []
            group.append(arr[i])
            for j in range(i, n-1):
                if np.sqrt(abs(arr[i][0] - arr[j+1][0])**2 + abs(arr[i][1] - arr[j+1][1])**2) <= 100:
                    flag = False
                    group.append(arr[j+1])
        else:
            for j in range(i, n-1):
                # Если текущий элемент больше следующего, меняем их местами
                if np.sqrt(abs(arr[i][0] - arr[j+1][0])**2 + abs(arr[i][1] - arr[j+1][1])**2) <= 100 and arr[j+1] not in group:
                    flag = False
                    group.append(arr[j+1])
    return group


# Загрузка YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Загрузка классов
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Загрузка изображения
image = cv2.imread("720x.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определение диапазонов для красного цвета
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2

# Формирование контуров головных уборов
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
people_helmet = []
if contours:
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            people_helmet.append([image.shape[0]-y-h//2, x+w//2])

height, width, _ = image.shape

# Подготовка изображения для нейросети
blob = cv2.dnn.blobFromImage(image, 0.0005, (800,700), (0, 0, 0), True, crop=False)
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
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
targets = []
print("Number of people: ", len(indexes))
helmet = 0
without_helmet = 0
for i in range(len(boxes)):
    if i in indexes:
        color = (0, 255, 0)
        x, y, w, h = boxes[i]
        targets.append([x + w // 2, y + h // 2, x, y, w, h])
        label = str(classes[class_ids[i]])
        for coord_h in people_helmet:
            if image.shape[0]-y-h//2 < coord_h[0] < image.shape[0]-y and x < coord_h[1] < x+w:
                color = (0, 255, 255)
                helmet += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, str(round(confidences[i], 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
group = np.array(bubble_sort(targets))
cv2.putText(image, "Number of people in the group: "+str(len(group)), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
print("Человек в группе: ", len(group))
cv2.rectangle(image, (int(min(group[:,2])), int(min(group[:,3]))), (int(max(group[:,2]+group[:,4])), int(max(group[:,3]+group[:,5]))), (0, 0, 255), 2)
cv2.putText(image, "Number of people with helmet:  "+str(helmet), (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
cv2.putText(image, "Number of people without helmet: " + str(len(indexes) - helmet), (0, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
print("Человек в каске: ", helmet)
print("Человек без каски: ", len(indexes)-helmet)
# Показать изображение
cv2.imwrite('output_image.jpg', image)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
