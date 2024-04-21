import cv2
import os

# อ่านวิดีโอจากกล้อง
cap = cv2.VideoCapture(0)

# อ่านไฟล์สำหรับการจำแนกใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# ตั้งค่าตัวแปรสำหรับปุ่ม Exit
exit_clicked = False

# ตัวแปรสำหรับเก็บข้อมูลใบหน้า
face_data = []
face_names = []

# ฟังก์ชันสำหรับการคลิกที่ปุ่ม Exit
def exit_button_callback(event, x, y, flags, param):
    global exit_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if 540 <= x <= 640 and 0 <= y <= 40:
            exit_clicked = True

# ฟังก์ชันสำหรับแสดงข้อความบนภาพ
def show_text(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# กำหนดฟังก์ชันให้กับการคลิกที่ภาพ
cv2.namedWindow("Output")
cv2.setMouseCallback("Output", exit_button_callback)

# แสดงผลวิดีโอ
while cap.isOpened():
    check, frame = cap.read()
    if check:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # จำแนกใบหน้า
        face_detect = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        # แสดงตำแหน่งที่เจอใบหน้า
        for (x, y, w, h) in face_detect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)
            # ตรวจจับตาในใบหน้า
            roi_gray = gray_img[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            # แสดงชื่อใบหน้า (ถ้ามี)
            if face_names:
                # ตรวจสอบความสอดคล้องระหว่างภาพใบหน้าและชื่อ
                for i, face_img in enumerate(face_data):
                    result = cv2.matchTemplate(gray_img[y:y+h, x:x+w], face_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > 0.8:
                        cv2.putText(frame, face_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break
                else:
                    # ใส่กรอบสีแดงสำหรับบุคคลที่ไม่รู้จัก
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=5)
        # แสดงข้อความเมื่อไม่พบใบหน้า
        if len(face_detect) == 0:
            show_text(frame, "No face detected", 10, 30)

        # เพิ่มปุ่ม Exit ที่มุมขวาบน
        exit_text = "Exit"
        exit_text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        frame_height, frame_width, _ = frame.shape
        exit_text_x = frame_width - exit_text_size[0] - 10
        exit_text_y = 30
        cv2.putText(frame, exit_text, (exit_text_x, exit_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Output", frame)

        # หากปุ่ม "e" ถูกกดหรือคลิกที่ปุ่ม "Exit" ในหน้าต่าง
        key = cv2.waitKey(1)
        if key == ord("e") or exit_clicked:
            break
        # หากปุ่ม "s" ถูกกด เพื่อเซฟใบหน้าและชื่อ
        elif key == ord("s"):
            name = input("Enter face name: ")  # รับชื่อใบหน้าจากผู้ใช้
            # เซฟใบหน้าและชื่อ
            if name:
                face_data.append(gray_img[y:y+h, x:x+w].copy())  # เก็บข้อมูลใบหน้า
                face_names.append(name)  # เก็บชื่อใบหน้า

    else:
        break

cap.release()
cv2.destroyAllWindows()

# ตรวจสอบว่ามีข้อมูลใบหน้าที่จะเซฟ
if face_data:
    # สร้างโฟลเดอร์สำหรับเก็บข้อมูลใบหน้าหากยังไม่มี
    if not os.path.exists("faces"):
        os.makedirs("faces")

    # เซฟข้อมูลใบหน้าลงในโฟลเดอร์ faces
    for i, face_img in enumerate(face_data):
        cv2.imwrite(f"faces/{face_names[i]}_{i}.jpg", face_img)

    print("Face data saved successfully.")
else:
    print("No face data to save.")
