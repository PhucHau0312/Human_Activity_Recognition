import cv2
import mediapipe as mp
import pandas as pd

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


lm_list = []
label = ["BODYSWING", "HANDSWING", "HEADSHAKING"]
no_of_frames = 300
i = 0
change = False

while i < len(label):
   if cv2.waitKey(1) == ord('c'):
    change = True
   if change:
    lm_list = []
    ret, frame = cap.read()
    while len(lm_list) <= no_of_frames:
      ret, frame = cap.read()
      if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('d'):
            break
# Write vào file csv
    df  = pd.DataFrame(lm_list)
    df.to_csv(label[i] + ".txt")
    print("Done label: ",label[i])
    i += 1
    change = False
   ret, frame = cap.read()
   if ret:
       cv2.imshow("image", frame)
   print("Processing label: ",label[i])

cap.release()
cv2.destroyAllWindows()
