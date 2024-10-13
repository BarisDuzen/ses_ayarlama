import cv2
import time
import mediapipe as mp
import math
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from comtypes import COMError

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands  # modül tanımı
hands = mp_hand.Hands()  # el takibi başlatıyoruz
mp_draw = mp.solutions.drawing_utils  # modül tanımı

max_distance = 295
min_distance = 27

def setVolume(level):
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(level / 100, None)
        volume.Release()  # COM nesnesini serbest bırak
        print("Bağlı cihaz:",devices)
    except COMError as e:
        print(f"COM Error: {e}")

ptime = 0

while True:
    ret, frame = cap.read()  # kamera takibi başlıyor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # el takibi için bgr'yi rgb'ye dönüştürüyoruz.
    
    results = hands.process(frame_rgb)  # elde ettiğimiz kamera görüntülerini el takibi fonksiyonuna atıyoruz.
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)  # kameramıza çizim yaptırıyoruz.
            
            bas_parmak = hand_landmarks.landmark[4]
            isaret_parmak = hand_landmarks.landmark[8]
            
            h, w, c = frame.shape
            bas_parmak_x, bas_parmak_y = int(w * bas_parmak.x), int(h * bas_parmak.y)
            isaret_parmak_x, isaret_parmak_y = int(w * isaret_parmak.x), int(h * isaret_parmak.y)
            
            # İki parmak arasındaki mesafeyi hesapla
            distance = math.sqrt((bas_parmak_x - isaret_parmak_x) ** 2 + (bas_parmak_y - isaret_parmak_y) ** 2)
            
            # Ses seviyesi aralığını belirle
            volume_level = np.interp(distance, [min_distance, max_distance], [0, 100])
            setVolume(volume_level)
    
    ntime = time.time()
    fps = 1 / (ntime - ptime)
    ptime = ntime
    
    cv2.putText(frame, "FPS:" + str(int(fps)), (10, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("El Takibi", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
