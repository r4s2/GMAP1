#%pip3 install opencv-python -q
#%pip3 install mediapipe -q 
#%pip3 install matplotlib
#%pip3 install numpy
#%pip3 install p

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np 
import time 
import matplotlib.pyplot as plt
import customtkinter as ctk

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
cap = cv2.VideoCapture(0)

# global variables
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
shoulder_angle_list = []


# global functions
def plot_graph():
    x = np.linspace(0, len(shoulder_angle_list) - 1, len(shoulder_angle_list))
    plt.plot(x, shoulder_angle_list)
    plt.title('Shoulder Angle over Time ')
    plt.xlabel('time')
    plt.ylabel('shoulder angle')
    plt.show()
    
def record_video():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            try: 
                # imaging 
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # local variables 
                landmarks = results.pose_landmarks.landmark
                
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                ref_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                shoulder_alignment = [np.abs(np.multiply(shoulder[0], frame_width)-np.multiply(ref_shoulder[0], frame_width)), np.abs(np.multiply(shoulder[1], frame_height)-np.multiply(ref_shoulder[1], frame_height))]
            
                if shoulder_alignment[0] <= 150:

                    # rendering 
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=2) 
                                            )               
                    
                    def display_parts(text, designation):
                        cv2.putText(image, text, 
                                    tuple(np.multiply(designation, [frame_width, frame_height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )
                    def three_point_angle(a,b,c):
                        a = np.array(a) # First
                        b = np.array(b) # Mid
                        c = np.array(c) # End
                        
                        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                        angle = np.abs(radians*180.0/np.pi)
                        
                        if angle >180.0:
                            angle = 360-angle
                            
                        return angle
                    elbow_angle = three_point_angle(shoulder, elbow, wrist)
                    def two_point_angle(p1, p2):
                        opp = p2[0] - p1[0]
                        adj = p2[1] - p1[1]
                        radians = np.arctan(opp/adj)
                        angle = radians*180/np.pi 
                        
                        return angle 
                    shoulder_angle = two_point_angle(shoulder, elbow)
                    
                    display_parts("elbow", elbow)
                    display_parts("shoulder", shoulder)
                    display_parts("wrist", wrist)        
                    #display_parts(("       angle:"+str(elbow_angle)), elbow)
                    #display_parts(("             angle:"+str(shoulder_angle)), shoulder)
                    shoulder_angle_list.append(int(shoulder_angle))
                    
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    cv2.putText(image, 'SHOULDER', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(shoulder_angle), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'ELBOW', (15,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(elbow_angle), 
                                (10,140),        
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
                    #wrap up statements 
        
        
                else: 
                    pass
            except: 
                pass
            
            
            # close 
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        pass

def exit_application():
    root.destroy()

root = ctk.CTk() 
root.title("G-MAP")
#root._set_appearance_mode("dark")

video_button = ctk.CTkButton(root, text = "record", command=record_video)
video_button.pack(padx=10, pady=10)
plot_button = ctk.CTkButton(root, text = "plot", command=plot_graph)
plot_button.pack(padx=10, pady=10)
exit_button = ctk.CTkButton(root, text="Exit", command=exit_application)
exit_button.pack(pady=20)

root.mainloop()