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
import os 
import json
from datetime import datetime 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
cap = cv2.VideoCapture(0)

# global variables
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_dims =  [frame_width, frame_height]
ref_list1 = []
ref_list2 = []
ref_list3 = []
refvar1 = 0.00
refvar2 = 0.00
task_state = "None"


# math 
def display_parts(image, text, designation):
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
def two_point_angle(p1, p2):
    opp = p2[0] - p1[0]
    adj = p2[1] - p1[1]
    radians = np.arctan(opp/adj)
    angle = radians*180/np.pi 
    
    return angle
def arc_math(main, p1, p2):
    try: 
        main = np.multiply(main, frame_dims)
        p1 = np.multiply(p1, frame_dims)
        p2 = np.multiply(p2, frame_dims)

        #radius = (main[0]-e2[0])**2 + (main[1]-e2[1])**2
        #bottom_side = (e2[0]-e1[0])**2 + (e2[1]-e1[0])**2 
        #angle = np.arcsin(-(bottom_side - 2*radius)/(2*(radius**2)))
        #arc = radius * angle
        #return arc
        if p2[0]-p1[0] > 10 or p2[0]-p1[0] < -10:
            length = np.sqrt(np.square(p2[0]-p1[0]) + np.square(p2[1]-p1[1]))
            
            
        else:
            length = 0
            
        return length
    
    except: 
        pass

# global functions
def setup(state):
    global task_state
    global refvar1
    global refvar2
    ref_list1.clear() 
    ref_list2.clear() 
    ref_list3.clear()
    refvar2 = 0.00
    refvar1 = 0.00
    task_state = state

def plot_graph():
    
    if task_state == "arm swing":
        global refvar2 
        
        mins_x = []
        for i in range(1, len(ref_list1) - 1):
            try: 
                if ref_list1[i] < ref_list1[i - 5] and ref_list1[i] < ref_list1[i + 5] and ref_list1[i] != ref_list1[i + 5]:
                    if ref_list1[i] < ref_list1[i - 2] and ref_list1[i] < ref_list1[i + 2] and ref_list1[i] != ref_list1[i + 2]:
                        if ref_list1[i] < ref_list1[i - 1] and ref_list1[i] < ref_list1[i + 1] and ref_list1[i] != ref_list1[i+ 1]:
                            mins_x.append(i)
            except: 
                pass   

        
        Data_inf_msg = ""
        refvar = 0
        for index, item in enumerate(mins_x):
            if index == 0:
                refpoint = 0
            else: 
                refpoint = ref_list2[mins_x[index-1]]
            change_in_interval = int(ref_list2[item] - refpoint)
            Data_inf_msg = f"arm swing was \n {change_in_interval}  \n pixels long"    
            refvar2 += change_in_interval 
            plt.annotate(Data_inf_msg, xy=(item, -30))  
        refvar2 = (refvar2 - ref_list2[mins_x[0]]) / float(len(mins_x)-1)
        
            
        angle_x = np.linspace(0, len(ref_list1) - 1, len(ref_list1))
        swing_x = np.linspace(0, len(ref_list2) - 1, len(ref_list2))
        plt.plot(angle_x, ref_list1)   
        plt.plot(swing_x, ref_list2)
        plt.vlines(mins_x, ymin=-30, ymax=30, color='black')
        plt.title('Shoulder Swing Over Time ')
        plt.xlabel('time')
        plt.ylabel('y')

    
    
    plt.show()  
    
def arm_swing(state="arm swing"):
    setup(state) 
    global refvar1
    

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
                landmarks = results.pose_landmarks.landmark
                
                # local variables 
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                ref_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                shoulder_alignment = [np.abs(np.multiply(shoulder[0], frame_width)-np.multiply(ref_shoulder[0], frame_width)), np.abs(np.multiply(shoulder[1], frame_height)-np.multiply(ref_shoulder[1], frame_height))]
            
                if shoulder_alignment[0] <= 200:

                    # rendering 
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=2) 
                                            )               
                    
                    elbow_angle = three_point_angle(shoulder, elbow, wrist)
                    shoulder_angle = two_point_angle(shoulder, elbow)  
                    display_parts(image, "elbow", elbow)
                    display_parts(image, "shoulder", shoulder)
                    display_parts(image, "wrist", wrist) 
                    ref_list1.append(float(shoulder_angle))                        
                    ref_list3.append(elbow)  
                    
                    arc = abs(arc_math(shoulder, ref_list3[-2], elbow))
                    refvar1 += arc * 0.1
                    ref_list2.append(refvar1)
                    
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    cv2.putText(image, 'stride', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(refvar1), 
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
        cv2.destroyAllWindows()
        pass

def save():
    global task_state
    global refvar1
    global refvar2
    
    current_date = datetime.now()
    
    with open(f"/Users/rehan/Desktop/C00D1NG/GMAP/records/{task_state}/{current_date}.json", "w") as json_file:
        contents = '{'+f'"{task_state}":{refvar2}'+'}'
        json_file.write(contents)

def reference():
    global task_state
    global refvar1
    global refvar2
    records_list = []
    json_files = [filename for filename in os.listdir(f"/Users/rehansha/Desktop/C00D1NG/GMAP/records/{task_state}") if filename.endswith('.json')]
    json_files.sort()
    
    for filename in json_files:
        if filename.endswith('.json'):
            file_path = os.path.join(f"/Users/rehansha/Desktop/C00D1NG/GMAP/records/{task_state}", filename)
            print(filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file) 
            records_list.append(json_data["arm swing"])
            
    percent_change = 100 * (records_list[-1]/records_list[0])
    message = f"your {task_state} \nis {percent_change} \nof what it was"     
    x = np.linspace(0, len(records_list) - 1, len(records_list))
    plt.plot(x, records_list)
    plt.title(f"History of {task_state}")
    plt.annotate(message, xy=(len(records_list)-1, records_list[-1]))
    plt.xlabel('entries')
    plt.ylabel(f"{task_state}")
    plt.show()
    
def load (): 
    pass 

def exit_application():
    cap.release()
    root.destroy()
    


root = ctk.CTk() 
root.title("G-MAP")
#root._set_appearance_mode("dark")

video_button = ctk.CTkButton(root, text = "record", command=arm_swing)
video_button.pack(padx=10, pady=10)
plot_button = ctk.CTkButton(root, text = "plot", command=plot_graph)
plot_button.pack(padx=10, pady=10)
plot_button = ctk.CTkButton(root, text = "save to files", command=save)
plot_button.pack(padx=10, pady=10)
plot_button = ctk.CTkButton(root, text = "review previous", command=reference)
plot_button.pack(padx=10, pady=10)
exit_button = ctk.CTkButton(root, text="Exit", command=exit_application)
exit_button.pack(pady=20)

root.mainloop()

