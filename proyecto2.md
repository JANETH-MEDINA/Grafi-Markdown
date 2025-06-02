```python
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import sys
import cv2
import mediapipe as mp
import threading
import time
import math


class HandDetector:
    def __init__(self): 
        
        self.detection_confidence = 0.7
        self.tracking_confidence = 0.7
        self.detection_timeout = 0.4
        self.gesture_sensitivity = 0.7
        self.position_smoothing = 0.4
        self.dead_zone_size = 0.15
        
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            max_num_hands=1
        )
        
        self.cap = cv2.VideoCapture(1)
        self.running = False
        
        
        self.hand_position = {'x': 0.5, 'y': 0.5}
        self.smooth_position = {'x': 0.5, 'y': 0.5}
        
        self.hand_detected = False
        self.gesture = "none"
        self.pointing_direction = "none"  
        self.last_detection_time = time.time()
        
        
        self.position_history = []
        self.gesture_history = []
        self.pointing_history = []  
        self.max_history = 5
        self.min_stable_frames = 3
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.detect_hands)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
    def smooth_position_update(self, new_x, new_y):
        """Aplica suavizado a la posición de la mano"""
        
        self.position_history.append({'x': new_x, 'y': new_y})
        
        
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            
        
        if len(self.position_history) > 0:
            
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for i, pos in enumerate(self.position_history):
                weight = (i + 1) / len(self.position_history)
                weighted_x += pos['x'] * weight
                weighted_y += pos['y'] * weight
                total_weight += weight
                
            avg_x = weighted_x / total_weight
            avg_y = weighted_y / total_weight
            
           
            self.smooth_position['x'] = (self.smooth_position['x'] * (1 - self.position_smoothing) + 
                                       avg_x * self.position_smoothing)
            self.smooth_position['y'] = (self.smooth_position['y'] * (1 - self.position_smoothing) + 
                                       avg_y * self.position_smoothing)
            
          
            self.hand_position['x'] = self.smooth_position['x']
            self.hand_position['y'] = self.smooth_position['y']
    
    def stabilize_gesture(self, new_gesture):
        """Estabiliza el reconocimiento de gestos"""
        self.gesture_history.append(new_gesture)
        
        
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
            
        
        if len(self.gesture_history) >= self.min_stable_frames:
            gesture_counts = {}
            for g in self.gesture_history:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
                
            
            most_common = max(gesture_counts, key=gesture_counts.get)
            
            
            consensus_threshold = max(2, len(self.gesture_history) * self.gesture_sensitivity)
            if gesture_counts[most_common] >= consensus_threshold:
                return most_common
                
        return self.gesture  
    def stabilize_pointing(self, new_direction):
        """Estabiliza la detección de dirección de señalado"""
        self.pointing_history.append(new_direction)
        
        if len(self.pointing_history) > self.max_history:
            self.pointing_history.pop(0)
            
        if len(self.pointing_history) >= self.min_stable_frames:
            direction_counts = {}
            for d in self.pointing_history:
                direction_counts[d] = direction_counts.get(d, 0) + 1
                
            most_common = max(direction_counts, key=direction_counts.get)
            
            consensus_threshold = max(2, len(self.pointing_history) * 0.6)
            if direction_counts[most_common] >= consensus_threshold:
                return most_common
                
        return self.pointing_direction
        
    def detect_hands(self):        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)  
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                current_time = time.time()
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        
                        wrist = hand_landmarks.landmark[0]
                        
                        
                        self.smooth_position_update(wrist.x, wrist.y)
                        
                        self.hand_detected = True
                        self.last_detection_time = current_time
                        
                       
                        raw_gesture = self.detect_gesture(hand_landmarks)
                        self.gesture = self.stabilize_gesture(raw_gesture)
                        
                        
                        raw_direction = self.detect_pointing_direction(hand_landmarks)
                        self.pointing_direction = self.stabilize_pointing(raw_direction)
                else:
                    
                    if current_time - self.last_detection_time > self.detection_timeout:
                        self.hand_detected = False
                        self.gesture = "none"
                        self.pointing_direction = "none"
                        
                        self.position_history.clear()
                        self.gesture_history.clear()
                        self.pointing_history.clear()
                
                cv2.imshow("Control de Manos", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
            except Exception as e:
                print(f"Error en detección de manos: {e}")
                continue
    
    def detect_gesture(self, hand_landmarks):
        """Detecta si la mano está abierta o cerrada con sensibilidad ajustable"""
        
        fingertips = [4, 8, 12, 16, 20]  
        
        pip_joints = [3, 6, 10, 14, 18]
        
        fingers_up = 0
        
        
        threshold_adjustment = (1.0 - self.gesture_sensitivity) * 0.1  
        
        for i in range(5):
            if i == 0:  
                if hand_landmarks.landmark[fingertips[i]].x > (hand_landmarks.landmark[pip_joints[i]].x + threshold_adjustment):
                    fingers_up += 1
            else: 
                if hand_landmarks.landmark[fingertips[i]].y < (hand_landmarks.landmark[pip_joints[i]].y - threshold_adjustment):
                    fingers_up += 1
        
        
        open_threshold = 4 * self.gesture_sensitivity  
        closed_threshold = 2 * (1.0 - self.gesture_sensitivity) 
        
        if fingers_up >= open_threshold:
            return "abierta"
        elif fingers_up <= closed_threshold:
            return "puño"
        else:
            return "parcial"

    def detect_pointing_direction(self, hand_landmarks):
        """Detecta la dirección hacia donde está señalando"""
        if not hand_landmarks:
            return "none"
        
        
        index_tip = hand_landmarks.landmark[8] 
        index_mcp = hand_landmarks.landmark[5]  
        wrist = hand_landmarks.landmark[0]    
        
        
        finger_vector_y = index_tip.y - index_mcp.y
        finger_vector_x = index_tip.x - index_mcp.x
        
        
        index_extended = index_tip.y < (index_mcp.y - 0.05)
        
        if not index_extended:
            return "none"
        
        
        if abs(finger_vector_y) > abs(finger_vector_x):
            if finger_vector_y < -0.15:  
                return "arriba"
            elif finger_vector_y > 0.15:  
                return "abajo"
        else:
            if finger_vector_x > 0.15:  
                return "derecha"
            elif finger_vector_x < -0.15: 
                return "izquierda"
        
        return "none"


class BearGesture:
    def __init__(self):  
        self.x, self.y, self.z = 0.0, -0.5, 0.0
        self.rotation_y = 0.0
        self.camera_distance = 8.0
        self.rotation_speed = 2.5  
        
       
        self.is_jumping = False
        self.jump_velocity = 0.0
        self.jump_height = 0.0
        self.gravity = -0.5
        self.jump_force = 8.0
        self.ground_y = -0.5
        
       
        self.ground_bounds = {
            'min_x': -45.0,
            'max_x': 45.0,
            'min_z': -45.0,
            'max_z': 45.0
        }
        
        
        self.hand_detector = HandDetector()
        
        
        self.rotating = False
        self.last_gesture_time = time.time()
        self.should_move = False
        self.last_jump_time = 0  

    def start_hand_detection(self):
        self.hand_detector.start()
        
    def stop_hand_detection(self):
        self.hand_detector.stop()

    def update_jump(self):
        """Actualiza la física del salto"""
        if self.is_jumping:
            
            self.jump_height += self.jump_velocity * 0.1 
            self.jump_velocity += self.gravity
            
         
            if self.jump_height <= 0:
                self.jump_height = 0
                self.jump_velocity = 0
                self.is_jumping = False
        
       
        self.y = self.ground_y + self.jump_height

    def start_jump(self):
        """Inicia un salto si no está ya saltando"""
        current_time = time.time()
       
        if not self.is_jumping and (current_time - self.last_jump_time) > 0.5:
            self.is_jumping = True
            self.jump_velocity = self.jump_force
            self.last_jump_time = current_time

    def update_movement_from_gestures(self):
        """Actualiza la rotación y salto basada en los gestos detectados"""
        if not self.hand_detector.hand_detected:
            self.rotating = False
            self.should_move = False
            return
            
        self.should_move = True
        hand_x = self.hand_detector.hand_position['x']
        hand_y = self.hand_detector.hand_position['y']
        gesture = self.hand_detector.gesture
        pointing_direction = self.hand_detector.pointing_direction
        
        
        if pointing_direction == "arriba":
            self.start_jump()
        
        
        if gesture == "puño":
            self.camera_distance = max(3.0, self.camera_distance - 0.1)
        elif gesture == "abierta":
            self.camera_distance = min(20.0, self.camera_distance + 0.1)
        
        
        dead_zone = self.hand_detector.dead_zone_size
        
       
        self.rotating = False
        
        
        if hand_x < 0.5 - dead_zone: 
            self.rotation_y += self.rotation_speed
            self.rotating = True
        elif hand_x > 0.5 + dead_zone: 
            self.rotation_y -= self.rotation_speed
            self.rotating = True
        
        if hand_y < 0.5 - dead_zone: 
            self.rotation_y += self.rotation_speed * 1.5
            self.rotating = True
        elif hand_y > 0.5 + dead_zone:  
            self.rotation_y -= self.rotation_speed * 1.5
            self.rotating = True
        
        
        if self.rotation_y >= 360:
            self.rotation_y -= 360
        elif self.rotation_y < 0:
            self.rotation_y += 360

    def draw_sphere(self, radius, slices=20, stacks=20):
        """Dibuja una esfera"""
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

    def update_camera(self):
        """Configura la cámara para que siga al oso correctamente"""
        glLoadIdentity()
        camera_x = self.x
        camera_y = self.y + 3
        camera_z = self.z + self.camera_distance
        
        gluLookAt(camera_x, camera_y, camera_z,
                  self.x, self.y, self.z,
                  0, 1, 0)

    def draw_bear(self):
        """Dibuja un oso con esferas organizadas según tu especificación"""
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.rotation_y, 0, 1, 0)

         
        if self.should_move and self.hand_detector.hand_detected:
            if self.rotating or self.is_jumping:
                glColor3f(0.7, 0.4, 0.2)   
            else:
                glColor3f(0.6, 0.3, 0.1)   
        else:
            glColor3f(0.4, 0.2, 0.05)   

         
        glPushMatrix()
        glTranslatef(0, 0, 0)
        self.draw_sphere(1.2)  
        glPopMatrix()

      
        glPushMatrix()
        glTranslatef(0, 1.8, 0)
        self.draw_sphere(0.8)   
        glPopMatrix()

        
        
        glPushMatrix()
        glTranslatef(-1.2, 0.5, 0.8)
        self.draw_sphere(0.25)   
        glPopMatrix()

       
        glPushMatrix()
        glTranslatef(1.2, 0.5, 0.8)
        self.draw_sphere(0.25)   
        glPopMatrix()
 
        
        glPushMatrix()
        glTranslatef(-0.6, -1.5, 0)
        self.draw_sphere(0.4)   
        glPopMatrix()
 
        glPushMatrix()
        glTranslatef(0.6, -1.5, 0)
        self.draw_sphere(0.4)   
        glPopMatrix()

        
        glColor3f(0.5, 0.25, 0.1)   

         
        glPushMatrix()
        glTranslatef(-0.4, 2.4, 0.2)
        self.draw_sphere(0.2)   
        glPopMatrix()

        
        glPushMatrix()
        glTranslatef(0.4, 2.4, 0.2)
        self.draw_sphere(0.2)   
        glPopMatrix()

         
        glColor3f(0.1, 0.05, 0.0)   
        glPushMatrix()
        glTranslatef(0, 1.8, 0.7)
        self.draw_sphere(0.08)
        glPopMatrix()

         
        glColor3f(0.0, 0.0, 0.0)   
        
         
        glPushMatrix()
        glTranslatef(-0.2, 2.0, 0.6)
        self.draw_sphere(0.05)
        glPopMatrix()

         
        glPushMatrix()
        glTranslatef(0.2, 2.0, 0.6)
        self.draw_sphere(0.05)
        glPopMatrix()

        glPopMatrix()

    def draw_ground(self):
        """Dibuja el suelo con textura de pasto"""
        glDisable(GL_TEXTURE_2D)
        
         
        self.draw_grass_pattern()
        
        
        self.draw_flowers()
        
         
        self.draw_trees()
        
         
        self.draw_ground_boundaries()

    def draw_grass_pattern(self):
        """Dibuja un patrón de pasto con diferentes tonos de verde"""
         
        glColor3f(0.2, 0.6, 0.2)
        glBegin(GL_QUADS)
        vertices = [
            [self.ground_bounds['min_x'], -2.5, self.ground_bounds['min_z']], 
            [self.ground_bounds['max_x'], -2.5, self.ground_bounds['min_z']], 
            [self.ground_bounds['max_x'], -2.5, self.ground_bounds['max_z']], 
            [self.ground_bounds['min_x'], -2.5, self.ground_bounds['max_z']]
        ]
        for vertex in vertices:
            glVertex3f(*vertex)
        glEnd()
        
         
        import random
        random.seed(42)  
        
        for i in range(200):   
            x = random.uniform(self.ground_bounds['min_x'], self.ground_bounds['max_x'])
            z = random.uniform(self.ground_bounds['min_z'], self.ground_bounds['max_z'])
            
            
            if abs(x) < 5 and abs(z) < 5:
                continue
                
             
            green_intensity = random.uniform(0.3, 0.8)
            glColor3f(0.1, green_intensity, 0.1)
            
             
            size = random.uniform(1.0, 3.0)
            glBegin(GL_QUADS)
            glVertex3f(x - size/2, -2.49, z - size/2)
            glVertex3f(x + size/2, -2.49, z - size/2)
            glVertex3f(x + size/2, -2.49, z + size/2)
            glVertex3f(x - size/2, -2.49, z + size/2)
            glEnd()

    def draw_flowers(self):
        """Dibuja flores coloridas dispersas por el suelo"""
        import random
        random.seed(123)   
        
        for i in range(50):   
            x = random.uniform(self.ground_bounds['min_x'] + 5, self.ground_bounds['max_x'] - 5)
            z = random.uniform(self.ground_bounds['min_z'] + 5, self.ground_bounds['max_z'] - 5)
            
            
            if abs(x) < 8 and abs(z) < 8:
                continue
            
             
            glColor3f(0.2, 0.8, 0.2)
            glPushMatrix()
            glTranslatef(x, -2.0, z)
            glScalef(0.05, 1.0, 0.05)
            self.draw_sphere(0.5)
            glPopMatrix()
            
            
            colors = [
                [1.0, 0.2, 0.2],   
                [1.0, 1.0, 0.2],   
                [0.8, 0.2, 1.0],   
                [1.0, 0.5, 0.8],   
                [0.2, 0.5, 1.0],   
                [1.0, 0.6, 0.2],   
            ]
            
            color = random.choice(colors)
            glColor3f(*color)
            
            glPushMatrix()
            glTranslatef(x, -1.8, z)
            self.draw_sphere(0.15)
            glPopMatrix()

    def draw_trees(self):
        """Dibuja árboles alrededor del área"""
        import random
        random.seed(456)   
        
        
        tree_positions = [
           
            [-40, -40], [-40, 0], [-40, 40],
            [40, -40], [40, 0], [40, 40],
            [-20, -42], [0, -42], [20, -42],
            [-20, 42], [0, 42], [20, 42],
            [-42, -20], [-42, 20],
            [42, -20], [42, 20],
        ]
        
        for pos in tree_positions:
            x, z = pos[0], pos[1]
            
             
            glColor3f(0.4, 0.2, 0.1)   
            trunk_height = random.uniform(3.0, 5.0)
            
            glPushMatrix()
            glTranslatef(x, -2.5 + trunk_height/2, z)
            glScalef(0.3, trunk_height, 0.3)
            self.draw_sphere(1.0)
            glPopMatrix()
            
             
            tree_green = [0.1, random.uniform(0.4, 0.7), 0.1]
            glColor3f(*tree_green)
            
            
            glPushMatrix()
            glTranslatef(x, -2.5 + trunk_height + 1.0, z)
            self.draw_sphere(1.5)
            glPopMatrix()
            
             
            for i in range(3):
                offset_x = random.uniform(-0.8, 0.8)
                offset_z = random.uniform(-0.8, 0.8)
                offset_y = random.uniform(-0.5, 0.5)
                
                glPushMatrix()
                glTranslatef(x + offset_x, -2.5 + trunk_height + 1.0 + offset_y, z + offset_z)
                self.draw_sphere(random.uniform(0.8, 1.2))
                glPopMatrix()

    def draw_small_bushes(self):
        """Dibuja arbustos pequeños"""
        import random
        random.seed(789)
        
        for i in range(30):
            x = random.uniform(self.ground_bounds['min_x'] + 10, self.ground_bounds['max_x'] - 10)
            z = random.uniform(self.ground_bounds['min_z'] + 10, self.ground_bounds['max_z'] - 10)
            
            
            if abs(x) < 12 and abs(z) < 12:
                continue
            
             
            bush_green = [0.2, random.uniform(0.5, 0.8), 0.2]
            glColor3f(*bush_green)
            
            glPushMatrix()
            glTranslatef(x, -2.2, z)
            self.draw_sphere(random.uniform(0.3, 0.6))
            glPopMatrix()

    def draw_ground_boundaries(self):
        """Dibuja los límites visibles del suelo"""
        glColor3f(0.1, 0.4, 0.1)   
        glLineWidth(3.0)
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(self.ground_bounds['min_x'], -2.4, self.ground_bounds['min_z'])
        glVertex3f(self.ground_bounds['max_x'], -2.4, self.ground_bounds['min_z'])
        glVertex3f(self.ground_bounds['max_x'], -2.4, self.ground_bounds['max_z'])
        glVertex3f(self.ground_bounds['min_x'], -2.4, self.ground_bounds['max_z'])
        glEnd()
        
        glLineWidth(1.0)

 
bear = None

def reshape(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    global bear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    bear.update_movement_from_gestures()
    bear.update_jump()   
    bear.update_camera()
    bear.draw_ground()
    bear.draw_small_bushes()  
    bear.draw_bear()
    
    glutSwapBuffers()

def timer(value):
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)

def keyboard(key, x, y):
    global bear
    if key == b'\x1b':   
        bear.stop_hand_detection()
        glutLeaveMainLoop()   

def main():
    global bear
    
    try:
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(1000, 800)
        window = glutCreateWindow(b"Oso 3D - Control por Gestos con Salto")

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.7, 0.9, 1.0, 1.0)   
        
         
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        light_pos = [5.0, 5.0, 5.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glEnable(GL_COLOR_MATERIAL)

        bear = BearGesture()
        bear.start_hand_detection()
        
        glutDisplayFunc(display)
        glutReshapeFunc(reshape)
        glutKeyboardFunc(keyboard)
        glutTimerFunc(16, timer, 0)
        
        print("Programa iniciado correctamente!")
        print("- Mueve tu mano para rotar el oso")
        print("- Haz puño para acercar la cámara") 
        print("- Abre la mano para alejar la cámara")
        print("- Señala hacia arriba para hacer saltar al oso")
        print("- Presiona ESC para salir")
        print("- ¡Disfruta el jardín con árboles y flores!")
        print("- Alejate con la cámara para ver todo el paisaje")
        
        glutMainLoop()
        
    except Exception as e:
        print(f"Error al iniciar el programa: {e}")
        if bear:
            bear.stop_hand_detection()
        sys.exit(1)

if __name__ == "__main__":   
    main()
    ```