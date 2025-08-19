import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
import time
import threading
from queue import Queue
from streamlit_drawable_canvas import st_canvas
import base64
import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

filterwarnings(action='ignore')


class VirtualCalculator:

    def __init__(self):
        """Initialize the calculator with optimized settings."""
        load_dotenv()
        if not os.environ.get('GOOGLE_API_KEY'):
            st.error("Please set GOOGLE_API_KEY in your environment variables")
            st.stop()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.warning("⚠ Webcam not found. Gesture mode disabled.")
            self.cap = None

        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.imgCanvas = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

        self.mphands = hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )

        self.p1, self.p2 = 0, 0
        self.fingers = []
        self.landmark_list = []
        self.frame_count = 0
        self.process_every_n_frames = 2
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.1
        self.last_analysis_time = 0
        self.analysis_cooldown = 2.0
        self.analysis_queue = Queue()
        self.analysis_in_progress = False
        self.current_gesture = "None"

        # Drawing customization
        self.draw_colors = {
            "Purple": (255, 0, 255),
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "Cyan": (255, 255, 0),
            "White": (255, 255, 255)
        }
        self.current_color = "Purple"
        self.draw_color = self.draw_colors[self.current_color]
        self.erase_color = (0, 0, 0)
        self.draw_thickness = 4
        self.erase_thickness = 15

        # Store last analysis for download
        self.last_drawing = None
        self.last_solution = None
        self.last_problem_type = None

    def streamlit_config(self):
        st.set_page_config(
            page_title='Virtual Hand Gesture Calculator',
            layout="wide",
            initial_sidebar_state="expanded"
        )

        page_style = """
        <style>
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        .block-container { padding-top: 0rem; }
        .main-title { 
            text-align: center; 
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradient 3s ease infinite;
            font-family: 'Arial Black', sans-serif; 
            margin-bottom: 20px;
            font-size: 2.5rem;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .status-indicator { 
            padding: 15px; 
            border-radius: 10px; 
            margin: 10px 0; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid;
        }
        .status-ready { 
            background: linear-gradient(135deg, #D4F6D4 0%, #E8F5E8 100%); 
            color: #2D5A2D; 
            border-left-color: #4CAF50;
        }
        .status-analyzing { 
            background: linear-gradient(135deg, #FFF3CD 0%, #FFF8E1 100%); 
            color: #856404; 
            border-left-color: #FF9800;
        }
        .result-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .gesture-guide { 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            padding: 20px; 
            border-radius: 15px; 
            color: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .tutorial-box { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 15px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .gesture-item { 
            background: rgba(255,255,255,0.15); 
            padding: 12px; 
            margin: 8px 0; 
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .color-picker {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .stButton > button {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        </style>
        """
        st.markdown(page_style, unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">🤖 AI Math & CS Problem Solver</h1>',
                    unsafe_allow_html=True)
        add_vertical_space(1)

    def process_frame(self):
        if not self.cap:
            return False
        success, img = self.cap.read()
        if not success:
            return False
        img = cv2.resize(img, (640, 480))
        self.img = cv2.flip(img, 1)
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        return True

    def process_hands(self):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return
        result = self.mphands.process(self.imgRGB)
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(
                    self.img,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                    drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                h, w = self.img.shape[:2]
                for id, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        if len(self.landmark_list) < 21:
            self.fingers = [0, 0, 0, 0, 0]
            return

        self.fingers = []
        tip_ids = [4, 8, 12, 16, 20]

        # Thumb (special case - check x-coordinate)
        if self.landmark_list[4][1] > self.landmark_list[3][1]:
            self.fingers.append(1)
        else:
            self.fingers.append(0)

        # Other fingers (check y-coordinate)
        for i in range(1, 5):
            tip_id = tip_ids[i]
            pip_id = tip_id - 2
            if self.landmark_list[tip_id][2] < self.landmark_list[pip_id][2]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)

    def handle_gestures(self):
        if len(self.fingers) != 5 or len(self.landmark_list) < 21:
            self.current_gesture = "No hand detected"
            return 0

        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return sum(self.fingers)

        finger_sum = sum(self.fingers)

        # Index finger up only - Drawing mode
        if finger_sum == 1 and self.fingers[1] == 1:
            self.current_gesture = "✏️ Drawing"
            if len(self.landmark_list) > 8:
                cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
                if self.p1 == 0 and self.p2 == 0:
                    self.p1, self.p2 = cx, cy
                else:
                    cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy),
                             self.draw_color, self.draw_thickness)
                    cv2.line(self.img, (self.p1, self.p2), (cx, cy),
                             self.draw_color, self.draw_thickness)
                self.p1, self.p2 = cx, cy

        # Index and middle finger up - Stop drawing
        elif finger_sum == 2 and self.fingers[1] == 1 and self.fingers[2] == 1:
            self.current_gesture = "✋ Stop Drawing"
            self.p1, self.p2 = 0, 0
            self.last_gesture_time = current_time

        # Thumb and pinky up - Clear canvas
        elif finger_sum == 2 and self.fingers[0] == 1 and self.fingers[4] == 1:
            self.current_gesture = "🗑️ Clear Canvas"
            self.imgCanvas = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
            self.last_gesture_time = current_time

        # All fingers up - Analyze
        elif finger_sum == 5:
            self.current_gesture = "🔍 Analyze"
            if current_time - self.last_analysis_time > self.analysis_cooldown:
                self.trigger_analysis()
                self.last_analysis_time = current_time
            self.last_gesture_time = current_time

        # Fist (no fingers) - Erase mode
        elif finger_sum == 0:
            self.current_gesture = "🧽 Erasing"
            if len(self.landmark_list) > 8:
                cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
                cv2.circle(self.imgCanvas, (cx, cy), self.erase_thickness, self.erase_color, -1)
                cv2.circle(self.img, (cx, cy), self.erase_thickness, (0, 255, 0), 2)

        else:
            self.current_gesture = f"👋 {finger_sum} fingers detected"
            self.p1, self.p2 = 0, 0

        return finger_sum

    def trigger_analysis(self):
        """Trigger analysis of the current canvas drawing."""
        if not self.analysis_in_progress:
            canvas_pil = PIL.Image.fromarray(cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB))
            if not self.analysis_queue.empty():
                try:
                    self.analysis_queue.get_nowait()
                except:
                    pass
            self.analysis_queue.put(canvas_pil)

    def show_gesture_tutorial(self):
        """Display gesture tutorial in sidebar."""
        st.sidebar.markdown("""
        <div class="tutorial-box">
        <h3 style="color: white; margin-top: 0;">🎯 Gesture Controls</h3>

        <div class="gesture-item">
        <strong>✏️ Draw:</strong><br>
        Point with INDEX finger only<br>
        <em>Move finger to draw lines</em>
        </div>

        <div class="gesture-item">
        <strong>✋ Stop Drawing:</strong><br>
        INDEX + MIDDLE fingers up<br>
        <em>Stops current drawing stroke</em>
        </div>

        <div class="gesture-item">
        <strong>🧽 Erase:</strong><br>
        Make a FIST (no fingers up)<br>
        <em>Move fist to erase areas</em>
        </div>

        <div class="gesture-item">
        <strong>🗑️ Clear All:</strong><br>
        THUMB + PINKY up only<br>
        <em>Clears entire canvas</em>
        </div>

        <div class="gesture-item">
        <strong>🔍 Analyze:</strong><br>
        ALL 5 fingers up (open palm)<br>
        <em>Sends drawing to AI for analysis</em>
        </div>
        </div>
        """, unsafe_allow_html=True)

    def show_drawing_controls(self):
        """Display drawing customization controls."""
        st.sidebar.markdown("""
        <div class="color-picker">
        <h4 style="margin-top: 0;">🎨 Drawing Settings</h4>
        </div>
        """, unsafe_allow_html=True)

        # Color selection
        selected_color = st.sidebar.selectbox(
            "🎨 Choose Color:",
            options=list(self.draw_colors.keys()),
            index=list(self.draw_colors.keys()).index(self.current_color)
        )

        if selected_color != self.current_color:
            self.current_color = selected_color
            self.draw_color = self.draw_colors[selected_color]

        # Thickness control
        self.draw_thickness = st.sidebar.slider(
            "✏️ Drawing Thickness:",
            min_value=1,
            max_value=15,
            value=self.draw_thickness,
            step=1
        )

        self.erase_thickness = st.sidebar.slider(
            "🧽 Eraser Size:",
            min_value=5,
            max_value=30,
            value=self.erase_thickness,
            step=1
        )

    def create_solution_document(self, drawing_image, solution_text, problem_type):
        """Create a comprehensive document with drawing and solution."""
        try:
            # Create a new image for the document
            doc_width = 800
            doc_height = 1200
            doc_img = Image.new('RGB', (doc_width, doc_height), 'white')
            draw = ImageDraw.Draw(doc_img)

            # Try to load a font, fallback to default if not available
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                header_font = ImageFont.truetype("arial.ttf", 18)
                text_font = ImageFont.truetype("arial.ttf", 14)
            except:
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                text_font = ImageFont.load_default()

            y_offset = 20

            # Title
            title = "AI Math & CS Problem Solver - Solution Report"
            draw.text((20, y_offset), title, fill='black', font=title_font)
            y_offset += 40

            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw.text((20, y_offset), f"Generated on: {timestamp}", fill='gray', font=text_font)
            y_offset += 30

            # Problem type
            type_text = {"auto": "Auto-Detected", "math": "Mathematics", "cs": "Computer Science"}[problem_type]
            draw.text((20, y_offset), f"Problem Type: {type_text}", fill='blue', font=header_font)
            y_offset += 40

            # Drawing section
            draw.text((20, y_offset), "Your Drawing:", fill='black', font=header_font)
            y_offset += 30

            # Resize and paste the drawing
            if drawing_image:
                # Resize drawing to fit in document
                drawing_resized = drawing_image.resize((400, 300), Image.Resampling.LANCZOS)
                doc_img.paste(drawing_resized, (20, y_offset))
                y_offset += 320

            # Solution section
            draw.text((20, y_offset), "AI Solution:", fill='black', font=header_font)
            y_offset += 30

            # Process solution text (wrap long lines)
            solution_lines = solution_text.replace('**', '').split('\n')
            max_width = 750

            for line in solution_lines:
                if line.strip():
                    # Word wrap for long lines
                    words = line.split(' ')
                    current_line = ""

                    for word in words:
                        test_line = current_line + word + " "
                        # Approximate text width (rough calculation)
                        if len(test_line) * 8 < max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                draw.text((20, y_offset), current_line.strip(), fill='black', font=text_font)
                                y_offset += 20
                            current_line = word + " "

                    if current_line:
                        draw.text((20, y_offset), current_line.strip(), fill='black', font=text_font)
                        y_offset += 20
                else:
                    y_offset += 10

                # Check if we need more space
                if y_offset > doc_height - 100:
                    break

            # Footer
            footer_y = doc_height - 40
            draw.text((20, footer_y), "Generated by AI Math & CS Problem Solver", fill='gray', font=text_font)

            return doc_img

        except Exception as e:
            st.error(f"Error creating document: {str(e)}")
            return None

    def create_download_link(self, img, filename):
        """Create a download link for the image."""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        b64 = base64.b64encode(img_data).decode()

        return f'''
        <div style="margin: 15px 0;">
            <a href="data:image/png;base64,{b64}" download="{filename}" 
               style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                      color: white; padding: 10px 20px; text-decoration: none;
                      border-radius: 10px; font-weight: bold; display: inline-block;
                      box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                📥 Download Solution Report
            </a>
        </div>
        '''

    def analyze_drawing(self, pil_image, problem_type="auto"):
        try:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')

            if problem_type == "math":
                prompt = """
                Analyze this hand-drawn mathematical expression and provide a clean solution:

                1. **Expression Identified:** [Write the mathematical expression you see]
                2. **Step-by-Step Solution:**
                   - Show each calculation step clearly
                   - Explain the mathematical operations used
                3. **Final Answer:** [Provide the final result]
                4. **Verification:** [Optional: verify the answer]

                If the drawing is unclear or not mathematical, please indicate that clearly.
                """
            elif problem_type == "cs":
                prompt = """
                Analyze this hand-drawn computer science problem and provide a comprehensive solution:

                1. **Problem Identified:** [Describe what CS concept/problem you see]
                2. **Solution Approach:**
                   - Explain the algorithm or concept
                   - Provide step-by-step solution
                   - Include code examples if applicable
                3. **Time/Space Complexity:** [If applicable]
                4. **Key Points:** [Important concepts to remember]

                This could be algorithms, data structures, logic diagrams, flowcharts, pseudocode, etc.
                If unclear or not CS-related, please indicate that.
                """
            else:  # auto
                prompt = """
                Analyze this hand-drawn content and determine if it's:
                1. A mathematical expression/equation - solve it step by step
                2. A computer science problem (algorithm, data structure, logic, etc.) - provide comprehensive solution
                3. Something else - describe what you see

                For Math Problems:
                - **Expression:** [What you see]
                - **Solution:** [Step-by-step calculation]
                - **Answer:** [Final result]

                For CS Problems:
                - **Problem Type:** [Algorithm, Data Structure, etc.]
                - **Solution:** [Step-by-step approach]
                - **Implementation:** [Code if applicable]
                - **Complexity:** [Time/Space if applicable]

                Provide clear, educational explanations.
                """

            response = model.generate_content([prompt, pil_image])

            # Store for download feature
            self.last_drawing = pil_image.copy()
            self.last_solution = response.text
            self.last_problem_type = problem_type

            return ("success", response.text)
        except Exception as e:
            return ("error", f"Analysis failed: {str(e)}")

    def main(self):
        st.sidebar.title("⚙️ Control Panel")
        mode = st.sidebar.radio("Choose Input Mode:", ["✋ Gesture Mode", "📝 Canvas Mode"])

        # Problem type selection
        st.sidebar.markdown("---")
        problem_type = st.sidebar.selectbox(
            "🧠 Problem Type:",
            ["auto", "math", "cs"],
            format_func=lambda x: {"auto": "🔍 Auto-Detect", "math": "📊 Mathematics", "cs": "💻 Computer Science"}[x]
        )

        if mode == "✋ Gesture Mode":
            self.show_gesture_tutorial()
            self.show_drawing_controls()

            col1, col2 = st.columns([0.65, 0.35])

            with col1:
                st.markdown("### 📹 Camera Feed")
                video_frame = st.empty()
                canvas_overlay = st.empty()

            with col2:
                st.markdown("### 🧮 AI Analysis")
                gesture_status = st.empty()
                result_display = st.empty()

            if not self.cap:
                st.error("❌ No webcam available for gesture mode.")
                return

            if 'running' not in st.session_state:
                st.session_state.running = False

            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("▶ Start Camera", type="primary"):
                    st.session_state.running = True
            with col_btn2:
                if st.button("⏹ Stop Camera", type="secondary"):
                    st.session_state.running = False
            with col_btn3:
                if st.button("🗑️ Clear Canvas"):
                    self.imgCanvas = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

            # Main camera loop
            if st.session_state.running:
                while st.session_state.running:
                    if not self.process_frame():
                        st.error("❌ Could not read from webcam")
                        break

                    self.process_hands()
                    self.identify_fingers()
                    finger_count = self.handle_gestures()

                    # Combine camera image with canvas overlay
                    combined_img = cv2.addWeighted(self.img, 0.7, self.imgCanvas, 0.3, 0)

                    # Display current gesture status (simplified during analysis)
                    if not self.analysis_in_progress:
                        gesture_status.markdown(f"""
                        <div class="status-indicator status-ready">
                        <strong>Current Action:</strong> {self.current_gesture}<br>
                        <strong>Color:</strong> {self.current_color} | <strong>Thickness:</strong> {self.draw_thickness}px
                        </div>
                        """, unsafe_allow_html=True)

                    # Check for analysis requests
                    if not self.analysis_queue.empty() and not self.analysis_in_progress:
                        self.analysis_in_progress = True
                        try:
                            canvas_img = self.analysis_queue.get_nowait()
                            gesture_status.markdown("""
                            <div class="status-indicator status-analyzing">
                            🤖 AI is analyzing your drawing...
                            </div>
                            """, unsafe_allow_html=True)

                            status, result = self.analyze_drawing(canvas_img, problem_type)
                            if status == "success":
                                result_display.markdown(f"""
                                <div class="result-box">
                                <h4>✅ Analysis Complete!</h4>
                                {result.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
                                </div>
                                """, unsafe_allow_html=True)

                                # Show download option
                                if self.last_drawing is not None:
                                    try:
                                        doc_img = self.create_solution_document(
                                            self.last_drawing,
                                            self.last_solution,
                                            self.last_problem_type
                                        )

                                        if doc_img:
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            filename = f"solution_report_{timestamp}.png"

                                            buffered = io.BytesIO()
                                            doc_img.save(buffered, format="PNG")

                                            result_display.download_button(
                                                label="📥 Download Solution Report",
                                                data=buffered.getvalue(),
                                                file_name=filename,
                                                mime="image/png",
                                                type="primary"
                                            )
                                    except Exception as e:
                                        result_display.error(f"❌ Download preparation failed: {str(e)}")
                            else:
                                result_display.error(f"❌ {result}")
                        except:
                            pass
                        finally:
                            self.analysis_in_progress = False

                    video_frame.image(combined_img, channels="RGB", use_column_width=True)

                    # Show canvas separately for better visibility
                    if np.any(self.imgCanvas):
                        canvas_overlay.image(self.imgCanvas, channels="RGB", caption="Your Drawing",
                                             use_column_width=True)

                    time.sleep(0.033)  # ~30 FPS

        else:  # Canvas Mode
            st.markdown("### 📝 Draw your math or CS problem below:")

            # Canvas color options
            col1, col2, col3 = st.columns(3)
            with col1:
                stroke_color = st.color_picker("Drawing Color", "#000000")
            with col2:
                stroke_width = st.slider("Line Thickness", 1, 20, 5)
            with col3:
                bg_color = st.color_picker("Background Color", "#FFFFFF")

            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                width=640,
                height=480,
                drawing_mode="freedraw",
                key="canvas",
            )

            if st.button("🔍 Analyze Drawing", type="primary"):
                if canvas_result.image_data is not None:
                    img = PIL.Image.fromarray((canvas_result.image_data).astype("uint8"))
                    with st.spinner("🤖 AI is analyzing your drawing..."):
                        status, result = self.analyze_drawing(img, problem_type)

                    if status == "success":
                        st.markdown(f"""
                        <div class="result-box">
                        <h4>✅ Analysis Complete!</h4>
                        {result.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
                        </div>
                        """, unsafe_allow_html=True)

                        # Add download button for canvas mode
                        if self.last_drawing is not None:
                            try:
                                doc_img = self.create_solution_document(
                                    self.last_drawing,
                                    self.last_solution,
                                    self.last_problem_type
                                )

                                if doc_img:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"solution_report_{timestamp}.png"

                                    buffered = io.BytesIO()
                                    doc_img.save(buffered, format="PNG")

                                    st.download_button(
                                        label="📥 Download Solution Report",
                                        data=buffered.getvalue(),
                                        file_name=filename,
                                        mime="image/png",
                                        type="primary",
                                        help="Download a comprehensive report with your drawing and the AI solution"
                                    )
                            except Exception as e:
                                st.error(f"❌ Download preparation failed: {str(e)}")
                    else:
                        st.error(f"❌ {result}")

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.cleanup()


def main():
    try:
        calc = VirtualCalculator()
        calc.streamlit_config()
        if not os.environ.get('GOOGLE_API_KEY'):
            st.error("⚠ **Missing API Key**: Please set your Google API key in the environment variables.")
            st.markdown("""
            <div class="gesture-guide">
            <h4>🔑 Setup Instructions:</h4>
            <p>1. Get your API key from Google AI Studio</p>
            <p>2. Set environment variable: <code>GOOGLE_API_KEY=your_key_here</code></p>
            <p>3. Restart the application</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        calc.main()
    except KeyboardInterrupt:
        st.info("👋 Application stopped by user")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.markdown("""
        <div class="gesture-guide">
        <h4>🔧 Troubleshooting:</h4>
        <p>• Ensure your webcam is connected and working</p>
        <p>• Check GOOGLE_API_KEY in environment variables</p>
        <p>• Install required dependencies:</p>
        <code>pip install streamlit opencv-python mediapipe google-generativeai pillow numpy python-dotenv streamlit-extras streamlit-drawable-canvas</code>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()