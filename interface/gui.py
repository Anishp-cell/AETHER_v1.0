import tkinter as tk
import math
import colorsys
import time
import cv2

try:
    from PIL import Image, ImageTk
except ImportError:
    Image, ImageTk = None, None

class AetherGUI:
    def __init__(self, root, on_mode_change=None, on_exit=None):
        self.root = root
        self.root.title("AETHER v2.0 - Hologram")
        
        # Make the window frameless and floating
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        # Set dimensions
        self.width = 400
        self.height = 400
        
        # Center the window on screen initially
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (self.width // 2) + 250
        y = (screen_height // 2) - (self.height // 2)
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.root.configure(bg='black')

        self._drag_data = {"x": 0, "y": 0}

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<Double-Button-1>", self._trigger_exit)

        self.hue = 0.0
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.mic_energy = 0.0
        self.is_speaking = False
        self.is_listening = False

        self.mode = "Wake Word"
        self.on_mode_change = on_mode_change
        self.on_exit_callback = on_exit

        self.canvas.create_text(
            self.width // 2, 20, 
            text="AETHER V2.0", 
            fill="white", 
            font=("Arial", 10, "bold"), 
            tags="ui_title"
        )
        self.mode_text_id = self.canvas.create_text(
            self.width // 2, self.height - 20, 
            text=f"Mode: {self.mode}", 
            fill="white", 
            font=("Arial", 10), 
            tags="ui_mode"
        )
        self.canvas.bind("<ButtonPress-3>", self.toggle_mode)

        # 3D Sphere Geometry Nodes Setup (latitudes and longitudes)
        self.nodes = []
        self.edges = []
        self.init_3d_sphere()

        # Camera Window Initialization
        self.cam_window = None
        self.cam_label = None
        self.init_camera_window(screen_width, screen_height)

        self.animate()

    def init_3d_sphere(self):
        # Generate points of a sphere
        lats = 12
        lons = 12
        radius = 1.0 # base scale
        for i in range(lats + 1):
            theta = i * math.pi / lats
            for j in range(lons):
                phi = j * 2 * math.pi / lons
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)
                self.nodes.append([x, y, z])
                
        # Generate edges
        for i in range(lats):
            for j in range(lons):
                curr = i * lons + j
                next_in_lat = i * lons + (j + 1) % lons
                next_in_lon = (i + 1) * lons + j
                self.edges.append((curr, next_in_lat))
                if i < lats - 1:
                    self.edges.append((curr, next_in_lon))

    def init_camera_window(self, sw, sh):
        self.cam_window = tk.Toplevel(self.root)
        self.cam_window.title("AETHER Vision Feed")
        self.cam_window.geometry(f"400x320+{sw//2 - 450}+{sh//2 - 160}")
        self.cam_window.configure(bg='black')
        
        # If user closes camera window, just hide it or recreate? 
        # For a HUD, we can just intercept close to prevent error.
        self.cam_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        self.cam_label = tk.Label(self.cam_window, bg='black')
        self.cam_label.pack(fill=tk.BOTH, expand=True)
        tk.Label(self.cam_window, text="[ LIVE SENSOR FEED ]", fg="cyan", bg="black", font=("Terminal", 10)).pack(side=tk.BOTTOM)

    def on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_drag_motion(self, event):
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + delta_x
        y = self.root.winfo_y() + delta_y
        self.root.geometry(f"+{x}+{y}")

    def _trigger_exit(self, event=None):
        if self.on_exit_callback:
            self.on_exit_callback()
        self.root.quit()

    def toggle_mode(self, event=None):
        self.mode = "Push-to-Talk" if self.mode == "Wake Word" else "Wake Word"
        if self.on_mode_change:
            self.on_mode_change(self.mode)

    def set_mic_energy(self, energy):
        normalized = min(energy / 2000.0, 1.0)
        self.mic_energy = normalized

    def set_speaking_state(self, is_speaking):
        self.is_speaking = is_speaking
        
    def set_listening_state(self, is_listening):
        self.is_listening = is_listening

    def get_rgb_color(self, saturation=1.0, value=1.0):
        r, g, b = colorsys.hsv_to_rgb(self.hue, saturation, value)
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    def set_camera_frame(self, frame):
        if frame is None or Image is None:
            return
        try:
            # frame is raw BGR from OpenCV
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2_im = cv2.resize(cv2_im, (400, 300))
            pil_im = Image.fromarray(cv2_im)
            
            # Apply a cyan sci-fi tint filter? Let's just keep it raw but maybe transparently
            self.photo = ImageTk.PhotoImage(image=pil_im)
            self.cam_label.configure(image=self.photo)
        except Exception as e:
            print(f"Vision feed err: {e}")

    def rotate_3d(self, node, angle_x, angle_y):
        x, y, z = node
        # Rotate X
        cos_x = math.cos(angle_x)
        sin_x = math.sin(angle_x)
        y1 = y * cos_x - z * sin_x
        z1 = y * sin_x + z * cos_x
        # Rotate Y
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        x2 = x * cos_y + z1 * sin_y
        z2 = -x * sin_y + z1 * cos_y
        return [x2, y1, z2]

    def animate(self):
        self.canvas.delete("hologram")

        cx = self.width // 2
        cy = self.height // 2
        
        self.hue = (self.hue + 0.005) % 1.0
        self.angle_x += 0.02
        self.angle_y += 0.03
        color = self.get_rgb_color()

        pulse_scale = 80 + (self.mic_energy * 30)

        # Draw 3D wireframe
        projected = []
        
        sphere_scale = pulse_scale
        if self.is_speaking:
            sphere_scale = 130 # much larger when speaking
            self.canvas.itemconfigure(self.mode_text_id, text="[ AETHER IS SPEAKING ]", fill=color)
        else:
            text_str = f"Mode: {self.mode}\n(Right-click to switch)"
            if self.is_listening:
                text_str = "[ LISTENING ... ]"
                sphere_scale += math.sin(time.time() * 10) * 10
            self.canvas.itemconfigure(self.mode_text_id, text=text_str, fill="white")

        for node in self.nodes:
            # scale
            sn = [node[0] * sphere_scale, node[1] * sphere_scale, node[2] * sphere_scale]
            # rotate
            rn = self.rotate_3d(sn, self.angle_x, self.angle_y)
            # projection (simple orthographic for ease, adding slight perspective makes it pop)
            z = rn[2] + 400 # push back
            f = 400 / z
            px = int(cx + rn[0] * f)
            py = int(cy + rn[1] * f)
            projected.append((px, py))
            
            # draw nodes if speaking
            if self.is_speaking:
                self.canvas.create_oval(px-1, py-1, px+1, py+1, fill=color, outline=color, tags="hologram")

        lineWidth = 2 if self.is_speaking else 1
        lineColor = color if not self.is_listening else "white"

        for edge in self.edges:
            p1 = projected[edge[0]]
            p2 = projected[edge[1]]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=lineColor, width=lineWidth, tags="hologram")

        self.root.after(30, self.animate)

if __name__ == "__main__":
    def dummy_exit():
        print("Exiting UI")
        
    root = tk.Tk()
    gui = AetherGUI(root, on_exit=dummy_exit)
    # Simulate some mic energy and speaking state for testing
    def simulate():
        gui.set_mic_energy(300 + 400 * math.sin(time.time() * 5))
        root.after(50, simulate)
    simulate()
    root.mainloop()
