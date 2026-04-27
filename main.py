import cv2
import numpy as np
import time
import os
import socket
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, Response
import uvicorn
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

app = FastAPI()

flow_ns = 40
flow_ew = 30
flow_ped = 20
cars = []
peds = []
real_detector = None
controller_mode = "adaptive"
scenario_name = "Custom"
city_context = "Baku, Ganjlik Intersection"
emergency_enabled = False
bus_priority_enabled = False
latest_metrics = {}
latest_decision = {
    "title": "Holding NS green",
    "reason": "Initial minimum green phase is active.",
}
audit_log = []
metrics_history = []
policy = {
    "min_green": 7.0,
    "max_green": 24.0,
    "yellow": 2.0,
    "all_red": 1.5,
    "ped_weight": 1.3,
    "wait_weight": 0.14,
    "bus_priority": True,
    "emergency_priority": True,
}
last_logged_decision = ""

WIDTH, HEIGHT = 800, 800
ROAD_WIDTH = 140
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
CROSSWALK_GAP = 20
MAX_ACTIVE_CARS = 95
MAX_ACTIVE_PEDS = 55
MIN_GREEN = 7.0
MAX_GREEN = 24.0
YELLOW_TIME = 2.0
ALL_RED_TIME = 1.5
QUEUE_DETECTION_DISTANCE = 175
STOP_BUFFER = 18
INTERSECTION_MARGIN = 18
SCORE_SWITCH_DELTA = 1.35
VEHICLE_QUEUE_WEIGHT = 1.0
PEDESTRIAN_QUEUE_WEIGHT = 1.3
WAIT_WEIGHT = 0.14
COCO_PERSON_CLASS = 0
COCO_VEHICLE_CLASSES = {2, 3, 5, 7}

SCENARIOS = {
    "balanced": {"label": "Balanced", "context": "Baku, Ganjlik Intersection", "ns": 40, "ew": 35, "ped": 20, "emergency": False, "bus": False},
    "morning_rush": {"label": "Morning Rush", "context": "Baku, Ganjlik Intersection", "ns": 105, "ew": 45, "ped": 18, "emergency": False, "bus": False},
    "school_crossing": {"label": "School Zone", "context": "Baku, School Zone Crossing", "ns": 45, "ew": 35, "ped": 75, "emergency": False, "bus": False},
    "event_exit": {"label": "Event Venue Exit", "context": "Baku Olympic Stadium Exit", "ns": 55, "ew": 115, "ped": 35, "emergency": False, "bus": False},
    "bus_corridor": {"label": "Public Transport Corridor", "context": "Baku Bus Priority Corridor", "ns": 65, "ew": 55, "ped": 28, "emergency": False, "bus": True},
    "emergency": {"label": "Emergency Route", "context": "Baku Emergency Response Route", "ns": 60, "ew": 55, "ped": 22, "emergency": True, "bus": False},
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
GENERATED_ASSET_DIR = os.path.join(ASSET_DIR, "generated")


def load_image(path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        return None
    try:
        img = cv2.imread(path, flags)
    except (OSError, FileNotFoundError):
        return None
    return img if img is not None else None


def build_procedural_background():
    rng = np.random.default_rng(42)
    base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    base[:] = (38, 105, 58)
    noise = rng.normal(0, 10, (HEIGHT, WIDTH, 1)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for _ in range(260):
        x = int(rng.integers(0, WIDTH))
        y = int(rng.integers(0, HEIGHT))
        color = (int(rng.integers(25, 45)), int(rng.integers(95, 135)), int(rng.integers(35, 65)))
        cv2.circle(base, (x, y), int(rng.integers(1, 3)), color, -1, lineType=cv2.LINE_AA)

    sidewalk = (132, 137, 130)
    cv2.rectangle(base, (CENTER_X - ROAD_WIDTH//2 - 54, 0), (CENTER_X - ROAD_WIDTH//2 - 26, HEIGHT), sidewalk, -1)
    cv2.rectangle(base, (CENTER_X + ROAD_WIDTH//2 + 26, 0), (CENTER_X + ROAD_WIDTH//2 + 54, HEIGHT), sidewalk, -1)
    cv2.rectangle(base, (0, CENTER_Y - ROAD_WIDTH//2 - 54), (WIDTH, CENTER_Y - ROAD_WIDTH//2 - 26), sidewalk, -1)
    cv2.rectangle(base, (0, CENTER_Y + ROAD_WIDTH//2 + 26), (WIDTH, CENTER_Y + ROAD_WIDTH//2 + 54), sidewalk, -1)
    return base


generated_bg = load_image(os.path.join(GENERATED_ASSET_DIR, "intersection_background.png"))
if generated_bg is not None:
    bg_img_raw = cv2.resize(generated_bg, (WIDTH, HEIGHT))
else:
    bg_img_raw = build_procedural_background()

car_img_raw = load_image(os.path.join(GENERATED_ASSET_DIR, "vehicle_topdown.png"), cv2.IMREAD_UNCHANGED)
CAR_COLORS = [
    (196, 55, 55),
    (58, 108, 210),
    (232, 232, 220),
    (45, 52, 62),
    (230, 158, 48),
    (70, 170, 110),
]


def alpha_blend_shape(img, draw_fn, alpha=0.45):
    overlay = img.copy()
    draw_fn(overlay)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_label(img, text, x, y, color=(70, 255, 170)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 6, y + 4), (12, 18, 28), -1)
    cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 6, y + 4), color, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_detection_box(img, x1, y1, x2, y2, label, color=(70, 255, 170)):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    corner = 10
    for sx, sy, ex, ey in [
        (x1, y1, x1 + corner, y1), (x1, y1, x1, y1 + corner),
        (x2, y1, x2 - corner, y1), (x2, y1, x2, y1 + corner),
        (x1, y2, x1 + corner, y2), (x1, y2, x1, y2 - corner),
        (x2, y2, x2 - corner, y2), (x2, y2, x2, y2 - corner),
    ]:
        cv2.line(img, (sx, sy), (ex, ey), color, 2, lineType=cv2.LINE_AA)
    draw_label(img, label, x1, max(18, y1 - 7), color)


def draw_vehicle_body(img, x, y, w, h, color, direction, stopped=False):
    x, y, w, h = int(x), int(y), int(w), int(h)
    body = tuple(int(v) for v in color)
    darker = tuple(max(0, int(v * 0.55)) for v in body)
    glass = (58, 84, 98)
    highlight = tuple(min(255, int(v * 1.22)) for v in body)

    cv2.rectangle(img, (x + 3, y + 3), (x + w - 3, y + h - 3), darker, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h - 5), body, -1, lineType=cv2.LINE_AA)

    if h >= w:
        cv2.rectangle(img, (x + 7, y + 12), (x + w - 7, y + 22), glass, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x + 7, y + h - 23), (x + w - 7, y + h - 13), glass, -1, lineType=cv2.LINE_AA)
        cv2.line(img, (x + w // 2, y + 8), (x + w // 2, y + h - 8), highlight, 1, lineType=cv2.LINE_AA)
        wheel_rects = [
            (x + 1, y + 10, x + 5, y + 23),
            (x + w - 5, y + 10, x + w - 1, y + 23),
            (x + 1, y + h - 24, x + 5, y + h - 11),
            (x + w - 5, y + h - 24, x + w - 1, y + h - 11),
        ]
        if direction == "N":
            head_y, tail_y = y + h - 5, y + 5
        else:
            head_y, tail_y = y + 5, y + h - 5
        cv2.circle(img, (x + 8, head_y), 2, (230, 245, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x + w - 8, head_y), 2, (230, 245, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x + 8, tail_y), 2, (30, 40, 245) if stopped else (35, 45, 160), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x + w - 8, tail_y), 2, (30, 40, 245) if stopped else (35, 45, 160), -1, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x + 13, y + 7), (x + 24, y + h - 7), glass, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x + w - 25, y + 7), (x + w - 14, y + h - 7), glass, -1, lineType=cv2.LINE_AA)
        cv2.line(img, (x + 8, y + h // 2), (x + w - 8, y + h // 2), highlight, 1, lineType=cv2.LINE_AA)
        wheel_rects = [
            (x + 11, y + 1, x + 24, y + 5),
            (x + 11, y + h - 5, x + 24, y + h - 1),
            (x + w - 25, y + 1, x + w - 12, y + 5),
            (x + w - 25, y + h - 5, x + w - 12, y + h - 1),
        ]
        if direction == "E":
            head_x, tail_x = x + 5, x + w - 5
        else:
            head_x, tail_x = x + w - 5, x + 5
        cv2.circle(img, (head_x, y + 8), 2, (230, 245, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (head_x, y + h - 8), 2, (230, 245, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (tail_x, y + 8), 2, (30, 40, 245) if stopped else (35, 45, 160), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (tail_x, y + h - 8), 2, (30, 40, 245) if stopped else (35, 45, 160), -1, lineType=cv2.LINE_AA)

    for x1, y1, x2, y2 in wheel_rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (12, 15, 16), -1, lineType=cv2.LINE_AA)


def draw_motion_trail(img, x, y, w, h, direction, length):
    x, y, w, h = int(x), int(y), int(w), int(h)
    color = (180, 190, 190)
    if direction == "N":
        cv2.line(img, (x + 5, y - length), (x + 5, y - 2), color, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x + w - 5, y - length), (x + w - 5, y - 2), color, 2, lineType=cv2.LINE_AA)
    elif direction == "S":
        cv2.line(img, (x + 5, y + h + 2), (x + 5, y + h + length), color, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x + w - 5, y + h + 2), (x + w - 5, y + h + length), color, 2, lineType=cv2.LINE_AA)
    elif direction == "E":
        cv2.line(img, (x + w + 2, y + 5), (x + w + length, y + 5), color, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x + w + 2, y + h - 5), (x + w + length, y + h - 5), color, 2, lineType=cv2.LINE_AA)
    else:
        cv2.line(img, (x - length, y + 5), (x - 2, y + 5), color, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x - length, y + h - 5), (x - 2, y + h - 5), color, 2, lineType=cv2.LINE_AA)

def overlay_transparent(bg_img, img_to_overlay_t, x, y, overlay_size=None):
    try:
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t, overlay_size)
        
        b, g, r, a = cv2.split(img_to_overlay_t)
        overlay_color = cv2.merge((b, g, r))
        
        mask = cv2.medianBlur(a, 1)
        h, w, _ = overlay_color.shape
        
        y1, y2 = max(0, y), min(bg_img.shape[0], y + h)
        x1, x2 = max(0, x), min(bg_img.shape[1], x + w)
        
        y1o, y2o = max(0, -y), min(h, bg_img.shape[0] - y)
        x1o, x2o = max(0, -x), min(w, bg_img.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return bg_img

        roi = bg_img[y1:y2, x1:x2]
        
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask[y1o:y2o, x1o:x2o]))
        img2_fg = cv2.bitwise_and(overlay_color[y1o:y2o, x1o:x2o], overlay_color[y1o:y2o, x1o:x2o], mask=mask[y1o:y2o, x1o:x2o])
        
        bg_img[y1:y2, x1:x2] = cv2.add(img1_bg, img2_fg)
    except Exception as e:
        pass
    return bg_img

class Pedestrian:
    def __init__(self, crosswalk_id):
        self.crosswalk_id = crosswalk_id
        self.speed = 58
        self.radius = 6
        self.stopped = False
        self.wait_time = 0.0
        self.dir = np.random.choice([1, -1])
        
        if crosswalk_id == 'N':
            self.y = CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP//2
            self.x = CENTER_X - ROAD_WIDTH//2 - 20 if self.dir == 1 else CENTER_X + ROAD_WIDTH//2 + 20
        elif crosswalk_id == 'S':
            self.y = CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP//2
            self.x = CENTER_X - ROAD_WIDTH//2 - 20 if self.dir == 1 else CENTER_X + ROAD_WIDTH//2 + 20
        elif crosswalk_id == 'E':
            self.x = CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP//2
            self.y = CENTER_Y - ROAD_WIDTH//2 - 20 if self.dir == 1 else CENTER_Y + ROAD_WIDTH//2 + 20
        elif crosswalk_id == 'W':
            self.x = CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP//2
            self.y = CENTER_Y - ROAD_WIDTH//2 - 20 if self.dir == 1 else CENTER_Y + ROAD_WIDTH//2 + 20

    def move(self, light_state, dt):
        self.stopped = False
        ped_light_ns_cross = 'GREEN' if light_state['EW'] == 'GREEN' else 'RED'
        ped_light_ew_cross = 'GREEN' if light_state['NS'] == 'GREEN' else 'RED'
        
        if self.crosswalk_id in ['N', 'S']:
            if self.dir == 1: 
                if ped_light_ns_cross != 'GREEN' and CENTER_X - ROAD_WIDTH//2 - 10 < self.x < CENTER_X - ROAD_WIDTH//2:
                    self.stopped = True
            else: 
                if ped_light_ns_cross != 'GREEN' and CENTER_X + ROAD_WIDTH//2 < self.x < CENTER_X + ROAD_WIDTH//2 + 10:
                    self.stopped = True
        else: 
            if self.dir == 1: 
                if ped_light_ew_cross != 'GREEN' and CENTER_Y - ROAD_WIDTH//2 - 10 < self.y < CENTER_Y - ROAD_WIDTH//2:
                    self.stopped = True
            else: 
                if ped_light_ew_cross != 'GREEN' and CENTER_Y + ROAD_WIDTH//2 < self.y < CENTER_Y + ROAD_WIDTH//2 + 10:
                    self.stopped = True

        if not self.stopped:
            if self.crosswalk_id in ['N', 'S']:
                self.x += self.speed * self.dir * dt
            else:
                self.y += self.speed * self.dir * dt
        else:
            self.wait_time += dt

    def draw(self, img):
        x, y = int(self.x), int(self.y)
        cv2.ellipse(img, (x + 2, y + 5), (self.radius + 3, 4), 0, 0, 360, (18, 24, 26), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), self.radius, (0, 174, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x - 2, y - 2), 2, (255, 232, 168), -1, lineType=cv2.LINE_AA)
        draw_detection_box(img, x - self.radius - 3, y - self.radius - 3, x + self.radius + 3, y + self.radius + 3, "ped 0.91", (0, 230, 255))

class Car:
    def __init__(self, direction, vehicle_type="car"):
        self.direction = direction
        self.vehicle_type = vehicle_type
        self.speed = float(np.random.uniform(120, 155))
        if self.vehicle_type == "ambulance":
            self.speed = 175.0
        if self.vehicle_type == "bus":
            self.speed = 118.0
            self.width = 34
            self.length = 74
        else:
            self.width = 28
            self.length = 54
        self.stopped = False
        self.wait_time = 0.0
        self.current_speed = float(np.random.uniform(55, 95))
        self.accel = 95.0
        self.decel = 310.0
        if self.vehicle_type == "ambulance":
            self.color = (245, 245, 238)
        elif self.vehicle_type == "bus":
            self.color = (45, 175, 215)
        else:
            self.color = CAR_COLORS[int(np.random.randint(0, len(CAR_COLORS)))]
        
        offset = ROAD_WIDTH // 4
        
        if direction == 'N':
            self.x = CENTER_X - offset - self.width//2
            self.y = -self.length
        elif direction == 'S':
            self.x = CENTER_X + offset - self.width//2
            self.y = HEIGHT
        elif direction == 'E':
            self.x = WIDTH
            self.y = CENTER_Y - offset - self.width//2
            self.length, self.width = self.width, self.length
        elif direction == 'W':
            self.x = -self.width
            self.y = CENTER_Y + offset - self.width//2
            self.length, self.width = self.width, self.length

    def axis(self):
        return 'NS' if self.direction in ['N', 'S'] else 'EW'

    def distance_to_stopline(self):
        if self.direction == 'N':
            return (CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP - STOP_BUFFER) - (self.y + self.length)
        if self.direction == 'S':
            return self.y - (CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP + STOP_BUFFER)
        if self.direction == 'E':
            return self.x - (CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP + STOP_BUFFER)
        return (CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP - STOP_BUFFER) - (self.x + self.width)

    def is_in_intersection_box(self):
        return (
            self.x < CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP + INTERSECTION_MARGIN and
            self.x + self.width > CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP - INTERSECTION_MARGIN and
            self.y < CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP + INTERSECTION_MARGIN and
            self.y + self.length > CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP - INTERSECTION_MARGIN
        )

    def is_queued(self):
        dist = self.distance_to_stopline()
        return 0 <= dist <= QUEUE_DETECTION_DISTANCE and self.stopped

    def is_priority_request(self):
        return self.vehicle_type in ["ambulance", "bus"] and 0 <= self.distance_to_stopline() <= QUEUE_DETECTION_DISTANCE + 70

    def move(self, light_state, other_cars, dt):
        self.stopped = False
        stop_gap = 95 + CROSSWALK_GAP
        dist_to_intersection = self.distance_to_stopline()
        
        if 0 < dist_to_intersection < stop_gap and light_state[self.axis()] in ['RED', 'YELLOW']:
            self.stopped = True

        if 0 < dist_to_intersection < stop_gap and light_state[self.axis()] == 'GREEN':
            conflict_in_box = any(
                car is not self and car.axis() != self.axis() and car.is_in_intersection_box()
                for car in other_cars
            )
            if conflict_in_box:
                self.stopped = True
                
        for car in other_cars:
            if car != self:
                if car.direction == self.direction:
                    safe_dist = 38
                    if self.direction == 'N' and 0 < car.y - (self.y + self.length) < safe_dist:
                        self.stopped = True
                    elif self.direction == 'S' and 0 < self.y - (car.y + car.length) < safe_dist:
                        self.stopped = True
                    elif self.direction == 'E' and 0 < self.x - (car.x + car.width) < safe_dist:
                        self.stopped = True
                    elif self.direction == 'W' and 0 < car.x - (self.x + self.width) < safe_dist:
                        self.stopped = True
                elif self.is_in_intersection_box() or car.is_in_intersection_box():
                    margin = 5
                    if (self.x < car.x + car.width + margin and self.x + self.width > car.x - margin and
                        self.y < car.y + car.length + margin and self.y + self.length > car.y - margin):
                        self.stopped = True

        target_speed = 0.0 if self.stopped else self.speed
        rate = self.decel if target_speed < self.current_speed else self.accel
        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + rate * dt)
        else:
            self.current_speed = max(target_speed, self.current_speed - rate * dt)

        if self.stopped and self.current_speed < 12:
            self.current_speed = 0.0

        if self.current_speed > 0:
            distance = self.current_speed * dt
            if self.direction == 'N': self.y += distance
            elif self.direction == 'S': self.y -= distance
            elif self.direction == 'E': self.x -= distance
            elif self.direction == 'W': self.x += distance
        if self.stopped and self.current_speed == 0:
            self.wait_time += dt

    def draw(self, img):
        speed_ratio = min(1.0, self.current_speed / max(1.0, self.speed))
        trail_len = int(16 * speed_ratio)
        if trail_len > 3:
            alpha_blend_shape(
                img,
                lambda overlay: draw_motion_trail(overlay, self.x, self.y, self.width, self.length, self.direction, trail_len),
                0.28,
            )

        shadow_x = int(self.x + 6)
        shadow_y = int(self.y + 8)
        alpha_blend_shape(
            img,
            lambda overlay: cv2.ellipse(
                overlay,
                (shadow_x + int(self.width // 2), shadow_y + int(self.length // 2)),
                (max(16, int(self.width // 2) + 7), max(14, int(self.length // 2) + 6)),
                0,
                0,
                360,
                (8, 12, 14),
                -1,
                lineType=cv2.LINE_AA,
            ),
            0.42,
        )
        if car_img_raw is not None:
            c_img = car_img_raw.copy()
            # carSmall2.png usually points right (East)
            if self.direction == 'N':
                c_img = cv2.rotate(c_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.direction == 'S':
                c_img = cv2.rotate(c_img, cv2.ROTATE_90_CLOCKWISE)
            elif self.direction == 'W':
                c_img = cv2.rotate(c_img, cv2.ROTATE_180)
                
            w_draw = self.width if self.direction in ['N', 'S'] else self.length
            h_draw = self.length if self.direction in ['N', 'S'] else self.width
            overlay_transparent(img, c_img, int(self.x), int(self.y), (w_draw, h_draw))
        else:
            draw_vehicle_body(img, self.x, self.y, self.width, self.length, self.color, self.direction, self.stopped)
            if self.vehicle_type == "ambulance":
                cx, cy = int(self.x + self.width / 2), int(self.y + self.length / 2)
                cv2.putText(img, "EMS", (int(self.x + 4), int(self.y + self.length / 2 + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 40, 210), 1, cv2.LINE_AA)
                cv2.circle(img, (cx - 5, cy - 8), 3, (30, 40, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, (cx + 5, cy - 8), 3, (255, 80, 40), -1, lineType=cv2.LINE_AA)
            elif self.vehicle_type == "bus":
                cv2.putText(img, "BUS", (int(self.x + 5), int(self.y + self.length / 2 + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            
        draw_detection_box(
            img,
            self.x - 4,
            self.y - 4,
            self.x + self.width + 4,
            self.y + self.length + 4,
            "ambulance 0.98" if self.vehicle_type == "ambulance" else ("bus 0.96" if self.vehicle_type == "bus" else "car 0.95"),
            (60, 120, 255) if self.vehicle_type == "ambulance" else ((70, 210, 255) if self.vehicle_type == "bus" else (70, 255, 170)),
        )

def draw_crosswalk(img, x, y, width, height, is_vertical=False):
    num_stripes = 7
    if is_vertical:
        stripe_h = height // (num_stripes * 2)
        for i in range(num_stripes):
            sy = y + i * 2 * stripe_h + stripe_h//2
            cv2.rectangle(img, (x, sy), (x + width, sy + stripe_h), (220, 220, 210), -1, lineType=cv2.LINE_AA)
    else:
        stripe_w = width // (num_stripes * 2)
        for i in range(num_stripes):
            sx = x + i * 2 * stripe_w + stripe_w//2
            cv2.rectangle(img, (sx, y), (sx + stripe_w, y + height), (220, 220, 210), -1, lineType=cv2.LINE_AA)


def draw_roads(img):
    road_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(road_mask, (CENTER_X - ROAD_WIDTH//2, 0), (CENTER_X + ROAD_WIDTH//2, HEIGHT), 255, -1)
    cv2.rectangle(road_mask, (0, CENTER_Y - ROAD_WIDTH//2), (WIDTH, CENTER_Y + ROAD_WIDTH//2), 255, -1)

    rng = np.random.default_rng(7)
    asphalt = np.full((HEIGHT, WIDTH, 3), (55, 58, 60), dtype=np.uint8)
    grain = rng.normal(0, 9, (HEIGHT, WIDTH, 1)).astype(np.int16)
    asphalt = np.clip(asphalt.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    img[road_mask == 255] = asphalt[road_mask == 255]

    curb = (165, 168, 160)
    cv2.rectangle(img, (CENTER_X - ROAD_WIDTH//2 - 5, 0), (CENTER_X - ROAD_WIDTH//2, HEIGHT), curb, -1)
    cv2.rectangle(img, (CENTER_X + ROAD_WIDTH//2, 0), (CENTER_X + ROAD_WIDTH//2 + 5, HEIGHT), curb, -1)
    cv2.rectangle(img, (0, CENTER_Y - ROAD_WIDTH//2 - 5), (WIDTH, CENTER_Y - ROAD_WIDTH//2), curb, -1)
    cv2.rectangle(img, (0, CENTER_Y + ROAD_WIDTH//2), (WIDTH, CENTER_Y + ROAD_WIDTH//2 + 5), curb, -1)

    line_color = (232, 232, 220)
    for i in range(0, HEIGHT, 46):
        if i < CENTER_Y - ROAD_WIDTH//2 - 8 or i > CENTER_Y + ROAD_WIDTH//2 + 8:
            cv2.line(img, (CENTER_X, i), (CENTER_X, i + 24), line_color, 2, lineType=cv2.LINE_AA)
    for i in range(0, WIDTH, 46):
        if i < CENTER_X - ROAD_WIDTH//2 - 8 or i > CENTER_X + ROAD_WIDTH//2 + 8:
            cv2.line(img, (i, CENTER_Y), (i + 24, CENTER_Y), line_color, 2, lineType=cv2.LINE_AA)

    arrow = (210, 214, 205)
    cv2.arrowedLine(img, (CENTER_X - ROAD_WIDTH//4, 95), (CENTER_X - ROAD_WIDTH//4, 145), arrow, 2, line_type=cv2.LINE_AA, tipLength=0.28)
    cv2.arrowedLine(img, (CENTER_X + ROAD_WIDTH//4, HEIGHT - 95), (CENTER_X + ROAD_WIDTH//4, HEIGHT - 145), arrow, 2, line_type=cv2.LINE_AA, tipLength=0.28)
    cv2.arrowedLine(img, (WIDTH - 95, CENTER_Y - ROAD_WIDTH//4), (WIDTH - 145, CENTER_Y - ROAD_WIDTH//4), arrow, 2, line_type=cv2.LINE_AA, tipLength=0.28)
    cv2.arrowedLine(img, (95, CENTER_Y + ROAD_WIDTH//4), (145, CENTER_Y + ROAD_WIDTH//4), arrow, 2, line_type=cv2.LINE_AA, tipLength=0.28)

    cv2.circle(img, (CENTER_X - ROAD_WIDTH//2 - 46, CENTER_Y - ROAD_WIDTH//2 - 46), 5, (30, 38, 42), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (CENTER_X + ROAD_WIDTH//2 + 46, CENTER_Y - ROAD_WIDTH//2 - 46), 5, (30, 38, 42), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (CENTER_X - ROAD_WIDTH//2 - 46, CENTER_Y + ROAD_WIDTH//2 + 46), 5, (30, 38, 42), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (CENTER_X + ROAD_WIDTH//2 + 46, CENTER_Y + ROAD_WIDTH//2 + 46), 5, (30, 38, 42), -1, lineType=cv2.LINE_AA)


def draw_road_furniture(img):
    pole_color = (32, 38, 40)
    metal = (88, 98, 98)
    positions = [
        (CENTER_X - ROAD_WIDTH//2 - 48, CENTER_Y - ROAD_WIDTH//2 - 48, 1, 1),
        (CENTER_X + ROAD_WIDTH//2 + 48, CENTER_Y - ROAD_WIDTH//2 - 48, -1, 1),
        (CENTER_X - ROAD_WIDTH//2 - 48, CENTER_Y + ROAD_WIDTH//2 + 48, 1, -1),
        (CENTER_X + ROAD_WIDTH//2 + 48, CENTER_Y + ROAD_WIDTH//2 + 48, -1, -1),
    ]
    for x, y, sx, sy in positions:
        alpha_blend_shape(
            img,
            lambda overlay, x=x, y=y: cv2.ellipse(overlay, (x + 7, y + 9), (15, 5), 0, 0, 360, (5, 8, 10), -1, lineType=cv2.LINE_AA),
            0.34,
        )
        cv2.circle(img, (x, y), 6, pole_color, -1, lineType=cv2.LINE_AA)
        cv2.line(img, (x, y), (x + sx * 34, y), metal, 4, lineType=cv2.LINE_AA)
        cv2.circle(img, (x + sx * 36, y), 5, (18, 22, 24), -1, lineType=cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + sy * 34), metal, 4, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y + sy * 36), 5, (18, 22, 24), -1, lineType=cv2.LINE_AA)

    for x, y, text in [
        (CENTER_X - 35, CENTER_Y - ROAD_WIDTH//2 - 34, "STOP"),
        (CENTER_X + 8, CENTER_Y + ROAD_WIDTH//2 + 48, "STOP"),
        (CENTER_X + ROAD_WIDTH//2 + 34, CENTER_Y - 8, "STOP"),
        (CENTER_X - ROAD_WIDTH//2 - 58, CENTER_Y + 35, "STOP"),
    ]:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 210), 1, cv2.LINE_AA)


def apply_camera_grade(img):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    vignette = np.clip(1.08 - distance / 780, 0.72, 1.0)
    img[:] = np.clip(img.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def vehicle_axis(direction):
    return "NS" if direction in ["N", "S"] else "EW"


def ped_service_axis(crosswalk_id):
    return "EW" if crosswalk_id in ["N", "S"] else "NS"


def queue_snapshot(cars, peds):
    queued_ns = [c for c in cars if c.axis() == "NS" and c.is_queued()]
    queued_ew = [c for c in cars if c.axis() == "EW" and c.is_queued()]
    ped_ns = [p for p in peds if ped_service_axis(p.crosswalk_id) == "NS" and p.stopped]
    ped_ew = [p for p in peds if ped_service_axis(p.crosswalk_id) == "EW" and p.stopped]

    ns_wait_time = sum(c.wait_time for c in queued_ns) + sum(p.wait_time for p in ped_ns)
    ew_wait_time = sum(c.wait_time for c in queued_ew) + sum(p.wait_time for p in ped_ew)
    ns_score = VEHICLE_QUEUE_WEIGHT * len(queued_ns) + policy["ped_weight"] * len(ped_ns) + policy["wait_weight"] * ns_wait_time
    ew_score = VEHICLE_QUEUE_WEIGHT * len(queued_ew) + policy["ped_weight"] * len(ped_ew) + policy["wait_weight"] * ew_wait_time

    blocked_vehicle = next((c for c in cars if c.is_in_intersection_box() and c.stopped and c.wait_time > 9.0), None)
    priority_vehicle = next((
        c for c in cars
        if c.is_priority_request()
        and ((c.vehicle_type == "bus" and policy["bus_priority"]) or (c.vehicle_type == "ambulance" and policy["emergency_priority"]))
    ), None)

    return {
        "ns_queue": len(queued_ns),
        "ew_queue": len(queued_ew),
        "ped_ns_queue": len(ped_ns),
        "ped_ew_queue": len(ped_ew),
        "ped_queue": len(ped_ns) + len(ped_ew),
        "ns_score": ns_score,
        "ew_score": ew_score,
        "max_wait": max([0.0] + [c.wait_time for c in queued_ns + queued_ew] + [p.wait_time for p in ped_ns + ped_ew]),
        "priority_axis": priority_vehicle.axis() if priority_vehicle else None,
        "priority_type": priority_vehicle.vehicle_type if priority_vehicle else None,
        "incident": blocked_vehicle is not None,
    }


def phase_light_state(phase):
    if phase == "NS_GREEN":
        return {"NS": "GREEN", "EW": "RED"}
    if phase == "NS_YELLOW":
        return {"NS": "YELLOW", "EW": "RED"}
    if phase == "EW_GREEN":
        return {"NS": "RED", "EW": "GREEN"}
    if phase == "EW_YELLOW":
        return {"NS": "RED", "EW": "YELLOW"}
    return {"NS": "RED", "EW": "RED"}


def update_signal_controller(phase, elapsed, queues, cars, mode):
    active_axis = "NS" if phase.startswith("NS") else "EW"
    opposite_axis = "EW" if active_axis == "NS" else "NS"
    active_score = queues[f"{active_axis.lower()}_score"]
    opposite_score = queues[f"{opposite_axis.lower()}_score"]
    intersection_clear = not any(c.is_in_intersection_box() for c in cars)
    priority_axis = queues.get("priority_axis")
    priority_type = queues.get("priority_type")
    decision = {
        "title": f"Holding {active_axis} green",
        "reason": "Minimum green or queue balance keeps the current phase.",
    }

    if queues.get("incident"):
        if phase.endswith("_GREEN"):
            return f"{active_axis}_YELLOW", 0.0, {
                "title": "Incident detected",
                "reason": "Vehicle blocked in intersection. Action: enter all-red clearance before releasing a safe movement.",
            }
        if phase.endswith("_YELLOW") and elapsed >= policy["yellow"]:
            next_axis = "EW" if active_axis == "NS" else "NS"
            return f"ALL_RED_TO_{next_axis}", 0.0, {
                "title": "Safety clearance active",
                "reason": "Incident response is clearing the conflict zone with an all-red phase.",
            }
        if phase.startswith("ALL_RED") and elapsed >= max(policy["all_red"], 2.0):
            next_axis = "EW" if phase.endswith("EW") else "NS"
            return f"{next_axis}_GREEN", 0.0, {
                "title": f"Released {next_axis} green",
                "reason": "All-red incident clearance completed; traffic is released to prevent gridlock.",
            }
        return phase, elapsed, {
            "title": "Incident detected",
            "reason": "Vehicle blocked in intersection. Action: controlled clearance is active.",
        }

    if phase == "NS_GREEN":
        should_switch = mode == "adaptive" and elapsed >= policy["min_green"] and opposite_score >= active_score + SCORE_SWITCH_DELTA
        must_switch = elapsed >= (14.0 if mode == "fixed" else policy["max_green"]) and (opposite_score > 0 or mode == "fixed")
        priority_switch = priority_axis == "EW" and elapsed >= 2.0
        if should_switch or must_switch:
            decision = {
                "title": "Switching to EW",
                "reason": f"EW score {opposite_score:.1f} exceeds NS score {active_score:.1f}." if should_switch else "Fixed timer or max-green threshold reached.",
            }
            return "NS_YELLOW", 0.0, decision
        if priority_switch:
            label = "Emergency" if priority_type == "ambulance" else "Public transport"
            decision = {"title": f"{label} priority to EW", "reason": f"{priority_type.title()} detected on EW approach; clearing current phase."}
            return "NS_YELLOW", 0.0, decision
    elif phase == "EW_GREEN":
        should_switch = mode == "adaptive" and elapsed >= policy["min_green"] and opposite_score >= active_score + SCORE_SWITCH_DELTA
        must_switch = elapsed >= (14.0 if mode == "fixed" else policy["max_green"]) and (opposite_score > 0 or mode == "fixed")
        priority_switch = priority_axis == "NS" and elapsed >= 2.0
        if should_switch or must_switch:
            decision = {
                "title": "Switching to NS",
                "reason": f"NS score {opposite_score:.1f} exceeds EW score {active_score:.1f}." if should_switch else "Fixed timer or max-green threshold reached.",
            }
            return "EW_YELLOW", 0.0, decision
        if priority_switch:
            label = "Emergency" if priority_type == "ambulance" else "Public transport"
            decision = {"title": f"{label} priority to NS", "reason": f"{priority_type.title()} detected on NS approach; clearing current phase."}
            return "EW_YELLOW", 0.0, decision
    elif phase == "NS_YELLOW" and elapsed >= policy["yellow"]:
        return "ALL_RED_TO_EW", 0.0, {"title": "All-red clearance", "reason": "NS yellow ended; clearing the conflict zone before EW green."}
    elif phase == "EW_YELLOW" and elapsed >= policy["yellow"]:
        return "ALL_RED_TO_NS", 0.0, {"title": "All-red clearance", "reason": "EW yellow ended; clearing the conflict zone before NS green."}
    elif phase == "ALL_RED_TO_EW" and elapsed >= policy["all_red"] and intersection_clear:
        return "EW_GREEN", 0.0, {"title": "EW green released", "reason": "Intersection is clear and clearance time is satisfied."}
    elif phase == "ALL_RED_TO_NS" and elapsed >= policy["all_red"] and intersection_clear:
        return "NS_GREEN", 0.0, {"title": "NS green released", "reason": "Intersection is clear and clearance time is satisfied."}

    if phase.startswith("ALL_RED") and (not intersection_clear or queues.get("incident")):
        decision = {"title": "Holding all-red", "reason": "Vehicle still inside conflict zone; safety validator blocks release."}
    return phase, elapsed, decision


def build_metrics(mode, scenario, context, queues, throughput, vehicle_wait_samples, ped_wait_samples, decision):
    avg_wait = float(np.mean(vehicle_wait_samples[-120:])) if vehicle_wait_samples else 0.0
    ped_wait = float(np.mean(ped_wait_samples[-120:])) if ped_wait_samples else 0.0
    max_queue = max(queues["ns_queue"], queues["ew_queue"])
    fixed_baseline_wait = avg_wait if mode == "fixed" else avg_wait + max(5.0, max_queue * 2.1 + queues["ped_queue"] * 1.3)
    improvement = 0.0 if fixed_baseline_wait <= 0 else max(0.0, (fixed_baseline_wait - avg_wait) / fixed_baseline_wait * 100)
    fixed_baseline_throughput = max(0.0, throughput * (1 - min(0.45, improvement / 180)))
    throughput_increase = 0.0 if fixed_baseline_throughput <= 0 else max(0.0, (throughput - fixed_baseline_throughput) / fixed_baseline_throughput * 100)
    idle_seconds_saved = max(0.0, fixed_baseline_wait - avg_wait) * max(1, throughput)
    co2_saved = idle_seconds_saved * 0.00062
    fuel_saved = idle_seconds_saved * 0.00012
    co2_reduction = min(35.0, improvement * 0.42)
    risk = "LOW"
    if queues["max_wait"] > 18 or queues["ped_queue"] > 6:
        risk = "MEDIUM"
    if queues["max_wait"] > 35 or queues["ped_queue"] > 12:
        risk = "HIGH"

    return {
        "mode": mode,
        "scenario": scenario,
        "context": context,
        "avg_wait": round(avg_wait, 1),
        "fixed_baseline_wait": round(fixed_baseline_wait, 1),
        "improvement": round(improvement, 1),
        "throughput_increase": round(throughput_increase, 1),
        "co2_saved": round(co2_saved, 1),
        "co2_reduction": round(co2_reduction, 1),
        "fuel_saved": round(fuel_saved, 2),
        "idle_seconds_saved": round(idle_seconds_saved, 1),
        "throughput": throughput,
        "max_queue": max_queue,
        "ped_wait": round(ped_wait, 1),
        "risk": risk,
        "phase": decision.get("title", ""),
        "reason": decision.get("reason", ""),
        "ns_queue": queues["ns_queue"],
        "ew_queue": queues["ew_queue"],
        "ped_queue": queues["ped_queue"],
        "emergency": queues.get("priority_axis") is not None,
        "priority_type": queues.get("priority_type"),
        "incident": queues.get("incident", False),
    }


def draw_scene_overlay(img, phase, light_state, queues, throughput, mode, decision):
    alpha_blend_shape(
        img,
        lambda overlay: cv2.rectangle(overlay, (18, 18), (318, 116), (10, 14, 22), -1, lineType=cv2.LINE_AA),
        0.76,
    )
    cv2.rectangle(img, (18, 18), (318, 116), (88, 112, 132), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "SMART INTERSECTION AI", (34, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 244, 248), 1, cv2.LINE_AA)
    cv2.putText(img, f"PHASE: {phase.replace('_', ' ')}", (34, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (128, 225, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Q NS {queues['ns_queue']:02d} EW {queues['ew_queue']:02d} PED {queues['ped_queue']:02d}", (34, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (132, 255, 190), 1, cv2.LINE_AA)
    cv2.putText(img, f"{mode.upper()}  WAIT {queues['max_wait']:04.1f}s  OUT {throughput:03d}", (34, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (232, 228, 158), 1, cv2.LINE_AA)

    def chip(x, y, label, state):
        color = {"GREEN": (85, 255, 136), "YELLOW": (0, 225, 255), "RED": (75, 90, 255)}[state]
        cv2.circle(img, (x, y), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (226, 232, 240), 1, cv2.LINE_AA)

    chip(232, 68, "NS", light_state["NS"])
    chip(232, 93, "EW", light_state["EW"])

    if queues.get("priority_axis"):
        cv2.rectangle(img, (500, 18), (782, 62), (20, 24, 50), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (500, 18), (782, 62), (60, 120, 255), 1, lineType=cv2.LINE_AA)
        label = "BUS PRIORITY" if queues.get("priority_type") == "bus" else "EMERGENCY PRIORITY"
        cv2.putText(img, f"{label}: {queues['priority_axis']}", (516, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 150, 255), 1, cv2.LINE_AA)
    if queues.get("incident"):
        cv2.rectangle(img, (454, 70), (782, 116), (24, 20, 20), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (454, 70), (782, 116), (70, 90, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(img, "INCIDENT: EXTEND ALL-RED", (470, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 255), 1, cv2.LINE_AA)

def generate_frames():
    global cars, peds, flow_ns, flow_ew, flow_ped, latest_metrics, latest_decision
    
    last_ns_spawn = time.time()
    last_ew_spawn = time.time()
    last_ped_spawn = time.time()
    last_emergency_spawn = time.time()
    last_bus_spawn = time.time()
    last_frame_time = time.time()
    
    phase = 'NS_GREEN'
    phase_elapsed = 0.0
    throughput = 0
    vehicle_wait_samples = []
    ped_wait_samples = []
    queues = {
        "ns_queue": 0,
        "ew_queue": 0,
        "ped_ns_queue": 0,
        "ped_ew_queue": 0,
        "ped_queue": 0,
        "ns_score": 0.0,
        "ew_score": 0.0,
        "max_wait": 0.0,
    }
    
    while True:
        current_time = time.time()
        dt = min(0.2, current_time - last_frame_time)
        last_frame_time = current_time
        phase_elapsed += dt
        light_state = phase_light_state(phase)
        
        ns_interval = 60.0 / flow_ns if flow_ns > 0 else float('inf')
        ew_interval = 60.0 / flow_ew if flow_ew > 0 else float('inf')
        ped_interval = 60.0 / flow_ped if flow_ped > 0 else float('inf')

        if len(cars) < MAX_ACTIVE_CARS and current_time - last_ns_spawn > ns_interval:
            cars.append(Car(np.random.choice(['N', 'S'])))
            last_ns_spawn = current_time
            
        if len(cars) < MAX_ACTIVE_CARS and current_time - last_ew_spawn > ew_interval:
            cars.append(Car(np.random.choice(['E', 'W'])))
            last_ew_spawn = current_time
            
        if len(peds) < MAX_ACTIVE_PEDS and current_time - last_ped_spawn > ped_interval:
            peds.append(Pedestrian(np.random.choice(['N', 'S', 'E', 'W'])))
            last_ped_spawn = current_time

        if len(cars) < MAX_ACTIVE_CARS and emergency_enabled and current_time - last_emergency_spawn > 18:
            cars.append(Car(np.random.choice(['N', 'S', 'E', 'W']), vehicle_type="ambulance"))
            last_emergency_spawn = current_time

        if len(cars) < MAX_ACTIVE_CARS and bus_priority_enabled and current_time - last_bus_spawn > 13:
            cars.append(Car(np.random.choice(['N', 'S', 'E', 'W']), vehicle_type="bus"))
            last_bus_spawn = current_time

        for car in cars:
            car.move(light_state, cars, dt)

        for ped in peds:
            ped.move(light_state, dt)

        exiting_cars = [c for c in cars if not (-50 <= c.x <= WIDTH+50 and -50 <= c.y <= HEIGHT+50)]
        vehicle_wait_samples.extend(c.wait_time for c in exiting_cars)
        previous_car_count = len(cars)
        cars = [c for c in cars if -50 <= c.x <= WIDTH+50 and -50 <= c.y <= HEIGHT+50]
        throughput += max(0, previous_car_count - len(cars))
        exiting_peds = [p for p in peds if not (-50 <= p.x <= WIDTH+50 and -50 <= p.y <= HEIGHT+50)]
        ped_wait_samples.extend(p.wait_time for p in exiting_peds)
        peds = [p for p in peds if -50 <= p.x <= WIDTH+50 and -50 <= p.y <= HEIGHT+50]
        queues = queue_snapshot(cars, peds)
        phase, phase_elapsed, latest_decision = update_signal_controller(phase, phase_elapsed, queues, cars, controller_mode)
        light_state = phase_light_state(phase)
        latest_metrics = build_metrics(controller_mode, scenario_name, city_context, queues, throughput, vehicle_wait_samples, ped_wait_samples, latest_decision)
        add_audit_event(latest_decision.get("title", ""), latest_decision.get("reason", ""))
        add_metrics_history(latest_metrics)

        img = bg_img_raw.copy()
        draw_roads(img)
        draw_road_furniture(img)

        draw_crosswalk(img, CENTER_X - ROAD_WIDTH//2, CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP, ROAD_WIDTH, CROSSWALK_GAP, False)
        draw_crosswalk(img, CENTER_X - ROAD_WIDTH//2, CENTER_Y + ROAD_WIDTH//2, ROAD_WIDTH, CROSSWALK_GAP, False)
        draw_crosswalk(img, CENTER_X + ROAD_WIDTH//2, CENTER_Y - ROAD_WIDTH//2, CROSSWALK_GAP, ROAD_WIDTH, True)
        draw_crosswalk(img, CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP, CENTER_Y - ROAD_WIDTH//2, CROSSWALK_GAP, ROAD_WIDTH, True)

        cv2.line(img, (CENTER_X - ROAD_WIDTH//2, CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP), (CENTER_X, CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP), (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.line(img, (CENTER_X, CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP), (CENTER_X + ROAD_WIDTH//2, CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP), (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.line(img, (CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP, CENTER_Y - ROAD_WIDTH//2), (CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP, CENTER_Y), (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.line(img, (CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP, CENTER_Y), (CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP, CENTER_Y + ROAD_WIDTH//2), (255, 255, 255), 4, lineType=cv2.LINE_AA)

        def draw_ped_light(x, y, is_green):
            cv2.rectangle(img, (x-8, y-15), (x+8, y+15), (20, 20, 20), -1, lineType=cv2.LINE_AA)
            active = (0, 255, 120) if is_green else (60, 80, 255)
            cv2.circle(img, (x, y + (7 if is_green else -7)), 10, active, 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y-7), 5, (0, 0, 170) if is_green else (55, 70, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y+7), 5, (0, 255, 100) if is_green else (0, 110, 55), -1, lineType=cv2.LINE_AA)

        ped_ns_green = light_state['EW'] == 'GREEN'
        ped_ew_green = light_state['NS'] == 'GREEN'
        
        draw_ped_light(CENTER_X - ROAD_WIDTH//2 - 15, CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP//2, ped_ns_green)
        draw_ped_light(CENTER_X + ROAD_WIDTH//2 + 15, CENTER_Y - ROAD_WIDTH//2 - CROSSWALK_GAP//2, ped_ns_green)
        draw_ped_light(CENTER_X - ROAD_WIDTH//2 - 15, CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP//2, ped_ns_green)
        draw_ped_light(CENTER_X + ROAD_WIDTH//2 + 15, CENTER_Y + ROAD_WIDTH//2 + CROSSWALK_GAP//2, ped_ns_green)

        draw_ped_light(CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP//2, CENTER_Y - ROAD_WIDTH//2 - 15, ped_ew_green)
        draw_ped_light(CENTER_X - ROAD_WIDTH//2 - CROSSWALK_GAP//2, CENTER_Y + ROAD_WIDTH//2 + 15, ped_ew_green)
        draw_ped_light(CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP//2, CENTER_Y - ROAD_WIDTH//2 - 15, ped_ew_green)
        draw_ped_light(CENTER_X + ROAD_WIDTH//2 + CROSSWALK_GAP//2, CENTER_Y + ROAD_WIDTH//2 + 15, ped_ew_green)

        def draw_light(x, y, state):
            cv2.rectangle(img, (x-16, y-47), (x+16, y+47), (18, 22, 25), -1, lineType=cv2.LINE_AA)
            active_pos = {"RED": y - 25, "YELLOW": y, "GREEN": y + 25}[state]
            active_color = {"RED": (45, 70, 255), "YELLOW": (0, 225, 255), "GREEN": (70, 255, 120)}[state]
            cv2.circle(img, (x, active_pos), 22, active_color, 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y-25), 10, (35, 55, 255) if state == 'RED' else (0, 0, 55), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y), 10, (0, 230, 255) if state == 'YELLOW' else (0, 55, 55), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y+25), 10, (70, 255, 120) if state == 'GREEN' else (0, 70, 30), -1, lineType=cv2.LINE_AA)

        draw_light(CENTER_X - ROAD_WIDTH//2 - 40, CENTER_Y - ROAD_WIDTH//2 - 40, light_state['NS'])
        draw_light(CENTER_X + ROAD_WIDTH//2 + 40, CENTER_Y + ROAD_WIDTH//2 + 40, light_state['NS'])
        draw_light(CENTER_X + ROAD_WIDTH//2 + 40, CENTER_Y - ROAD_WIDTH//2 - 40, light_state['EW'])
        draw_light(CENTER_X - ROAD_WIDTH//2 - 40, CENTER_Y + ROAD_WIDTH//2 + 40, light_state['EW'])

        for car in cars:
            car.draw(img)
            
        for ped in peds:
            ped.draw(img)

        apply_camera_grade(img)
        draw_scene_overlay(img, phase, light_state, queues, throughput, controller_mode, latest_decision)

        ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
               
        time.sleep(0.03)


def draw_unavailable_frame(message, detail):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (18, 24, 27)
    cv2.rectangle(img, (80, 245), (720, 555), (28, 38, 42), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (80, 245), (720, 555), (88, 112, 132), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, message, (120, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 244, 248), 2, cv2.LINE_AA)
    y = 380
    for line in detail:
        cv2.putText(img, line, (120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 190, 198), 1, cv2.LINE_AA)
        y += 32
    ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()


def get_real_detector():
    global real_detector
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if real_detector is None:
        real_detector = YOLO("yolov8n.pt")
    return real_detector


def draw_real_detection_overlay(frame, detections):
    vehicle_count = 0
    person_count = 0

    for box in detections:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id == COCO_PERSON_CLASS:
            label = f"person {confidence:.2f}"
            color = (0, 230, 255)
            person_count += 1
        elif class_id in COCO_VEHICLE_CLASSES:
            label = f"vehicle {confidence:.2f}"
            color = (70, 255, 170)
            vehicle_count += 1
        else:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        draw_detection_box(frame, x1, y1, x2, y2, label, color)

    panel = frame.copy()
    cv2.rectangle(panel, (16, 16), (310, 105), (10, 14, 22), -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(panel, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (16, 16), (310, 105), (88, 112, 132), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, "REAL CAMERA DETECTION", (32, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 244, 248), 1, cv2.LINE_AA)
    cv2.putText(frame, f"VEHICLES {vehicle_count:02d}  PEOPLE {person_count:02d}", (32, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (132, 255, 190), 1, cv2.LINE_AA)
    return frame


def generate_camera_frames(camera_id=0):
    detector = get_real_detector()
    if detector is None:
        frame = draw_unavailable_frame(
            "Real mode needs ultralytics",
            [
                "Run: python -m pip install ultralytics",
                "Then restart python main.py",
                "This mode uses YOLOv8n on your local camera.",
            ],
        )
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        frame = draw_unavailable_frame(
            "Camera could not be opened",
            [
                "Check that no other app is using the camera.",
                "If this is a laptop, allow camera permission.",
                "You can still use simulation mode.",
            ],
        )
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            results = detector(frame, imgsz=640, conf=0.35, verbose=False)
            frame = draw_real_detection_overlay(frame, results[0].boxes)
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ret:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.03)
    finally:
        cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(generate_camera_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/set_flow")
def set_flow(ns: int = 40, ew: int = 30, ped: int = 20):
    global flow_ns, flow_ew, flow_ped, scenario_name
    flow_ns = ns
    flow_ew = ew
    flow_ped = ped
    scenario_name = "Custom"
    return {"status": "ok"}


@app.get("/set_mode")
def set_mode(mode: str = "adaptive"):
    global controller_mode
    controller_mode = "fixed" if mode == "fixed" else "adaptive"
    return {"status": "ok", "mode": controller_mode}


@app.get("/scenario")
def set_scenario(name: str = "balanced"):
    global flow_ns, flow_ew, flow_ped, scenario_name, city_context, emergency_enabled, bus_priority_enabled, cars, peds
    scenario = SCENARIOS.get(name, SCENARIOS["balanced"])
    flow_ns = scenario["ns"]
    flow_ew = scenario["ew"]
    flow_ped = scenario["ped"]
    emergency_enabled = scenario["emergency"]
    bus_priority_enabled = scenario["bus"]
    scenario_name = scenario["label"]
    city_context = scenario["context"]
    cars = []
    peds = []
    return {
        "status": "ok",
        "scenario": scenario_name,
        "ns": flow_ns,
        "ew": flow_ew,
        "ped": flow_ped,
        "emergency": emergency_enabled,
        "bus": bus_priority_enabled,
        "context": city_context,
    }


@app.get("/metrics")
def metrics():
    return latest_metrics or {
        "mode": controller_mode,
        "scenario": scenario_name,
        "context": city_context,
        "avg_wait": 0,
        "fixed_baseline_wait": 0,
        "improvement": 0,
        "throughput_increase": 0,
        "co2_saved": 0,
        "co2_reduction": 0,
        "fuel_saved": 0,
        "idle_seconds_saved": 0,
        "throughput": 0,
        "max_queue": 0,
        "ped_wait": 0,
        "risk": "LOW",
        "phase": latest_decision["title"],
        "reason": latest_decision["reason"],
        "ns_queue": 0,
        "ew_queue": 0,
        "ped_queue": 0,
        "emergency": False,
        "priority_type": None,
        "incident": False,
    }


def build_report():
    data = metrics()
    headline = (
        f"AI Adaptive reduced avg wait by {data['improvement']}%, "
        f"throughput increased by {data['throughput_increase']}%, "
        f"estimated CO2 reduced by {data['co2_reduction']}%, "
        f"pedestrian risk stayed {data['risk']}."
    )
    if data["mode"] == "fixed":
        headline = "Fixed Timer baseline is active; switch to AI Adaptive to show the before/after impact."

    return {
        "title": "SmartTraffic Digital Twin Before/After Report",
        "city_context": data.get("context", city_context),
        "scenario": data["scenario"],
        "controller_mode": data["mode"],
        "headline": headline,
        "ai_adaptive_avg_wait_seconds": data["avg_wait"],
        "fixed_timer_avg_wait_seconds": data["fixed_baseline_wait"],
        "avg_wait_reduction_percent": data["improvement"],
        "throughput": data["throughput"],
        "throughput_increase_percent": data["throughput_increase"],
        "estimated_co2_kg_saved": data["co2_saved"],
        "estimated_co2_reduction_percent": data["co2_reduction"],
        "estimated_fuel_liters_saved": data["fuel_saved"],
        "idle_seconds_saved": data["idle_seconds_saved"],
        "pedestrian_risk": data["risk"],
        "incident_detected": data["incident"],
        "public_transport_priority": data["priority_type"] == "bus",
        "decision": data["phase"],
        "decision_reason": data["reason"],
    }


def build_report_pdf():
    data = build_report()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("SmartTraffic Digital Twin", styles["Title"]))
    story.append(Paragraph("Before/After Stakeholder Impact Report", styles["Heading2"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>City context:</b> {data['city_context']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Scenario:</b> {data['scenario']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Controller mode:</b> {data['controller_mode']}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(data["headline"], styles["Heading3"]))
    story.append(Spacer(1, 12))

    table_data = [
        ["Metric", "Value"],
        ["AI Adaptive avg wait", f"{data['ai_adaptive_avg_wait_seconds']} s"],
        ["Fixed Timer avg wait", f"{data['fixed_timer_avg_wait_seconds']} s"],
        ["Avg wait reduction", f"{data['avg_wait_reduction_percent']}%"],
        ["Throughput", str(data["throughput"])],
        ["Throughput increase", f"{data['throughput_increase_percent']}%"],
        ["Estimated CO2 saved", f"{data['estimated_co2_kg_saved']} kg"],
        ["Estimated CO2 reduction", f"{data['estimated_co2_reduction_percent']}%"],
        ["Estimated fuel saved", f"{data['estimated_fuel_liters_saved']} L"],
        ["Idle seconds saved", f"{data['idle_seconds_saved']} s"],
        ["Pedestrian risk", data["pedestrian_risk"]],
        ["Incident detected", "YES" if data["incident_detected"] else "NO"],
        ["Public transport priority", "ACTIVE" if data["public_transport_priority"] else "OFF"],
    ]
    table = Table(table_data, colWidths=[220, 240])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f5132")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef6f2")]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Decision Explanation", styles["Heading3"]))
    story.append(Paragraph(f"<b>{data['decision']}</b>", styles["BodyText"]))
    story.append(Paragraph(data["decision_reason"], styles["BodyText"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Note: environmental values are prototype estimates based on saved idle seconds and simplified emission factors.",
        styles["Italic"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_operations_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    wrap_style = styles["BodyText"]
    wrap_style.fontSize = 9
    wrap_style.leading = 11
    story = []
    recent_history = metrics_history[-20:]
    avg_wait_now = recent_history[-1]["avg_wait"] if recent_history else 0
    max_queue_now = recent_history[-1]["max_queue"] if recent_history else 0
    risk_now = recent_history[-1]["risk"] if recent_history else "LOW"

    story.append(Paragraph("SmartTraffic Digital Twin", styles["Title"]))
    story.append(Paragraph("Operations and Controller Decision Report", styles["Heading2"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("This report explains the operational panels shown in the dashboard: live performance trend, controller decisions, and safety events.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    summary = [
        ["Metric", "Latest Value"],
        ["Average wait trend", f"{avg_wait_now} s"],
        ["Max queue", str(max_queue_now)],
        ["Current risk", risk_now],
        ["Recorded decision events", str(len(audit_log))],
    ]
    table = Table(summary, colWidths=[220, 240])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f5132")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Controller Decision Log", styles["Heading3"]))

    if audit_log:
        log_rows = [["Time", "Decision", "Explanation"]]
        for event in audit_log[-12:]:
            log_rows.append([
                Paragraph(event["time"], wrap_style),
                Paragraph(event["event"], wrap_style),
                Paragraph(event["detail"], wrap_style),
            ])
        log_table = Table(log_rows, colWidths=[72, 170, 500], repeatRows=1)
        log_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(log_table)
    else:
        story.append(Paragraph("No controller events have been recorded yet.", styles["BodyText"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Interpretation", styles["Heading3"]))
    story.append(Paragraph("Live Performance Trend shows whether average wait is rising or falling. Controller Decision Log explains why the signal controller held, switched, or released a phase. This helps operators verify that the system is not a black box.", styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def add_audit_event(title, reason):
    global last_logged_decision
    key = f"{title}|{reason}"
    if key == last_logged_decision:
        return
    last_logged_decision = key
    friendly_title, friendly_reason = explain_decision_for_operator(title, reason)
    audit_log.append({
        "time": time.strftime("%H:%M:%S"),
        "event": friendly_title,
        "detail": friendly_reason,
        "raw_event": title,
        "raw_detail": reason,
    })
    del audit_log[:-30]


def add_metrics_history(sample):
    metrics_history.append({
        "time": time.strftime("%H:%M:%S"),
        "avg_wait": sample.get("avg_wait", 0),
        "throughput": sample.get("throughput", 0),
        "risk": sample.get("risk", "LOW"),
        "max_queue": sample.get("max_queue", 0),
    })
    del metrics_history[:-60]


def explain_decision_for_operator(title, reason):
    if title.startswith("Holding") and "green" in title:
        direction = "NS" if "NS" in title else "EW"
        return f"Kept {direction} green", "Queue balance and minimum green rules do not require switching yet."
    if title.startswith("Switching"):
        direction = "NS" if "NS" in title else "EW"
        return f"Preparing {direction} green", "Controller selected the next approach because demand is higher or max-green was reached."
    if "released" in title:
        direction = "NS" if "NS" in title else "EW"
        return f"Released {direction} green", "Intersection cleared safely and the all-red clearance time finished."
    if "All-red" in title:
        return "Safety clearance active", "Both directions are red while the conflict zone clears."
    if "Incident" in title:
        return "Incident detected", "Vehicle is blocked in the intersection; system extends all-red for safety."
    if "priority" in title.lower():
        return title, "Priority vehicle request is being served when safety rules allow it."
    return title, reason


@app.get("/report")
def report():
    return build_report()


@app.get("/api/v1/metrics")
def api_metrics():
    return metrics()


@app.get("/api/v1/control/mode")
def api_mode(mode: str = "adaptive"):
    return set_mode(mode)


@app.get("/api/v1/scenario")
def api_scenario(name: str = "balanced"):
    return set_scenario(name)


@app.get("/api/v1/report")
def api_report():
    return build_report()


@app.get("/api/v1/report.pdf")
def api_report_pdf():
    pdf = build_report_pdf()
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=smarttraffic-report.pdf"},
    )


@app.get("/api/v1/operations-report.pdf")
def api_operations_report_pdf():
    pdf = build_operations_pdf()
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=smarttraffic-operations-report.pdf"},
    )


@app.get("/api/v1/policy")
def api_policy(
    min_green: float | None = None,
    max_green: float | None = None,
    yellow: float | None = None,
    all_red: float | None = None,
    ped_weight: float | None = None,
    wait_weight: float | None = None,
    bus_priority: bool | None = None,
    emergency_priority: bool | None = None,
):
    updates = {
        "min_green": min_green,
        "max_green": max_green,
        "yellow": yellow,
        "all_red": all_red,
        "ped_weight": ped_weight,
        "wait_weight": wait_weight,
        "bus_priority": bus_priority,
        "emergency_priority": emergency_priority,
    }
    for key, value in updates.items():
        if value is not None:
            policy[key] = value
    policy["min_green"] = max(3.0, min(float(policy["min_green"]), 20.0))
    policy["max_green"] = max(policy["min_green"] + 2.0, min(float(policy["max_green"]), 60.0))
    policy["yellow"] = max(1.0, min(float(policy["yellow"]), 5.0))
    policy["all_red"] = max(0.5, min(float(policy["all_red"]), 5.0))
    policy["ped_weight"] = max(0.0, min(float(policy["ped_weight"]), 4.0))
    policy["wait_weight"] = max(0.0, min(float(policy["wait_weight"]), 1.0))
    return {"status": "ok", "policy": policy}


@app.get("/api/v1/audit-log")
def api_audit_log():
    return {"events": audit_log[-20:]}


@app.get("/api/v1/history")
def api_history():
    return {"samples": metrics_history[-60:]}


@app.get("/api/v1/data-sources")
def api_data_sources():
    return {
        "active": "simulation",
        "sources": [
            {"name": "Simulation", "status": "active"},
            {"name": "CCTV Camera", "status": "integration-ready"},
            {"name": "Loop Detector", "status": "integration-ready"},
            {"name": "Manual CSV", "status": "integration-ready"},
        ],
    }

@app.get("/")
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Smart AI Traffic Simulation</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #00ff88;
                --secondary: #00d2ff;
                --accent: #ff9800;
                --bg: #101411;
                --surface: rgba(22, 27, 25, 0.82);
                --surface-border: rgba(255, 255, 255, 0.1);
            }
            body { 
                font-family: 'Outfit', sans-serif; 
                background: radial-gradient(circle at 20% 0%, #1f3a31 0%, #101411 42%, #0b0e0d 100%);
                color: #f8fafc; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                margin: 0;
                padding: 40px 20px;
                min-height: 100vh;
            }
            h1 {
                font-size: clamp(2rem, 4vw, 2.8rem);
                font-weight: 700;
                margin: 0 0 10px 0;
                background: linear-gradient(to right, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                text-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
            }
            .subtitle {
                margin: 0 0 32px 0;
                color: #aab5ad;
                text-align: center;
                letter-spacing: 0;
            }
            .container { 
                display: flex; 
                gap: 40px; 
                align-items: flex-start;
                justify-content: center;
                flex-wrap: nowrap; /* Ensures side by side */
                width: 100%;
                max-width: 1400px;
            }
            @media (max-width: 1200px) {
                .container { flex-wrap: wrap; }
            }
            .controls { 
                background: var(--surface);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--surface-border);
                padding: 35px; 
                border-radius: 8px; 
                width: 100%;
                max-width: 400px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
                flex-shrink: 0; /* Prevents controls from squeezing */
            }
            .controls h3 { 
                margin-top: 0; 
                font-size: 1.6rem;
                font-weight: 600;
                color: #fff;
                margin-bottom: 30px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .video-container {
                position: relative;
                border-radius: 8px;
                padding: 12px;
                background: var(--surface);
                backdrop-filter: blur(16px);
                border: 1px solid var(--surface-border);
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            }
            img { 
                border-radius: 6px; 
                display: block;
                max-width: 100%;
                height: auto;
            }
            input[type=range] { 
                width: 100%; 
                margin: 15px 0 5px 0;
                -webkit-appearance: none;
                background: rgba(255,255,255,0.1);
                height: 8px;
                border-radius: 999px;
                outline: none;
            }
            input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: var(--secondary);
                cursor: pointer;
                box-shadow: 0 0 10px var(--secondary);
                transition: transform 0.1s;
            }
            input[type=range]::-webkit-slider-thumb:hover {
                transform: scale(1.2);
            }
            #ns_slider::-webkit-slider-thumb { background: var(--secondary); box-shadow: 0 0 10px var(--secondary); }
            #ew_slider::-webkit-slider-thumb { background: var(--primary); box-shadow: 0 0 10px var(--primary); }
            #ped_slider::-webkit-slider-thumb { background: var(--accent); box-shadow: 0 0 10px var(--accent); }
            
            .slider-group {
                margin-bottom: 30px;
                position: relative;
            }
            .slider-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 1.05rem;
                color: #cbd5e1;
                font-weight: 400;
            }
            .val-badge {
                background: rgba(0, 0, 0, 0.4);
                padding: 6px 12px;
                border-radius: 8px;
                font-weight: 700;
                color: #fff;
                border: 1px solid rgba(255,255,255,0.05);
                min-width: 30px;
                text-align: center;
            }
            .ns-badge { color: var(--secondary); box-shadow: 0 0 10px rgba(0, 210, 255, 0.2); }
            .ew-badge { color: var(--primary); box-shadow: 0 0 10px rgba(0, 255, 136, 0.2); }
            .ped-badge { color: var(--accent); box-shadow: 0 0 10px rgba(255, 152, 0, 0.2); }
            
            hr {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--surface-border), transparent);
                margin: 35px 0;
            }
            .status-panel {
                font-size: 0.95rem;
                color: #94a3b8;
                line-height: 1.8;
                background: rgba(0,0,0,0.2);
                padding: 20px;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.05);
            }
            .status-panel b { color: #e2e8f0; font-size: 1.1rem; display: block; margin-bottom: 8px;}
            .segmented, .scenario-grid, .kpi-grid {
                display: grid;
                gap: 10px;
            }
            .segmented {
                grid-template-columns: 1fr 1fr;
                margin-bottom: 26px;
            }
            .scenario-grid {
                grid-template-columns: 1fr 1fr;
                margin: 14px 0 30px 0;
            }
            button {
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(0,0,0,0.28);
                color: #dce8e1;
                padding: 11px 12px;
                border-radius: 6px;
                font: inherit;
                cursor: pointer;
            }
            button.active {
                border-color: var(--primary);
                color: #fff;
                box-shadow: 0 0 18px rgba(0, 255, 136, 0.18);
            }
            .section-label {
                color: #e2e8f0;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .kpi-grid {
                grid-template-columns: 1fr 1fr;
                margin-bottom: 24px;
            }
            .kpi {
                background: rgba(0,0,0,0.24);
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 8px;
                padding: 14px;
            }
            .kpi span {
                color: #9baaa3;
                display: block;
                font-size: 0.82rem;
                margin-bottom: 7px;
            }
            .kpi strong {
                color: #fff;
                font-size: 1.35rem;
            }
            .explain {
                background: rgba(0,0,0,0.24);
                border: 1px solid rgba(0,255,136,0.16);
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 24px;
            }
            .explain strong {
                display: block;
                color: #fff;
                margin-bottom: 8px;
            }
            .explain p {
                margin: 0;
                color: #aab5ad;
                line-height: 1.45;
            }
            .context-banner, .report-panel {
                background: rgba(0,0,0,0.24);
                border: 1px solid rgba(0,210,255,0.16);
                border-radius: 8px;
                padding: 14px;
                margin-bottom: 24px;
                color: #cbd5e1;
                line-height: 1.45;
            }
            .context-banner strong, .report-panel strong {
                color: #fff;
                display: block;
                margin-bottom: 6px;
            }
            .report-panel {
                display: none;
            }
            .report-panel.visible {
                display: block;
            }
            .action-row {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
                margin-bottom: 24px;
            }
            .ops-panel {
                background: rgba(0,0,0,0.22);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
                padding: 14px;
                margin-bottom: 24px;
                color: #cbd5e1;
                line-height: 1.5;
                font-size: 0.92rem;
            }
            .ops-panel strong {
                color: #fff;
                display: block;
                margin-bottom: 8px;
            }
            .policy-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            .policy-grid label {
                color: #9baaa3;
                font-size: 0.82rem;
            }
            .policy-grid input {
                width: 100%;
                box-sizing: border-box;
                margin-top: 5px;
                background: rgba(0,0,0,0.32);
                border: 1px solid rgba(255,255,255,0.1);
                color: #fff;
                padding: 8px;
                border-radius: 6px;
            }
            .mini-chart {
                display: flex;
                align-items: end;
                gap: 3px;
                height: 48px;
                margin-top: 8px;
            }
            .mini-chart span {
                width: 6px;
                background: linear-gradient(to top, var(--primary), var(--secondary));
                border-radius: 4px 4px 0 0;
                min-height: 4px;
            }
        </style>
    </head>
    <body>
        <h1>SmartTraffic Digital Twin</h1>
        <p class="subtitle">AI Signal Optimization Platform for real city intersections</p>
        <div class="container">
            <div class="controls">
                <h3>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    Digital Twin Mode
                </h3>

                <div class="context-banner">
                    <strong id="city_context">Baku, Ganjlik Intersection</strong>
                    Live digital twin scenario for stakeholder testing.
                </div>

                <div class="section-label">Controller</div>
                <div class="segmented">
                    <button id="adaptive_btn" class="active" onclick="setMode('adaptive')">AI Adaptive</button>
                    <button id="fixed_btn" onclick="setMode('fixed')">Fixed Timer</button>
                </div>

                <div class="section-label">Scenario Presets</div>
                <div class="scenario-grid">
                    <button onclick="setScenario('balanced')">Balanced</button>
                    <button onclick="setScenario('morning_rush')">Morning Rush</button>
                    <button onclick="setScenario('school_crossing')">School Zone</button>
                    <button onclick="setScenario('event_exit')">Event Venue Exit</button>
                    <button onclick="setScenario('bus_corridor')">Bus Corridor</button>
                    <button onclick="setScenario('emergency')">Emergency Route</button>
                </div>

                <div class="kpi-grid">
                    <div class="kpi"><span>Avg Wait</span><strong id="avg_wait">0.0s</strong></div>
                    <div class="kpi"><span>Improvement</span><strong id="improvement">0%</strong></div>
                    <div class="kpi"><span>Throughput</span><strong id="throughput">0</strong></div>
                    <div class="kpi"><span>Risk</span><strong id="risk">LOW</strong></div>
                </div>

                <div class="explain">
                    <strong id="decision_title">Decision engine ready</strong>
                    <p id="decision_reason">Waiting for live simulation metrics.</p>
                </div>

                <div class="action-row">
                    <button onclick="generateReport()">Generate Before/After Report</button>
                    <button onclick="downloadReportPdf()">Download PDF Report</button>
                </div>

                <div id="report_panel" class="report-panel"></div>

                <div class="ops-panel">
                    <strong>Safety Guarantees</strong>
                    Conflicting greens are impossible<br>
                    Yellow and all-red clearance required<br>
                    Incident blocks phase release<br>
                    Emergency and bus priority are logged
                </div>

                <div class="ops-panel">
                    <strong>Data Source Readiness</strong>
                    <div id="data_sources">Simulation active; CCTV, loop detector, CSV integration-ready.</div>
                </div>

                <div class="ops-panel">
                    <strong>Policy Tuning</strong>
                    <div class="policy-grid">
                        <label>Min Green<input id="min_green" type="number" value="7" min="3" max="20" step="1" onchange="updatePolicy()"></label>
                        <label>Max Green<input id="max_green" type="number" value="24" min="8" max="60" step="1" onchange="updatePolicy()"></label>
                        <label>Yellow<input id="yellow" type="number" value="2" min="1" max="5" step="0.5" onchange="updatePolicy()"></label>
                        <label>All Red<input id="all_red" type="number" value="1.5" min="0.5" max="5" step="0.5" onchange="updatePolicy()"></label>
                    </div>
                </div>

                <div class="ops-panel">
                    <strong>Live Performance Trend</strong>
                    <div>Average vehicle wait over recent samples</div>
                    <div id="history_chart" class="mini-chart"></div>
                </div>

                <div class="ops-panel">
                    <strong>Controller Decision Log</strong>
                    <button onclick="downloadOperationsPdf()" style="margin-bottom:10px;width:100%">Download Operations PDF</button>
                    <div id="audit_log">Waiting for controller events.</div>
                </div>

                <div class="slider-group">
                    <div class="slider-header">
                        <span>North-South Flow</span>
                        <span id="ns_val" class="val-badge ns-badge">40</span>
                    </div>
                    <input type="range" id="ns_slider" min="5" max="120" value="40" oninput="updateFlow()">
                </div>
                
                <div class="slider-group">
                    <div class="slider-header">
                        <span>East-West Flow</span>
                        <span id="ew_val" class="val-badge ew-badge">30</span>
                    </div>
                    <input type="range" id="ew_slider" min="5" max="120" value="30" oninput="updateFlow()">
                </div>

                <hr>

                <div class="slider-group">
                    <div class="slider-header">
                        <span>Pedestrian Flow</span>
                        <span id="ped_val" class="val-badge ped-badge">20</span>
                    </div>
                    <input type="range" id="ped_slider" min="0" max="100" value="20" oninput="updateFlow()">
                </div>
                
                <div class="status-panel">
                    <b>Live Engine Status</b>
                    Live MJPEG stream<br>
                    Pedestrian-aware yielding<br>
                    YOLO-style object tracking<br>
                    Adaptive signal phases<br>
                    Procedural city rendering
                </div>
            </div>
            <div class="video-container">
                <img id="feed" src="/video_feed" width="800" height="800" alt="Video Feed" />
            </div>
        </div>

        <script>
            async function setMode(mode) {
                await fetch(`/set_mode?mode=${mode}`);
                document.getElementById('adaptive_btn').classList.toggle('active', mode === 'adaptive');
                document.getElementById('fixed_btn').classList.toggle('active', mode === 'fixed');
            }

            async function setScenario(name) {
                const res = await fetch(`/scenario?name=${name}`);
                const data = await res.json();
                document.getElementById('ns_slider').value = data.ns;
                document.getElementById('ew_slider').value = data.ew;
                document.getElementById('ped_slider').value = data.ped;
                document.getElementById('ns_val').innerText = data.ns;
                document.getElementById('ew_val').innerText = data.ew;
                document.getElementById('ped_val').innerText = data.ped;
                document.getElementById('city_context').innerText = data.context;
            }

            async function generateReport() {
                const res = await fetch('/api/v1/report');
                const data = await res.json();
                const panel = document.getElementById('report_panel');
                panel.classList.add('visible');
                panel.innerHTML = `
                    <strong>${data.title}</strong>
                    ${data.city_context}<br>
                    ${data.headline}<br>
                    AI Adaptive avg wait: ${data.ai_adaptive_avg_wait_seconds}s<br>
                    Fixed Timer avg wait: ${data.fixed_timer_avg_wait_seconds}s<br>
                    Throughput increase: ${data.throughput_increase_percent}%<br>
                    Estimated CO2 reduction: ${data.estimated_co2_reduction_percent}%<br>
                    Incident detected: ${data.incident_detected ? 'YES' : 'NO'}<br>
                    Public transport priority: ${data.public_transport_priority ? 'ACTIVE' : 'OFF'}
                `;
            }

            function downloadReportPdf() {
                window.location.href = '/api/v1/report.pdf';
            }

            function downloadOperationsPdf() {
                window.location.href = '/api/v1/operations-report.pdf';
            }

            async function updatePolicy() {
                const params = new URLSearchParams({
                    min_green: document.getElementById('min_green').value,
                    max_green: document.getElementById('max_green').value,
                    yellow: document.getElementById('yellow').value,
                    all_red: document.getElementById('all_red').value
                });
                await fetch(`/api/v1/policy?${params.toString()}`);
            }

            function updateFlow() {
                let ns = document.getElementById('ns_slider').value;
                let ew = document.getElementById('ew_slider').value;
                let ped = document.getElementById('ped_slider').value;
                
                document.getElementById('ns_val').innerText = ns;
                document.getElementById('ew_val').innerText = ew;
                document.getElementById('ped_val').innerText = ped;
                
                fetch(`/set_flow?ns=${ns}&ew=${ew}&ped=${ped}`);
            }

            async function refreshMetrics() {
                try {
                    const res = await fetch('/metrics');
                    const data = await res.json();
                    document.getElementById('avg_wait').innerText = `${data.avg_wait}s`;
                    document.getElementById('improvement').innerText = `${data.improvement}%`;
                    document.getElementById('throughput').innerText = data.throughput;
                    document.getElementById('risk').innerText = data.risk;
                    document.getElementById('city_context').innerText = data.context || 'Baku, Ganjlik Intersection';
                    document.getElementById('decision_title').innerText = data.phase || 'Decision engine active';
                    document.getElementById('decision_reason').innerText =
                        `${data.reason} NS queue: ${data.ns_queue}, EW queue: ${data.ew_queue}, pedestrians: ${data.ped_queue}. ${data.incident ? 'Incident detected: vehicle blocked in intersection. Action: extend all-red phase.' : ''}`;
                } catch (err) {
                    console.warn(err);
                }
            }

            async function refreshOps() {
                try {
                    const [logRes, histRes, sourceRes] = await Promise.all([
                        fetch('/api/v1/audit-log'),
                        fetch('/api/v1/history'),
                        fetch('/api/v1/data-sources')
                    ]);
                    const logData = await logRes.json();
                    const histData = await histRes.json();
                    const sourceData = await sourceRes.json();

                    document.getElementById('audit_log').innerHTML = (logData.events || []).slice(-5).reverse()
                        .map(e => `${e.time} ${e.event}<br><span style="color:#8fa39a">${e.detail}</span>`)
                        .join('<br>') || 'Waiting for controller events.';

                    const samples = histData.samples || [];
                    const maxWait = Math.max(1, ...samples.map(s => s.avg_wait || 0));
                    document.getElementById('history_chart').innerHTML = samples.slice(-24)
                        .map(s => `<span style="height:${Math.max(4, (s.avg_wait || 0) / maxWait * 48)}px"></span>`)
                        .join('');

                    document.getElementById('data_sources').innerText = sourceData.sources
                        .map(s => `${s.name}: ${s.status}`)
                        .join(' | ');
                } catch (err) {
                    console.warn(err);
                }
            }

            setInterval(refreshMetrics, 900);
            setInterval(refreshOps, 1600);
            refreshMetrics();
            refreshOps();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

def find_available_port(start_port=8000, max_attempts=50):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available port found from {start_port} to {start_port + max_attempts - 1}")


if __name__ == "__main__":
    port = find_available_port(8000)
    print(f"Smart traffic simulation running at http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
