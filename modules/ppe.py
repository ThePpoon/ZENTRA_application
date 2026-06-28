# modules/ppe.py — PPE Detection (stub — model not yet implemented)
# All functions are no-ops. Re-implement when the PPE detection module is ready.

stats = {"frames": 0, "violations": 0}


def on_frame(frame, meta, window_title=""):
    stats["frames"] += 1


def on_data(data, meta, frame=None):
    pass


def draw_predictions(frame, predictions):
    return frame


def draw_person_status(frame, predictions, tracks):
    return frame


def get_fps():
    return 0.0


def send_line_notify(msg, image=None, level="warning", **kwargs):
    pass
