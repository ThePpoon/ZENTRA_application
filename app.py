import threading
import time
import webview
from pathlib import Path


def start_server():
    import uvicorn
    from server.api import app as fastapi_app
    uvicorn.run(fastapi_app, host="127.0.0.1", port=7788, log_level="warning")


class JsApi:
    def open_file_dialog(self):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="เลือกไฟล์วิดีโอ",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or ""

    def save_file(self, content: str, filename: str):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="บันทึกไฟล์",
            defaultextension=".csv",
            initialfile=filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            Path(path).write_text(content, encoding="utf-8-sig")
        return path or ""

    def toggle_fullscreen(self):
        if webview.windows:
            webview.windows[0].toggle_fullscreen()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1.0)

    window = webview.create_window(
        title="ZENTRA Safety AI System",
        url="http://127.0.0.1:7788/",
        width=1280,
        height=800,
        min_size=(1024, 640),
        js_api=JsApi(),
        background_color="#0d1b2a",
    )
    webview.start(debug=False)
