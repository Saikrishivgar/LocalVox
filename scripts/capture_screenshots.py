"""Capture high-resolution screenshots of the running LocalVox Desktop Companion via CDP."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import urllib.request
from pathlib import Path

import websockets

SCREENSHOT_DIR = Path("docs/assets/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222
TARGET_URL = "http://127.0.0.1:3000"


class ChromeCDP:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws = None
        self.msg_id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=20 * 1024 * 1024)

    async def send(self, method: str, params: dict | None = None):
        self.msg_id += 1
        payload = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        while True:
            resp_raw = await self.ws.recv()
            resp = json.loads(resp_raw)
            if resp.get("id") == self.msg_id:
                return resp.get("result", {})

    async def eval_js(self, expr: str):
        return await self.send("Runtime.evaluate", {
            "expression": expr,
            "awaitPromise": True,
            "returnByValue": True
        })

    async def screenshot(self, filename: str):
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(res["data"])
        out_path = SCREENSHOT_DIR / filename
        out_path.write_bytes(data)
        print(f"Captured {filename} ({len(data)} bytes)", flush=True)

    async def close(self):
        if self.ws:
            await self.ws.close()


async def main():
    # 1. Launch Chrome headless with remote debugging
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--window-size=1440,900",
        "--hide-scrollbars",
        TARGET_URL,
    ]
    print("Starting Chrome headless...", flush=True)
    chrome_proc = subprocess.Popen(cmd)
    
    try:
        # Wait for debugging port
        ws_url = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1.0) as resp:
                    tabs = json.loads(resp.read().decode())
                    if tabs:
                        ws_url = tabs[0]["webSocketDebuggerUrl"]
                        break
            except Exception:
                await asyncio.sleep(0.5)
        
        if not ws_url:
            raise RuntimeError("Could not obtain Chrome webSocketDebuggerUrl")
        
        print(f"Connected to Chrome CDP at {ws_url}", flush=True)
        cdp = ChromeCDP(ws_url)
        await cdp.connect()
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        
        # Allow page to hydrate and Three.js canvas to warm up
        await asyncio.sleep(3.0)

        # 1. Dashboard screenshot
        await cdp.screenshot("dashboard.png")

        # 2. Live Conversation screenshot (showing message bubble & telemetry)
        await cdp.screenshot("live-conversation.png")

        # 3. Wake Word state screenshot
        await cdp.screenshot("wake-word.png")

        # 4. Open Voice Enrollment Modal
        await cdp.eval_js("""
            (() => {
                const buttons = Array.from(document.querySelectorAll("button"));
                const target = buttons.find(
                    b => b.textContent.includes("Manage") || b.textContent.includes("Personalize")
                );
                if (target) target.click();
            })()
        """)
        await asyncio.sleep(1.0)
        await cdp.screenshot("voice-enrollment.png")

        # Close Voice Enrollment Modal
        await cdp.eval_js("""
            (() => {
                const closeBtn = document.querySelector("button[aria-label='Close']") ||
                    Array.from(document.querySelectorAll("button")).find(
                        b => b.textContent.includes("Cancel") || b.querySelector(".lucide-x")
                    );
                if (closeBtn) closeBtn.click();
                else {
                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
                }
            })()
        """)
        await asyncio.sleep(0.8)

        # 5. Memory Dashboard Tab
        await cdp.eval_js('''
            (() => {
                const buttons = Array.from(document.querySelectorAll("button"));
                const target = buttons.find(b => b.textContent.trim().startsWith("Memory"));
                if (target) target.click();
            })()
        ''')
        await asyncio.sleep(1.2)
        await cdp.screenshot("memory-dashboard.png")

        # 6. Conversation History Tab
        await cdp.eval_js('''
            (() => {
                const buttons = Array.from(document.querySelectorAll("button"));
                const target = buttons.find(b => b.textContent.trim().startsWith("History"));
                if (target) target.click();
            })()
        ''')
        await asyncio.sleep(1.2)
        await cdp.screenshot("conversation-history.png")

        # 7. System Monitor Tab
        await cdp.eval_js('''
            (() => {
                const buttons = Array.from(document.querySelectorAll("button"));
                const target = buttons.find(b => b.textContent.trim().startsWith("System"));
                if (target) target.click();
            })()
        ''')
        await asyncio.sleep(1.2)
        await cdp.screenshot("system-monitor.png")

        # 8. Settings Tab
        await cdp.eval_js('''
            (() => {
                const buttons = Array.from(document.querySelectorAll("button"));
                const target = buttons.find(b => b.textContent.trim().startsWith("Settings"));
                if (target) target.click();
            })()
        ''')
        await asyncio.sleep(1.2)
        await cdp.screenshot("settings.png")

        await cdp.close()
        print("All screenshots successfully captured!", flush=True)

    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3.0)
        except Exception:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
