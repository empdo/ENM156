"""
example:
    from pi_server import StatusServer

    server = StatusServer(port=8765)
    server.start()              # runs in a background thread
    server.fault("fault1")
    server.fault("normal")
    server.stop()
"""

import asyncio
import json
import threading

import websockets

NORMAL = "normal"
FAULT1 = "fault1"
FAULT2 = "fault2"
VALID_STATUSES = (NORMAL, FAULT1, FAULT2)


class StatusServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.status = NORMAL
        self._clients = set()
        self._loop = None
        self._thread = None
        self._stop_event = None
        self._ready = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def stop(self):
        if self._thread is None:
            return
        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join()
        self._thread = None
        self._ready.clear()

    def set_status(self, status):
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}, expected one of {VALID_STATUSES}")
        if status == self.status:
            return
        self.status = status
        print(f"Status -> {status}")
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._broadcast, self._message())

    def normal(self):
        self.set_status(NORMAL)

    def fault1(self):
        self.set_status(FAULT1)

    def fault2(self):
        self.set_status(FAULT2)

    def _message(self):
        return json.dumps({"status": self.status})

    def _broadcast(self, message):
        websockets.broadcast(self._clients, message)

    async def _handler(self, websocket):
        self._clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        try:
            await websocket.send(self._message())
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")

    async def _serve(self):
        self._stop_event = asyncio.Event()
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            self._ready.set()
            await self._stop_event.wait()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()
            self._loop = None
