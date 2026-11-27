import time

from pi_server import StatusServer

server = StatusServer(port=8765)
server.start()

try:
    while True:
        server.normal()
        time.sleep(5)
        server.fault1()
        time.sleep(5)
        server.fault2()
        time.sleep(5)
except KeyboardInterrupt:
    server.stop()
