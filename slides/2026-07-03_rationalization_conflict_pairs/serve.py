import http.server
import socketserver
import sys
import socket

orig_end_headers = http.server.SimpleHTTPRequestHandler.end_headers
def end_headers(self):
    self.send_header('Cache-Control', 'no-store')
    orig_end_headers(self)
http.server.SimpleHTTPRequestHandler.end_headers = end_headers

def is_free(p):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(('localhost', p)) != 0

if len(sys.argv) > 1:
    port = int(sys.argv[1])
    if not is_free(port):
        print(f"Error: Port {port} is already in use.")
        sys.exit(1)
else:
    port = next((p for p in range(8000, 9001) if is_free(p)), None)
    if port is None:
        print("Error: No free port in range 8000-9000.")
        sys.exit(1)

print(f"Serving on http://localhost:{port}")
socketserver.TCPServer(('', port), http.server.SimpleHTTPRequestHandler).serve_forever()
