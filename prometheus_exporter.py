import time
from http.server import HTTPServer
from prometheus_client import make_wsgi_app, REGISTRY

# Create WSGI application for Prometheus metrics
app = make_wsgi_app(REGISTRY)

if __name__ == "__main__":
    # Start HTTP server to expose metrics
    httpd = HTTPServer(("", 8000), app)
    print("Serving metrics on port 8000...")
    httpd.serve_forever()
