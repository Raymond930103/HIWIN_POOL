import socket

HOST = "0.0.0.0"      # or set to 192.168.0.155
PORT = 4000
REPLY = "ROBOT_OK"    # custom response text

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Dummy robot listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            data = conn.recv(1024)
            if not data:
                continue
            print("Received:", data.decode())
            conn.sendall(REPLY.encode())
            print("Responded with:", REPLY)
