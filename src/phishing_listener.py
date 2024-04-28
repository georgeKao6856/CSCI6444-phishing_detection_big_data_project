import socket
import os

# Define the host and port to listen on
HOST = 'localhost'
PORT = 5678

filename = 'phishing_email.txt'

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address
server_socket.bind((HOST, PORT))

# Listen for incoming connections
server_socket.listen(5)

print(f"Server listening on {HOST}:{PORT}")

while True:
    # Accept a new connection
    client_socket, address = server_socket.accept()
    print(f"Connected to {address}")

    # Receive data from the client
    data = client_socket.recv(1024)
    if not data:
        break

    print(f"Received from client: {data.decode('utf-8')}")

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    f = open(filename,append_write)
    f.write(data.decode('utf-8')+"\n")
    f.close()

    # Send a response back to the client
    client_socket.sendall(b"Message received. Thanks!")

    # Close the connection
    client_socket.close()
