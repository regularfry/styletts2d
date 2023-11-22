# client.py

import socket
import sys

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python client.py <socket_filename> [text_to_say]")
        return

    socket_filename = sys.argv[1]
    # If no text is given, read from stdin.
    if len(sys.argv) == 2:
        text_to_say = sys.stdin.read()
    else:
        text_to_say = sys.argv[2]

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(socket_filename)
            client_socket.sendall(text_to_say.encode())
    except socket.error as e:
        print(f"Socket error: {e}")
        return

if __name__ == "__main__":
    main()