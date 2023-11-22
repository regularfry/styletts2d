import asyncio
import os
import signal
import tts


class Server:
    def __init__(self, voice):
        self.voice = voice

    def connection_made(self, transport):
        self.transport = transport
        self.address = transport.get_extra_info("socket")


    def data_received(self, data):
        self.voice.say(data.decode())
        self.transport.close()


    def connection_lost(self, exc): pass


class VerboseDecorator:
    def __init__(self, server):
        self.server = server


    def connection_made(self, transport):
        self.server.connection_made(transport)
        print(f"Accepted connection from {self.server.address}")


    def data_received(self, data):
        print(f"Received {data!r} from {self.server.address}")
        self.server.data_received(data)


    def connection_lost(self, exc):
        if exc:
            print(f"Client {self.server.address} error: {exc}")
        else:
            print(f"Client {self.server.address} closed the connection")


def start(socket_path, socket_server_factory):
    loop = asyncio.get_event_loop()
    unix_server = loop.create_unix_server(socket_server_factory, socket_path)
    server = loop.run_until_complete(unix_server)
    loop.add_signal_handler(signal.SIGINT, lambda: loop.stop())
    loop.add_signal_handler(signal.SIGTERM, lambda: loop.stop())
    print(f"Listening on socket: {socket_path}")
    try:
        loop.run_forever()
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()
        os.unlink(socket_path)
        print("Server stopped.")


def main(socket, reference_audio, verbose=False):
    print(f"Starting server on socket: {socket}")
    voice = tts.Voice(reference_audio)
    def server_factory():
        server = Server(voice)
        if verbose: server = VerboseDecorator(server)
        return server

    start(socket, server_factory)

