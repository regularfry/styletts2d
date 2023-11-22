import argparse
import server
import sys
import tts

def main():
    # Parses a command line argument called "socket" to get the socket filename to listen on,
    # then starts a server on that socket.
    parser = argparse.ArgumentParser()
    parser.add_argument("socket", help="socket filename to listen on")
    parser.add_argument("reference_audio", help="reference audio filename")
    parser.add_argument("-e", "--espeak-path", help="path to the espeak library (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print verbose output")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.espeak_path:
        tts.set_espeak_path(args.espeak_path)

    try:
        server.main(args.socket, args.reference_audio, args.verbose)
    except tts.EspeakNotFoundError:
        print("espeak not found. Please install espeak or specify the path to the espeak library with the -e option.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()