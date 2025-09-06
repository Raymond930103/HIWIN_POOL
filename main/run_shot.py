import argparse
from .communicate import tcp_communicate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a shot command to the robot")
    parser.add_argument("angle", type=float, help="shot angle in degrees")
    parser.add_argument("power", type=float, help="normalized shot power (0-1)")
    parser.add_argument("spin", type=float, nargs="?", default=0.0, help="optional spin value")
    args = parser.parse_args()

    success, resp = tcp_communicate.send_shot(args.angle, args.power, args.spin)
    if success:
        print(f"Robot response: {resp}")
    else:
        print(f"Failed to send shot: {resp}")
