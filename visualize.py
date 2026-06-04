import subprocess
import sys


def main() -> None:
    if len(sys.argv) == 1:
        print("Running static visualizations (fast)...")
        subprocess.run([sys.executable, "visualize_static.py"], check=True)
        print("\nTo run convergence animations (slow), execute:")
        print("  python visualize_animations.py")
    elif "--static" in sys.argv:
        subprocess.run([sys.executable, "visualize_static.py"], check=True)
    elif "--animations" in sys.argv:
        subprocess.run([sys.executable, "visualize_animations.py"], check=True)
    elif "--all" in sys.argv:
        subprocess.run([sys.executable, "visualize_static.py"], check=True)
        subprocess.run([sys.executable, "visualize_animations.py"], check=True)
    else:
        print("Usage: python visualize.py [--static | --animations | --all]")


if __name__ == "__main__":
    main()
