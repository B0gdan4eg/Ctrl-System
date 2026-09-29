#!/usr/bin/env python3
"""
JAI Camera Viewer - eBUS SDK Edition
Automatically discovers, connects to JAI camera and displays live video
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger, logger
from connection.ebus_client import eBUSClient


class JAICameraViewer:
    """Simple JAI camera viewer using eBUS SDK"""

    def __init__(self):
        self.client = eBUSClient()

    def print_banner(self):
        """Print application banner"""
        print("\n" + "="*70)
        print("    JAI Camera Viewer - eBUS SDK Edition")
        print("    Automatic Discovery & Live Video Stream")
        print("="*70 + "\n")

    def discover_and_connect(self):
        """Discover cameras and connect to the first one"""
        if not self.client.ebus_available:
            print("ERROR: eBUS SDK not available!")
            print("Install eBUS Python API: py -3.11 -m pip install ebus_python-6.5.4-7277_jai-py311-none-win_amd64.whl")
            return False

        print("Searching for JAI cameras...")
        devices = self.client.discover_devices(timeout=5.0)

        if not devices:
            print("\nNo cameras found!")
            print("\nPossible reasons:")
            print("  1. Camera is not powered on")
            print("  2. Network cable is not connected")
            print("  3. Firewall is blocking GigE Vision packets")
            return False

        # Show found devices
        print(f"\nFound {len(devices)} camera(s):")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device['manufacturer']} {device['model']} at {device['ip']}")
            print(f"      S/N: {device['serial']}, MAC: {device['mac']}")

        camera_ip = devices[0]['ip']

        # Run network diagnostics
        print("\n" + "="*70)
        print("NETWORK DIAGNOSTICS")
        print("="*70)
        diag = self.client.diagnose_network(camera_ip)

        if diag['problem']:
            print("\n" + "="*70)
            print("WARNING: Network configuration issue detected!")
            print("="*70)
            print(f"\nCamera IP: {camera_ip}")
            print(f"Problem: {diag['problem']}")
            if diag['solution']:
                print(f"\nSuggested PC IP: {diag['solution']['suggested_pc_ip']}")
                print(f"Subnet mask: {diag['solution']['subnet_mask']}")
            print("\nAttempting connection anyway...")
            print("="*70)

        # Connect to first camera
        print(f"\nConnecting to camera [0]...")
        print(f"Camera: {devices[0]['manufacturer']} {devices[0]['model']}")
        print(f"IP: {devices[0]['ip']}, S/N: {devices[0]['serial']}")
        print(f"Connection ID: {devices[0]['connection_id']}")

        print("\nConnecting...")
        if self.client.connect(device_info=devices[0]):
            print("Successfully connected!")

            # Show camera info
            info = self.client.get_device_info()
            if info:
                print("\nCamera Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")

            return True
        else:
            print("\n" + "="*70)
            print("CONNECTION FAILED!")
            print("="*70)
            print("\nCamera was FOUND but connection FAILED.")
            print("")
            print("Possible causes:")
            print("  1. Another application has exclusive access to camera")
            print("  2. Firewall blocking GigE Vision ports (3956 UDP)")
            print("  3. Network interface issue")
            print("")
            print("Quick fixes:")
            print("  1. Close eBUS Player if running")
            print("  2. Run as Administrator")
            print("  3. Disable Windows Firewall temporarily")
            print("  4. Check jai_camera.log for detailed error")
            print("")
            if camera_ip.startswith('169.254.'):
                print(f"Camera IP: {camera_ip} (Link-Local)")
                print("PC network adapter should have IP like: 169.254.x.x")
                print("")
            print("="*70)
            return False

    def stream_video(self):
        """Stream and display video from camera"""
        print("\n" + "="*70)
        print("Starting live video stream...")
        print("Controls:")
        print("  - Press 'q' or ESC to exit")
        print("  - Press 's' to save current frame")
        print("="*70 + "\n")

        # Start acquisition
        if not self.client.start_acquisition():
            print("Failed to start acquisition")
            return

        frame_count = 0
        saved_count = 0
        timeout_counter = 0
        max_initial_timeouts = 50  # Wait up to 50 seconds for first frame

        print("Waiting for frames...")

        try:
            while True:
                # Grab frame
                frame = self.client.grab_frame(timeout=1000)

                if frame is None:
                    # Count timeouts before first frame
                    if frame_count == 0:
                        timeout_counter += 1
                        print(f"Waiting for first frame... ({timeout_counter}s)", end='\r')
                        if timeout_counter >= max_initial_timeouts:
                            print(f"\n\nERROR: No frames received after {max_initial_timeouts} seconds")
                            print("Possible issues:")
                            print("  1. Camera is not streaming")
                            print("  2. Network/firewall blocking")
                            print("  3. Incorrect camera configuration")
                            break
                    continue

                # First frame received
                if frame_count == 0:
                    print(f"\nFirst frame received! Size: {frame.shape}")

                frame_count += 1

                # Convert to 8-bit if needed
                if frame.dtype != np.uint8:
                    frame = (frame / frame.max() * 255).astype(np.uint8)

                # Convert grayscale to BGR for display
                if len(frame.shape) == 2:
                    frame_display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Assume RGB, convert to BGR for OpenCV
                    frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_display = frame

                # Add frame counter
                cv2.putText(
                    frame_display,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Display frame
                cv2.imshow('JAI Camera - eBUS SDK', frame_display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    print("\nStopping stream...")
                    break
                elif key == ord('s'):  # s to save
                    filename = f"frame_{saved_count:04d}.png"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")
                    saved_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            logger.error(f"Stream error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"\nStream error: {e}")

        finally:
            # Cleanup (page 25-26 of official guide)
            print("\nStopping acquisition...")
            self.client.stop_acquisition()
            cv2.destroyAllWindows()
            print(f"Total frames captured: {frame_count}")
            print(f"Frames saved: {saved_count}")

    def run(self):
        """Main application flow"""
        self.print_banner()

        try:
            # Discover and connect
            if not self.discover_and_connect():
                return

            # Stream video
            self.stream_video()

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Disconnect
            print("\nDisconnecting...")
            self.client.disconnect()
            print("Done.\n")


def main():
    """Application entry point"""
    # Setup logger
    setup_logger(log_file="jai_camera.log")

    try:
        # Run application
        app = JAICameraViewer()
        app.run()
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        print("\n" + "="*70)
        print("FATAL ERROR!")
        print("="*70)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        print("Press Enter to exit...")
        input()


if __name__ == "__main__":
    main()
