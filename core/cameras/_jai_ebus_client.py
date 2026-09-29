"""
eBUS SDK Client for JAI GigE Vision cameras
Official JAI/Pleora SDK Python API wrapper

Key findings from research:
- CreateAndConnect() can hang indefinitely if camera is locked (CCP)
- Use PvDeviceGEV directly to have more control
- GetAccessType() checks camera availability WITHOUT connecting
- GevHeartbeatTimeout controls how long camera waits for heartbeat (default 3s)
- Camera stays locked for up to 45 seconds after improper disconnect
"""

import sys
import time
import subprocess
import socket
import struct
import multiprocessing
from typing import Optional, List, Dict, Tuple
import numpy as np
from utils.logger import logger
from utils.network_utils import (
    get_all_interfaces_detailed,
    is_same_subnet,
    is_link_local,
    find_compatible_interface,
    suggest_pc_ip_for_camera,
    configure_interface_ip
)


class eBUSClient:
    """Client for connecting to JAI cameras using eBUS SDK Python API"""

    def __init__(self):
        """Initialize eBUS client"""
        self.ebus_available = self._check_ebus()
        self.device = None
        self.stream = None
        self.pipeline = None
        self.is_streaming = False

        if self.ebus_available:
            # Import eBUS modules
            try:
                import eBUS as eb
                self.eb = eb
            except ImportError:
                logger.error("Failed to import eBUS module")
                self.ebus_available = False

    def _check_ebus(self) -> bool:
        """Check if eBUS SDK is available"""
        try:
            import eBUS
            return True
        except ImportError:
            return False

    def diagnose_network(self, camera_ip: str) -> Dict:
        """
        Diagnose network configuration for camera connectivity.

        Args:
            camera_ip: Camera's IP address

        Returns:
            Dict: Diagnostic information
        """
        diag = {
            'camera_ip': camera_ip,
            'is_link_local': is_link_local(camera_ip),
            'pc_interfaces': [],
            'compatible_interface': None,
            'problem': None,
            'solution': None
        }

        # Get all PC interfaces
        interfaces = get_all_interfaces_detailed()
        diag['pc_interfaces'] = interfaces

        # Log all interfaces
        logger.info("="*60)
        logger.info("NETWORK DIAGNOSTIC")
        logger.info("="*60)
        logger.info(f"Camera IP: {camera_ip}")
        logger.info(f"Is Link-Local (APIPA): {diag['is_link_local']}")
        logger.info("")
        logger.info("PC Network Interfaces:")

        for iface in interfaces:
            logger.info(f"  [{iface['name']}]")
            logger.info(f"    IP: {iface['ip']}")
            logger.info(f"    Subnet: {iface['subnet']}")
            logger.info(f"    MAC: {iface['mac']}")

        # Find compatible interface
        compatible = find_compatible_interface(camera_ip)
        diag['compatible_interface'] = compatible

        if compatible:
            logger.info("")
            logger.info(f"COMPATIBLE INTERFACE FOUND: {compatible['name']}")
            logger.info(f"  PC IP: {compatible['ip']} - Camera IP: {camera_ip}")
            logger.info("  Network connectivity should work!")
        else:
            diag['problem'] = "NO_COMPATIBLE_INTERFACE"
            suggested_ip = suggest_pc_ip_for_camera(camera_ip)

            logger.error("")
            logger.error("="*60)
            logger.error("PROBLEM DETECTED: NO COMPATIBLE INTERFACE!")
            logger.error("="*60)
            logger.error("")
            logger.error(f"Camera IP ({camera_ip}) is NOT reachable from any PC interface.")
            logger.error("")

            if diag['is_link_local']:
                logger.error("Camera has Link-Local IP (169.254.x.x) which means:")
                logger.error("  1. Camera didn't get DHCP IP")
                logger.error("  2. PC must have IP in 169.254.x.x range to connect")
                logger.error("")

            diag['solution'] = {
                'suggested_pc_ip': suggested_ip,
                'subnet_mask': '255.255.0.0'
            }

            logger.error("SOLUTIONS:")
            logger.error("")
            logger.error("Option 1: Configure PC network adapter manually")
            logger.error(f"  Set IP: {suggested_ip}")
            logger.error("  Subnet: 255.255.0.0")
            logger.error("  Gateway: (leave empty)")
            logger.error("")
            logger.error("Option 2: Use eBUS Player to set camera IP")
            logger.error("  C:\\Program Files\\JAI\\eBUS SDK\\bin\\eBUSPlayer.exe")
            logger.error("  Device → Communication Control → Set persistent IP")
            logger.error("")
            logger.error("Option 3: Run this app as Administrator for auto-config")
            logger.error("="*60)

        return diag

    def check_device_access(self, ip_address: str) -> Dict:
        """
        Check if device is available for connection WITHOUT actually connecting.

        Uses GetAccessType() which queries the device's CCP (Control Channel Privilege).
        Note: This check is optional - if it fails, we still try to connect.

        Returns:
            Dict with:
                - available: bool - True if device can be connected (or unknown)
                - access_type: str - Current access type
                - locked_by_other: bool - True if another application has control
                - message: str - Human readable status
        """
        result = {
            'available': True,  # Default to True - try connecting anyway
            'access_type': 'Unknown',
            'locked_by_other': False,
            'message': 'Access check skipped or unavailable'
        }

        if not self.ebus_available:
            result['message'] = 'eBUS SDK not available'
            result['available'] = False
            return result

        try:
            logger.info(f"Checking device access for: {ip_address}")

            # GetAccessType is tricky in Python - signature varies
            # We'll try multiple approaches
            access_type = None
            check_ok = False

            if hasattr(self.eb, 'PvDeviceGEV'):
                # Approach 1: Try creating a temp device and checking its access
                # This is safer than static method which has signature issues
                try:
                    temp_device = self.eb.PvDeviceGEV()
                    # Some versions have GetAccessType() on instance
                    if hasattr(temp_device, 'GetAccessType'):
                        # This might return just the access type after device is created
                        access_type = temp_device.GetAccessType()
                        check_ok = True
                        logger.info(f"GetAccessType returned: {access_type}")
                    del temp_device
                except Exception as e1:
                    logger.debug(f"Instance GetAccessType failed: {e1}")

            if check_ok and access_type is not None:
                # Analyze access type
                access_str = str(access_type)
                logger.info(f"Access type string: {access_str}")

                if 'Open' in access_str or 'open' in access_str:
                    result['access_type'] = 'Open'
                    result['available'] = True
                    result['message'] = 'Device appears available'
                elif 'Control' in access_str:
                    result['access_type'] = 'Control'
                    result['locked_by_other'] = True
                    result['available'] = False
                    result['message'] = 'Device may be locked by another application'
                else:
                    result['access_type'] = access_str
                    result['available'] = True
                    result['message'] = f'Access type: {access_str}, will try connecting'
            else:
                # Could not check - proceed anyway
                result['message'] = 'Access check unavailable, will try connecting directly'
                logger.info(result['message'])

        except Exception as e:
            # Don't block on access check failure
            result['message'] = f'Access check skipped: {e}'
            result['available'] = True  # Try anyway
            logger.warning(result['message'])

        return result

    def connect_gev_direct(self, ip_address: str, timeout_seconds: int = 30) -> bool:
        """
        Connect to GigE Vision device - SIMPLIFIED VERSION.
        Skip all checks, just try to connect with timeout.
        """
        # Skip this method entirely - go straight to CreateAndConnect fallback
        logger.info("connect_gev_direct: Skipping (using CreateAndConnect instead)")
        return False

    def force_ip(self, device_info: Dict, new_ip: str, subnet: str = "255.255.255.0", gateway: str = "") -> bool:
        """
        Force IP configuration on a GigE Vision camera.

        Args:
            device_info: Device info from discover_devices()
            new_ip: New IP address for camera
            subnet: Subnet mask
            gateway: Gateway address (optional)

        Returns:
            bool: True if successful
        """
        if not self.ebus_available:
            logger.error("eBUS SDK not available")
            return False

        try:
            mac = device_info.get('mac', '')
            if not mac:
                logger.error("MAC address not available")
                return False

            logger.info(f"Attempting ForceIP on camera {mac}")
            logger.info(f"  New IP: {new_ip}, Subnet: {subnet}, Gateway: {gateway}")

            # Use PvDeviceGEV for GigE-specific operations
            device_info_obj = device_info.get('device_info_object')
            if device_info_obj:
                # Cast to GEV device info if possible
                if hasattr(self.eb, 'PvDeviceInfoGEV'):
                    try:
                        # Try to use SetIPConfiguration if available
                        gev_info = self.eb.PvDeviceInfoGEV(device_info_obj)

                        # Convert IP strings to integers
                        ip_int = struct.unpack("!I", socket.inet_aton(new_ip))[0]
                        subnet_int = struct.unpack("!I", socket.inet_aton(subnet))[0]
                        gw_int = struct.unpack("!I", socket.inet_aton(gateway))[0] if gateway else 0

                        result = gev_info.SetIPConfiguration(ip_int, subnet_int, gw_int)
                        if result.IsOK():
                            logger.info("ForceIP successful!")
                            return True
                        else:
                            logger.error(f"ForceIP failed: {result.GetDescription()}")
                    except Exception as e:
                        logger.warning(f"PvDeviceInfoGEV method failed: {e}")

            # Alternative: Use raw GVCP ForceIP command
            logger.info("Trying raw GVCP ForceIP...")
            return self._send_gvcp_forceip(mac, new_ip, subnet, gateway)

        except Exception as e:
            logger.error(f"ForceIP error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _send_gvcp_forceip(self, mac: str, ip: str, subnet: str, gateway: str) -> bool:
        """
        Send raw GVCP FORCEIP command.

        This is a low-level implementation of the GigE Vision ForceIP command.
        """
        try:
            # Parse MAC address
            mac_bytes = bytes.fromhex(mac.replace(':', '').replace('-', ''))
            if len(mac_bytes) != 6:
                logger.error("Invalid MAC address format")
                return False

            # Convert IPs to bytes
            ip_bytes = socket.inet_aton(ip)
            subnet_bytes = socket.inet_aton(subnet)
            gateway_bytes = socket.inet_aton(gateway) if gateway else b'\x00\x00\x00\x00'

            # Build GVCP FORCEIP_CMD packet
            # Message type: 0x42 (FORCEIP_CMD)
            # Header: Flag(1) + Command(1) + Length(2) + ReqID(2)
            cmd_flag = 0x42
            cmd_code = 0x0004  # FORCEIP_CMD
            req_id = 1

            # Payload: MAC(6) + pad(2) + IP(4) + pad(12) + Subnet(4) + pad(12) + Gateway(4) + pad(12)
            payload = (
                b'\x00\x00' +  # Reserved
                mac_bytes +  # MAC address (6 bytes)
                b'\x00\x00' +  # Padding
                b'\x00\x00\x00\x00' +  # Reserved
                ip_bytes +  # IP address
                b'\x00' * 12 +  # Padding
                subnet_bytes +  # Subnet mask
                b'\x00' * 12 +  # Padding
                gateway_bytes +  # Gateway
                b'\x00' * 12  # Padding
            )

            # GVCP Header
            header = struct.pack('>BBHH', cmd_flag, 0x00, len(payload), req_id)
            packet = header + payload

            # Send to broadcast on GVCP port
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(2.0)

            # Send to broadcast
            sock.sendto(packet, ('255.255.255.255', 3956))

            logger.info("GVCP ForceIP packet sent, waiting for response...")

            # Wait for acknowledgment (optional)
            try:
                response, addr = sock.recvfrom(1024)
                logger.info(f"Response from {addr}: {response.hex()}")
            except socket.timeout:
                logger.info("No response (this is normal for ForceIP)")

            sock.close()
            logger.info("ForceIP command sent. Camera may need a few seconds to reconfigure.")
            return True

        except Exception as e:
            logger.error(f"GVCP ForceIP error: {e}")
            return False

    def discover_devices(self, timeout: float = 3.0) -> List[Dict]:
        """
        Discover GigE Vision devices using eBUS SDK

        Args:
            timeout: Discovery timeout in seconds

        Returns:
            List[Dict]: List of discovered devices
        """
        if not self.ebus_available:
            logger.warning("eBUS SDK not available. Install from: https://www.jai.com/support-software/jai-software")
            return []

        devices = []

        try:
            # Create system object
            system = self.eb.PvSystem()

            # Find all devices (GigE Vision)
            logger.info(f"Searching for eBUS devices (timeout: {timeout}s)...")
            system.Find()

            # Get device count
            interface_count = system.GetInterfaceCount()

            for i in range(interface_count):
                interface = system.GetInterface(i)

                # Get interface IP for diagnostics
                interface_ip = None
                try:
                    if hasattr(interface, 'GetIPAddress'):
                        interface_ip = interface.GetIPAddress()
                except:
                    pass

                # Get devices on this interface
                for j in range(interface.GetDeviceCount()):
                    device_info = interface.GetDeviceInfo(j)

                    # Convert all PvString objects to regular Python strings
                    device = {
                        'index': len(devices),
                        'device_info_object': device_info,  # СОХРАНЯЕМ САМ ОБЪЕКТ!
                        'interface_object': interface,  # Сохраняем интерфейс тоже
                        'interface_ip': str(interface_ip) if interface_ip else None,
                        'connection_id': str(device_info.GetConnectionID()),
                        'ip': str(device_info.GetIPAddress()),
                        'mac': str(device_info.GetMACAddress()),
                        'manufacturer': str(device_info.GetManufacturerInfo()),
                        'model': str(device_info.GetModelName()),
                        'name': str(device_info.GetUserDefinedName()),
                        'serial': str(device_info.GetSerialNumber()),
                        'protocol': 'eBUS SDK (GigE Vision)',
                        'type': str(device_info.GetType())
                    }

                    devices.append(device)
                    logger.info(f"Found: {device['manufacturer']} {device['model']} at {device['ip']}")
                    logger.info(f"  ConnectionID: '{device['connection_id']}'")
                    logger.info(f"  MAC: {device['mac']}, S/N: {device['serial']}")

        except Exception as e:
            logger.error(f"eBUS discovery error: {e}")

        logger.info(f"eBUS discovery completed: found {len(devices)} device(s)")
        return devices

    def connect(self, device_info: Dict = None, connection_id: str = None, ip_address: str = None) -> bool:
        """
        Connect to a device using CreateAndConnect with timeout.
        SIMPLIFIED - no fancy checks, just connect.
        """
        if not self.ebus_available:
            logger.error("eBUS SDK not available")
            return False

        try:
            # Disconnect if already connected
            if self.device:
                self.disconnect()

            # Get IP address for connection
            ip = None

            if device_info:
                ip = device_info.get('ip')
            elif ip_address:
                ip = ip_address
            elif connection_id:
                ip = connection_id
            else:
                logger.error("No device specified for connection")
                return False

            logger.info("="*60)
            logger.info(f"CONNECTING TO: {ip}")
            logger.info("="*60)
            logger.info("Using PvDevice.CreateAndConnect() with 30s timeout...")
            logger.info("If camera is locked, will timeout and show error.")
            logger.info("")

            # Use thread with timeout for CreateAndConnect
            import threading
            connect_result = [None, None]  # [result, device]
            connect_error = [None]

            def do_create_connect():
                try:
                    logger.info("Thread started: calling CreateAndConnect...")
                    connect_result[0], connect_result[1] = self.eb.PvDevice.CreateAndConnect(ip)
                    logger.info("Thread finished: CreateAndConnect returned")
                except Exception as e:
                    connect_error[0] = str(e)
                    logger.error(f"Thread exception: {e}")

            connect_thread = threading.Thread(target=do_create_connect)
            connect_thread.daemon = True
            connect_thread.start()

            # Wait with timeout
            logger.info("Waiting for connection (max 30 seconds)...")
            connect_thread.join(timeout=30)

            if connect_thread.is_alive():
                logger.error("="*60)
                logger.error("CONNECTION TIMEOUT (30 seconds)!")
                logger.error("="*60)
                logger.error("")
                logger.error("Camera is NOT responding. This usually means:")
                logger.error("")
                logger.error("1. CAMERA IS LOCKED (CCP Lock)")
                logger.error("   - Another app was connected and didn't disconnect properly")
                logger.error("   - Solution: POWER CYCLE the camera (turn off, wait 5s, turn on)")
                logger.error("   - Or wait 45 seconds for auto-unlock")
                logger.error("")
                logger.error("2. NETWORK ISSUE")
                logger.error("   - Check cable connection")
                logger.error("   - Disable firewall temporarily")
                logger.error("")
                logger.error("3. eBUS SDK ISSUE")
                logger.error("   - Try running eBUS Player first to verify SDK works")
                logger.error("="*60)
                return False

            if connect_error[0]:
                logger.error(f"Connection exception: {connect_error[0]}")
                return False

            result = connect_result[0]
            self.device = connect_result[1]

            if not result.IsOK():
                logger.error(f"PvDevice.CreateAndConnect failed!")
                logger.error(f"  Code: {result.GetCodeString()}")
                logger.error(f"  Description: {result.GetDescription()}")
                self.device = None
                return False

            logger.info("Device connected successfully!")

            # Configure GigE Vision specific settings
            if isinstance(self.device, self.eb.PvDeviceGEV):
                logger.info("GigE Vision device detected, configuring...")
                logger.info("Negotiating packet size...")
                self.device.NegotiatePacketSize()

            # Open stream
            logger.info(f"Opening stream to: {conn_id}")
            result, self.stream = self.eb.PvStream.CreateAndOpen(conn_id)

            if not result.IsOK():
                logger.error(f"PvStream.CreateAndOpen failed!")
                logger.error(f"  Code: {result.GetCodeString()}")
                logger.error(f"  Description: {result.GetDescription()}")
                self.device.Disconnect()
                self.device = None
                return False

            logger.info("Stream opened successfully!")

            # Configure GigE Vision stream destination
            if isinstance(self.device, self.eb.PvDeviceGEV):
                logger.info("Setting stream destination...")
                local_ip = self.stream.GetLocalIPAddress()
                local_port = self.stream.GetLocalPort()
                logger.info(f"  Local endpoint: {local_ip}:{local_port}")
                self.device.SetStreamDestination(local_ip, local_port)

            # Create pipeline for buffer management
            logger.info("Creating pipeline...")
            self.pipeline = self.eb.PvPipeline(self.stream)
            self.pipeline.SetBufferCount(16)
            self.pipeline.Start()

            logger.info("="*60)
            logger.info("CONNECTION SUCCESSFUL!")
            logger.info("="*60)
            return True

        except Exception as e:
            logger.error(f"Connection error: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def connect_alternative(self, device_index: int = 0) -> bool:
        """
        Alternative connection method using system discovery

        Args:
            device_index: Index of device to connect (default 0)

        Returns:
            bool: True if successful
        """
        if not self.ebus_available:
            logger.error("eBUS SDK not available")
            return False

        try:
            logger.info("Using alternative connection method...")

            # Create system and find devices
            system = self.eb.PvSystem()
            logger.info("Finding devices...")
            system.Find()

            # Get first interface
            if system.GetInterfaceCount() == 0:
                logger.error("No interfaces found")
                return False

            interface = system.GetInterface(0)
            logger.info(f"Found {interface.GetDeviceCount()} device(s) on interface")

            if interface.GetDeviceCount() == 0:
                logger.error("No devices found on interface")
                return False

            if device_index >= interface.GetDeviceCount():
                logger.error(f"Device index {device_index} out of range")
                return False

            # Get device info
            device_info = interface.GetDeviceInfo(device_index)
            conn_id = device_info.GetConnectionID()

            logger.info(f"Connecting to device {device_index}: {conn_id}")
            logger.info("Creating device and connecting...")

            # Create and connect
            result, self.device = self.eb.PvDevice.CreateAndConnect(conn_id)

            if not result.IsOK():
                logger.error(f"Failed to connect: {result.GetCodeString()}")
                return False

            logger.info("Device connected, opening stream...")
            result, self.stream = self.eb.PvStream.CreateAndOpen(conn_id)

            if not result.IsOK():
                logger.error(f"Failed to open stream: {result.GetCodeString()}")
                self.device.Disconnect()
                self.device = None
                return False

            logger.info("Creating pipeline...")
            self.pipeline = self.eb.PvPipeline(self.stream)
            self.pipeline.SetBufferCount(16)
            self.pipeline.Start()

            logger.info("Successfully connected using alternative method")
            return True

        except Exception as e:
            logger.error(f"Alternative connection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def disconnect(self):
        """Disconnect from device"""
        try:
            # Stop acquisition if running
            if self.is_streaming:
                self.stop_acquisition()

            # Stop and release pipeline
            if self.pipeline:
                if self.pipeline.IsStarted():
                    self.pipeline.Stop()
                del self.pipeline
                self.pipeline = None

            # Close stream
            if self.stream:
                if self.stream.IsOpen():
                    self.stream.Close()
                del self.stream
                self.stream = None

            # Disconnect device
            if self.device:
                if self.device.IsConnected():
                    self.device.Disconnect()
                del self.device
                self.device = None

            logger.info("Disconnected from device")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def start_acquisition(self) -> bool:
        """
        Start image acquisition

        Returns:
            bool: True if successful
        """
        if not self.device or not self.device.IsConnected():
            logger.error("Not connected to device")
            return False

        try:
            logger.info("Starting acquisition...")

            # Get device parameters
            logger.info("Getting device parameters...")
            params = self.device.GetParameters()
            logger.info("Device parameters retrieved")

            # Set acquisition mode to continuous
            logger.info("Setting AcquisitionMode to Continuous...")
            params.SetEnumValue("AcquisitionMode", "Continuous")
            logger.info("AcquisitionMode set")

            # Start acquisition
            logger.info("Executing AcquisitionStart command...")
            params.ExecuteCommand("AcquisitionStart")
            logger.info("AcquisitionStart executed")

            self.is_streaming = True
            logger.info("Started acquisition successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start acquisition: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def stop_acquisition(self):
        """Stop image acquisition"""
        if not self.device or not self.device.IsConnected():
            return

        try:
            # Stop acquisition
            params = self.device.GetParameters()
            params.ExecuteCommand("AcquisitionStop")

            self.is_streaming = False
            logger.info("Stopped acquisition")

        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")

    def grab_frame(self, timeout: int = 1000) -> Optional[np.ndarray]:
        """
        Grab a single frame

        Args:
            timeout: Timeout in milliseconds

        Returns:
            Optional[np.ndarray]: Frame as numpy array or None
        """
        if not self.pipeline or not self.pipeline.IsStarted():
            logger.error("Pipeline not started")
            return None

        try:
            # Retrieve next buffer (based on official PvStreamSample page 21-22)
            result, buffer, operational_result = self.pipeline.RetrieveNextBuffer(timeout)

            if not result.IsOK():
                logger.debug(f"Failed to retrieve buffer: {result.GetCodeString()}")
                return None

            if not operational_result.IsOK():
                logger.debug(f"Buffer not complete: {operational_result.GetCodeString()}")
                self.pipeline.ReleaseBuffer(buffer)
                return None

            # Check payload type (page 23)
            payload_type = buffer.GetPayloadType()
            if payload_type != self.eb.PvPayloadTypeImage:
                logger.warning(f"Unexpected payload type: {payload_type}")
                self.pipeline.ReleaseBuffer(buffer)
                return None

            # Get image data (page 24)
            image = buffer.GetImage()
            width = image.GetWidth()
            height = image.GetHeight()
            pixel_type = image.GetPixelType()

            logger.debug(f"Frame: {width}x{height}, PixelType: {pixel_type}")

            # Get raw data pointer
            raw_data = image.GetDataPointer()

            # Handle different pixel formats (page 24)
            frame = None
            if pixel_type == self.eb.PvPixelMono8:
                # Mono8: 1 byte per pixel
                frame = np.frombuffer(raw_data, dtype=np.uint8, count=width*height)
                frame = frame.reshape((height, width))
            elif pixel_type == self.eb.PvPixelRGB8:
                # RGB8: 3 bytes per pixel
                frame = np.frombuffer(raw_data, dtype=np.uint8, count=width*height*3)
                frame = frame.reshape((height, width, 3))
            elif pixel_type == self.eb.PvPixelBGR8:
                # BGR8: 3 bytes per pixel
                frame = np.frombuffer(raw_data, dtype=np.uint8, count=width*height*3)
                frame = frame.reshape((height, width, 3))
            else:
                logger.warning(f"Unsupported pixel format: {pixel_type}")
                self.pipeline.ReleaseBuffer(buffer)
                return None

            # Release buffer back to pipeline
            self.pipeline.ReleaseBuffer(buffer)

            return frame

        except Exception as e:
            logger.error(f"Error grabbing frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_device_info(self) -> Dict:
        """
        Get connected device information

        Returns:
            Dict: Device information
        """
        if not self.device or not self.device.IsConnected():
            return {}

        try:
            info = {}

            # Get device info
            device_info = self.device.GetDeviceInfo()

            info['manufacturer'] = device_info.GetManufacturerInfo()
            info['model'] = device_info.GetModelName()
            info['serial'] = device_info.GetSerialNumber()
            info['ip'] = device_info.GetIPAddress()
            info['mac'] = device_info.GetMACAddress()
            info['name'] = device_info.GetUserDefinedName()

            # Get device parameters
            params = self.device.GetParameters()

            # Try to get common parameters
            try:
                info['width'] = params.GetIntegerValue("Width")
                info['height'] = params.GetIntegerValue("Height")
                info['pixel_format'] = params.GetEnumValueString("PixelFormat")
            except:
                pass

            return info

        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {}

    def set_parameter(self, param_name: str, value) -> bool:
        """
        Set device parameter

        Args:
            param_name: Parameter name (e.g., "ExposureTime", "Gain")
            value: Parameter value

        Returns:
            bool: True if successful
        """
        if not self.device or not self.device.IsConnected():
            logger.error("Not connected to device")
            return False

        try:
            params = self.device.GetParameters()

            # Try to set parameter (type detection)
            param = params.Get(param_name)

            if param.GetType() == self.eb.PvGenTypeInteger:
                params.SetIntegerValue(param_name, int(value))
            elif param.GetType() == self.eb.PvGenTypeFloat:
                params.SetFloatValue(param_name, float(value))
            elif param.GetType() == self.eb.PvGenTypeEnum:
                params.SetEnumValue(param_name, str(value))
            elif param.GetType() == self.eb.PvGenTypeBoolean:
                params.SetBooleanValue(param_name, bool(value))
            else:
                logger.warning(f"Unsupported parameter type for {param_name}")
                return False

            logger.info(f"Set {param_name} = {value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set parameter {param_name}: {e}")
            return False


def discover_ebus(timeout: float = 3.0) -> List[Dict]:
    """
    Convenience function for eBUS device discovery

    Args:
        timeout: Discovery timeout in seconds

    Returns:
        List[Dict]: List of discovered devices
    """
    client = eBUSClient()
    return client.discover_devices(timeout=timeout)
