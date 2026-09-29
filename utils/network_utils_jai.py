"""
Network utilities for camera discovery
"""

import socket
import struct
import ipaddress
import platform
import subprocess
import re
from typing import List, Tuple, Optional, Dict


def get_all_interfaces_detailed() -> List[Dict]:
    """
    Get detailed information about all network interfaces on Windows.

    Uses multiple methods for reliability:
    1. Parse ipconfig /all output
    2. Fallback to socket.gethostbyname_ex()
    3. Fallback to netifaces if available

    Returns:
        List[Dict]: List of interface info dicts with name, ip, subnet, gateway
    """
    interfaces = []

    # Method 1: Try ipconfig (Windows only)
    if platform.system() == "Windows":
        try:
            # Run ipconfig /all and parse output
            result = subprocess.run(
                ['ipconfig', '/all'],
                capture_output=True,
                text=True,
                encoding='cp866',  # Use DOS codepage for Russian Windows
                errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            current_interface = None

            for line in result.stdout.split('\n'):
                line = line.strip()

                # New adapter section (match both English and Russian)
                if ('adapter' in line.lower() or 'адаптер' in line.lower()) and ':' in line:
                    if current_interface and current_interface.get('ip'):
                        interfaces.append(current_interface)

                    # Extract adapter name
                    name_match = re.search(r'(?:adapter|адаптер)\s+(.+?):', line, re.IGNORECASE)
                    if name_match:
                        current_interface = {
                            'name': name_match.group(1).strip(),
                            'ip': None,
                            'subnet': None,
                            'gateway': None,
                            'mac': None,
                            'description': None
                        }

                elif current_interface:
                    # Parse IPv4 Address (English and Russian)
                    if ('IPv4' in line or 'IP-адрес' in line or 'IP Address' in line) and ':' in line:
                        ip_match = re.search(r':\s*(\d+\.\d+\.\d+\.\d+)', line)
                        if ip_match:
                            ip = ip_match.group(1)
                            # Skip localhost and invalid IPs
                            if not ip.startswith('127.') and not ip.startswith('0.'):
                                current_interface['ip'] = ip

                    # Parse Subnet Mask
                    elif 'Subnet Mask' in line or 'Маска подсети' in line:
                        mask_match = re.search(r':\s*(\d+\.\d+\.\d+\.\d+)', line)
                        if mask_match:
                            current_interface['subnet'] = mask_match.group(1)

                    # Parse Default Gateway
                    elif 'Default Gateway' in line or 'Основной шлюз' in line:
                        gw_match = re.search(r':\s*(\d+\.\d+\.\d+\.\d+)', line)
                        if gw_match:
                            current_interface['gateway'] = gw_match.group(1)

                    # Parse Physical Address (MAC)
                    elif 'Physical Address' in line or 'Физический адрес' in line:
                        mac_match = re.search(r':\s*([0-9A-Fa-f-]+)', line)
                        if mac_match:
                            current_interface['mac'] = mac_match.group(1).replace('-', ':')

                    # Parse Description
                    elif 'Description' in line or 'Описание' in line:
                        desc_match = re.search(r':\s*(.+)', line)
                        if desc_match:
                            current_interface['description'] = desc_match.group(1)

            # Add last interface
            if current_interface and current_interface.get('ip'):
                interfaces.append(current_interface)

        except Exception as e:
            print(f"Error parsing ipconfig: {e}")

    # Method 2: Fallback to socket if ipconfig didn't work
    if not interfaces:
        try:
            hostname = socket.gethostname()
            # Get all IPs for this host
            _, _, ip_list = socket.gethostbyname_ex(hostname)

            for ip in ip_list:
                if not ip.startswith('127.'):
                    interfaces.append({
                        'name': 'Socket Interface',
                        'ip': ip,
                        'subnet': '255.255.255.0',  # Default assumption
                        'gateway': None,
                        'mac': None,
                        'description': f'Interface from socket ({hostname})'
                    })
        except Exception as e:
            print(f"Error getting interfaces via socket: {e}")

    # Method 3: Try to get ALL local IPs via UDP trick
    if not interfaces:
        try:
            # This trick gets the IP that would be used to reach the internet
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            try:
                # Doesn't actually connect, just determines route
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith('127.'):
                    interfaces.append({
                        'name': 'Default Route',
                        'ip': ip,
                        'subnet': '255.255.255.0',
                        'gateway': None,
                        'mac': None,
                        'description': 'Default routing interface'
                    })
            except:
                pass
            finally:
                s.close()
        except:
            pass

    # Method 4: Also add link-local if camera uses link-local
    # Try to detect link-local interface
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        try:
            # Try to route to link-local address
            s.connect(('169.254.1.1', 80))
            ip = s.getsockname()[0]
            if ip and ip.startswith('169.254.'):
                # Check if we already have this
                existing_ips = [iface.get('ip') for iface in interfaces]
                if ip not in existing_ips:
                    interfaces.append({
                        'name': 'Link-Local Interface',
                        'ip': ip,
                        'subnet': '255.255.0.0',
                        'gateway': None,
                        'mac': None,
                        'description': 'Link-Local/APIPA interface'
                    })
        except:
            pass
        finally:
            s.close()
    except:
        pass

    return interfaces


def is_same_subnet(ip1: str, ip2: str, subnet_mask: str = "255.255.0.0") -> bool:
    """
    Check if two IP addresses are in the same subnet.

    Args:
        ip1: First IP address
        ip2: Second IP address
        subnet_mask: Subnet mask (default 255.255.0.0 for link-local)

    Returns:
        bool: True if in same subnet
    """
    try:
        # Convert to integers
        ip1_int = struct.unpack("!I", socket.inet_aton(ip1))[0]
        ip2_int = struct.unpack("!I", socket.inet_aton(ip2))[0]
        mask_int = struct.unpack("!I", socket.inet_aton(subnet_mask))[0]

        # Check if network portions match
        return (ip1_int & mask_int) == (ip2_int & mask_int)
    except Exception:
        return False


def is_link_local(ip: str) -> bool:
    """
    Check if IP is in link-local range (169.254.x.x / APIPA)

    Args:
        ip: IP address

    Returns:
        bool: True if link-local
    """
    return ip.startswith("169.254.")


def find_compatible_interface(camera_ip: str) -> Optional[Dict]:
    """
    Find a PC interface that is compatible with the camera IP.

    Args:
        camera_ip: Camera's IP address

    Returns:
        Optional[Dict]: Interface info if found, None otherwise
    """
    interfaces = get_all_interfaces_detailed()

    for iface in interfaces:
        if iface.get('ip'):
            # For link-local addresses, use 255.255.0.0 mask
            if is_link_local(camera_ip):
                if is_same_subnet(iface['ip'], camera_ip, "255.255.0.0"):
                    return iface
            else:
                # Use interface's subnet mask
                mask = iface.get('subnet', '255.255.255.0')
                if is_same_subnet(iface['ip'], camera_ip, mask):
                    return iface

    return None


def suggest_pc_ip_for_camera(camera_ip: str) -> str:
    """
    Suggest a compatible IP for PC based on camera IP.

    Args:
        camera_ip: Camera's IP address

    Returns:
        str: Suggested PC IP
    """
    try:
        parts = camera_ip.split('.')
        # Use .100 in same subnet
        parts[3] = '100'
        return '.'.join(parts)
    except:
        return "169.254.1.100"


def configure_interface_ip(interface_name: str, ip: str, subnet: str = "255.255.0.0") -> Tuple[bool, str]:
    """
    Configure static IP on a Windows network interface using netsh.
    REQUIRES ADMINISTRATOR PRIVILEGES!

    Args:
        interface_name: Name of the network interface
        ip: IP address to set
        subnet: Subnet mask

    Returns:
        Tuple[bool, str]: (success, message)
    """
    if platform.system() != "Windows":
        return False, "Only supported on Windows"

    try:
        # Build netsh command
        cmd = [
            'netsh', 'interface', 'ip', 'set', 'address',
            f'name={interface_name}',
            'static',
            ip,
            subnet
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )

        if result.returncode == 0:
            return True, f"Successfully set {interface_name} to {ip}/{subnet}"
        else:
            error = result.stderr or result.stdout
            if "requires elevation" in error.lower() or "access is denied" in error.lower():
                return False, "Administrator privileges required. Run as Administrator."
            return False, f"Failed: {error}"

    except Exception as e:
        return False, f"Error: {e}"


def get_local_ip() -> str:
    """
    Get local IP address

    Returns:
        str: Local IP address
    """
    try:
        # Create a socket and connect to external host (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def get_network_interfaces() -> List[Tuple[str, str]]:
    """
    Get list of network interfaces with their IP addresses

    Returns:
        List[Tuple[str, str]]: List of (interface_name, ip_address)
    """
    interfaces = []
    hostname = socket.gethostname()

    try:
        # Get all addresses for the hostname
        addr_info = socket.getaddrinfo(hostname, None)
        for addr in addr_info:
            if addr[0] == socket.AF_INET:  # IPv4
                ip = addr[4][0]
                if not ip.startswith("127."):
                    interfaces.append((hostname, ip))
    except Exception:
        pass

    # Add default local IP
    local_ip = get_local_ip()
    if local_ip != "127.0.0.1":
        interfaces.append(("default", local_ip))

    return list(set(interfaces))  # Remove duplicates


def get_local_network_cidr(ip: str = None) -> str:
    """
    Get local network in CIDR notation (e.g., 192.168.1.0/24)

    Args:
        ip: IP address to calculate network from (default: local IP)

    Returns:
        str: Network in CIDR notation
    """
    if ip is None:
        ip = get_local_ip()

    try:
        # Assume /24 subnet for typical local networks
        network = ipaddress.IPv4Network(f"{ip}/24", strict=False)
        return str(network)
    except Exception:
        return "192.168.1.0/24"


def get_broadcast_address(ip: str = None) -> str:
    """
    Get broadcast address for the local network

    Args:
        ip: IP address to calculate broadcast from (default: local IP)

    Returns:
        str: Broadcast address
    """
    if ip is None:
        ip = get_local_ip()

    try:
        network = ipaddress.IPv4Network(f"{ip}/24", strict=False)
        return str(network.broadcast_address)
    except Exception:
        return "255.255.255.255"


def is_host_alive(ip: str, port: int = 3956, timeout: float = 1.0) -> bool:
    """
    Check if host is alive by attempting to connect to a port

    Args:
        ip: IP address to check
        port: Port to check (default: 3956 for GigE Vision)
        timeout: Connection timeout in seconds

    Returns:
        bool: True if host responds, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def ip_to_int(ip: str) -> int:
    """
    Convert IP address string to integer

    Args:
        ip: IP address string

    Returns:
        int: IP address as integer
    """
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def int_to_ip(ip_int: int) -> str:
    """
    Convert integer to IP address string

    Args:
        ip_int: IP address as integer

    Returns:
        str: IP address string
    """
    return socket.inet_ntoa(struct.pack("!I", ip_int))


def generate_ip_range(network_cidr: str) -> List[str]:
    """
    Generate list of all IP addresses in a network

    Args:
        network_cidr: Network in CIDR notation (e.g., "192.168.1.0/24")

    Returns:
        List[str]: List of IP addresses
    """
    try:
        network = ipaddress.IPv4Network(network_cidr, strict=False)
        # Exclude network and broadcast addresses
        return [str(ip) for ip in network.hosts()]
    except Exception:
        return []


def get_mac_vendor(mac: str) -> Optional[str]:
    """
    Get vendor name from MAC address OUI (first 3 octets)

    Args:
        mac: MAC address string

    Returns:
        Optional[str]: Vendor name if known, None otherwise
    """
    # Common OUI prefixes for camera manufacturers
    oui_database = {
        "00:07:8E": "JAI Corporation",
        "00:0C:DF": "JAI Corporation",
        # Add more as needed
    }

    if not mac or len(mac) < 8:
        return None

    oui = mac[:8].upper()
    return oui_database.get(oui)


def format_mac(mac: str) -> str:
    """
    Format MAC address to standard format (XX:XX:XX:XX:XX:XX)

    Args:
        mac: MAC address in any format

    Returns:
        str: Formatted MAC address
    """
    # Remove common separators
    mac = mac.replace(":", "").replace("-", "").replace(".", "")

    # Insert colons every 2 characters
    if len(mac) == 12:
        return ":".join(mac[i:i+2] for i in range(0, 12, 2)).upper()
    return mac.upper()


def is_windows() -> bool:
    """Check if running on Windows"""
    return platform.system() == "Windows"


def is_linux() -> bool:
    """Check if running on Linux"""
    return platform.system() == "Linux"
