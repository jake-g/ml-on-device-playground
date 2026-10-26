# arduino_serial_handler.py
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal

_ESPRESSIF_VID = 0x303A
_SEEED_VID = 0x2886


class ArduinoSerialHandler(QThread):
    data_received = pyqtSignal(float, float, float)
    probability_received = pyqtSignal(list)
    connection_status = pyqtSignal(bool, str)
    serial_line_received = pyqtSignal(str)
    latency_received = pyqtSignal(int)
    info_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.running = False
        self.device_port = None

    def find_xiao_device(self):
        """Find XIAO nRF52840 Sense via USB while skipping Espressif boards."""
        try:
            from codegen_task import deploy
            port = deploy.find_port(required=False)
            if port:
                return port
        except Exception:
            pass

        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if getattr(port, "vid", None) == _ESPRESSIF_VID:
                continue
            desc = f"{port.description or ''} {port.manufacturer or ''}"
            if (
                getattr(port, "vid", None) == _SEEED_VID
                or "XIAO" in desc
                or "Seeed" in desc
            ):
                return port.device
        return None

    def connect(self):
        """Connect to the XIAO BLE Sense device."""
        try:
            device_port = self.find_xiao_device()

            if not device_port:
                self.connection_status.emit(
                    False, "XIAO BLE Sense not found. Check connection."
                )
                return False

            self.serial_port = serial.Serial(
                device_port, 115200, timeout=0.15, write_timeout=0.5
            )
            self.device_port = device_port
            self.running = True
            self.connection_status.emit(True, f"Connected to {device_port}")
            # Query device status and ensure live streaming mode on Codegen.ino
            self.send_command("LIVE")
            self.send_command("STATUS")
            return True

        except Exception as e:
            self.connection_status.emit(False, f"Connection error: {str(e)}")
            return False

    def send_command(self, cmd: str) -> bool:
        """Send a line command (such as STATUS or LIVE) to the Arduino."""
        port = self.serial_port
        if not port or not getattr(port, "is_open", False):
            return False
        try:
            payload = (cmd.strip() + "\n").encode("utf-8")
            port.write(payload)
            port.flush()
            return True
        except Exception:
            return False

    def disconnect(self):
        """Disconnect from the serial port."""
        was_running = self.running
        self.running = False
        if self.isRunning() and QThread.currentThread() != self:
            self.wait(600)
        port = self.serial_port
        self.serial_port = None
        if port is not None:
            try:
                if port.is_open:
                    port.close()
            except Exception:
                pass
        if was_running:
            self.connection_status.emit(False, "Disconnected")

    def run(self):
        """Thread main loop to read data."""
        if not self.connect():
            return

        while self.running:
            try:
                port = self.serial_port
                if port and port.is_open:
                    raw = port.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    self.serial_line_received.emit(line)

                    if line.startswith("IMU:"):
                        try:
                            data_part = line[4:].strip()
                            x, y, z = map(float, data_part.split(","))
                            self.data_received.emit(x, y, z)
                        except ValueError:
                            pass

                    elif line.startswith("PROB:"):
                        try:
                            data_part = line[5:].strip()
                            probabilities = [
                                float(p)
                                for p in data_part.split(",")
                                if p.strip()
                            ]
                            self.probability_received.emit(probabilities)
                        except ValueError as e:
                            print(f"Error parsing probability data: {e}")

                    elif line.startswith("LAT:"):
                        try:
                            us = int(float(line[4:].strip()))
                            self.latency_received.emit(us)
                        except ValueError:
                            pass

                    elif line.startswith(("INFO:", "ARENA:", "UNSURE:", "ERR:")):
                        self.info_received.emit(line)

            except Exception as e:
                if self.running:
                    print(f"Serial reading error: {str(e)}")
                self.running = False
                break

        self.disconnect()