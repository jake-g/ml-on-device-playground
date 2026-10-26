# arduino_serial_handler.py
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal

class ArduinoSerialHandler(QThread):
    data_received = pyqtSignal(float, float, float)
    gyro_received = pyqtSignal(float, float, float)  # GYR: deg/s (MotionSynth)
    probability_received = pyqtSignal(list)  # New signal for probability data
    connection_status = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.running = False

    def find_xiao_device(self):
        """Find XIAO BLE Sense device connected via USB"""
        xiao_device = None
        ports = list(serial.tools.list_ports.comports())

        for port in ports:
            # Look for Seeed XIAO devices - may need to adjust these identifiers
            if "XIAO" in port.description or "Seeed" in port.description:
                xiao_device = port.device
                break

        return xiao_device

    def connect(self):
        """Connect to the XIAO BLE Sense device"""
        try:
            device_port = self.find_xiao_device()

            if not device_port:
                self.connection_status.emit(False, "XIAO BLE Sense not found. Check connection.")
                return False

            # Short timeout so disconnect() wakes cleanly without closing the fd mid-read
            self.serial_port = serial.Serial(device_port, 115200, timeout=0.15)
            self.running = True
            self.connection_status.emit(True, f"Connected to {device_port}")
            return True

        except Exception as e:
            self.running = False
            self.connection_status.emit(False, f"Connection error: {str(e)}")
            return False

    def _close_port(self):
        """Close the underlying serial port if open."""
        port = self.serial_port
        self.serial_port = None
        if port is not None:
            try:
                if port.is_open:
                    port.close()
            except Exception:
                pass

    def disconnect(self):
        """Disconnect from the serial port cleanly without racing readline()."""
        was_running = self.running
        self.running = False
        if self.isRunning() and QThread.currentThread() != self:
            self.wait(600)
        self._close_port()
        if was_running:
            self.connection_status.emit(False, "Disconnected")

    def run(self):
        """Thread main loop to read data"""
        if not self.connect():
            return

        try:
            while self.running:
                try:
                    if not (self.serial_port and self.serial_port.is_open):
                        break
                    raw = self.serial_port.readline()
                    if not self.running or not raw:
                        continue
                    line = raw.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    if line.startswith("IMU:"):
                        try:
                            data_part = line[4:].strip()
                            x, y, z = map(float, data_part.split(','))
                            self.data_received.emit(x, y, z)
                        except ValueError:
                            pass

                    elif line.startswith("GYR:"):
                        try:
                            gx, gy, gz = map(float, line[4:].strip().split(','))
                            self.gyro_received.emit(gx, gy, gz)
                        except ValueError:
                            pass

                    elif line.startswith("PROB:"):
                        try:
                            data_part = line[5:].strip()
                            probabilities = [float(p) for p in data_part.split(',') if p.strip()]
                            self.probability_received.emit(probabilities)
                        except ValueError:
                            pass

                except Exception as e:
                    if not self.running or "Bad file descriptor" in str(e):
                        break
                    print(f"Serial reading error: {str(e)}")
                    self.running = False
                    self.connection_status.emit(False, f"Serial lost: {str(e)}")
                    break
        finally:
            self._close_port()
            if self.running:
                self.running = False
                self.connection_status.emit(False, "Disconnected")