import struct
import os


class BinaryReader:
    """
    A helper class to read structured binary data from a stream.
    """

    def __init__(self, stream, endian="<"):
        self.stream = stream
        self.endian = endian

    def read_bytes(self, num_bytes: int) -> bytes:
        if num_bytes == 0:
            return b""
        data = self.stream.read(num_bytes)
        if len(data) < num_bytes:
            raise EOFError(f"Tried to read {num_bytes} bytes, but only got {len(data)}.")
        return data

    def read_struct(self, fmt: str, num_bytes: int) -> tuple:
        return struct.unpack(self.endian + fmt, self.read_bytes(num_bytes))

    # --- Basic Data Types ---
    def read_u8(self) -> int:
        return self.read_struct("B", 1)[0]

    def read_i8(self) -> int:
        return self.read_struct("b", 1)[0]

    def read_u16(self) -> int:
        return self.read_struct("H", 2)[0]

    def read_i16(self) -> int:
        return self.read_struct("h", 2)[0]

    def read_u32(self) -> int:
        return self.read_struct("I", 4)[0]

    def read_i32(self) -> int:
        return self.read_struct("i", 4)[0]

    def read_f32(self) -> float:
        return self.read_struct("f", 4)[0]

    # --- 3D Graphics Data Types ---
    def read_vec3(self) -> tuple:
        return self.read_struct("fff", 12)

    def read_quat(self) -> tuple:
        return self.read_struct("ffff", 16)

    def read_color(self) -> tuple:
        return self.read_struct("ffff", 16)

    def read_quat16(self) -> tuple:
        x, y, z, w = self.read_struct("hhhh", 8)
        return (x / 32767.0, y / 32767.0, z / 32767.0, w / 32767.0)

    # --- String Reading ---
    def read_string(self, length: int = None, encoding: str = 'utf-8') -> str:
        """
        Reads a string from the stream.
        If length is None, reads a U32-prefixed string.
        If length is provided, reads that many bytes and decodes as a string.
        """
        if length is None:
            length = self.read_u32()
            if length == 0:
                return ""
            return self.read_bytes(length).decode(encoding, errors="ignore")
        else:
            if length == 0:
                return ""
            return self.read_bytes(length).decode(encoding, errors="ignore")

    # --- Stream Control ---
    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)

    def is_eof(self) -> bool:
        """Checks if the stream pointer is at or past the end of the file."""
        current_pos = self.tell()
        try:
            self.seek(0, os.SEEK_END)
            end_pos = self.tell()
            self.seek(current_pos, os.SEEK_SET)
            return current_pos >= end_pos
        except (IOError, OSError):
            return False
