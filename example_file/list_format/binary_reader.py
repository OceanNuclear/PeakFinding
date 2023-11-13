"""
Exploratory file for reading binary data (.Lis file)

I'm attempting to decode the .Lis format file using the .Lis file format manual (list-mode-file-formats.pdf)
that mysteriously appeared in here and here:
https://docplayer.net/57761118-1-0-introduction-ortec-maestro-list-mode-file-formats.html
https://www.ortec-online.com/-/media/ametekortec/manuals/list-mode-file-formats.pdf?la=en&revision=85d3b40e-f300-4ce4-9873-13cdc812083c

2021-07-15 00:32:19 My current understanding is:

# 1. To get to the bit representation described by list-mode-file-formats.pdf, I must first invert the ordering using get_bytes.
# (I don't want to think too hard/spend too much energy on figuring out why this works.
# It vaguely has to do with binary represntation and endianness.)

2. We can actually let struct handle this instead. We just have to know that the data is stored as little endian (afaik, since I've only tested it on a linux machine)

End note:
Using Python to do this conversion would be too slow.
In the future it would be desirable to use Fortran or C++ instead.
"""
import struct
import datetime as dt
import math

class BinaryData:
    def __init__(self, data):
        assert isinstance(data, bytes)
        self.data = data

    def get_bytes(self, offset, size):
        """
        Gets a chunk of data

        Parameters
        ----------
        offset: the offset value of the first byte
        size: the number of byes that this ch

        Returns
        -------
        binary_repr : a series of 
        """
        return "".join([bin(byte)[2:].zfill(8)[::-1] for byte in self.data[offset:offset+size]])[::-1]
        # the [2:] remove the '0b' which python uses to denote that this str is actually a binary number.
        # the .zfill addes the leading zeros back in.
        # the first [::-1] inverts each binary representation back to how it's supposed to be stored on the disk.
        # then we concatenate them together using .join.
        # The second[::-1] inverts these bytes back to the order that's more human readable.

    def get_repr(self, offset, size, fmt):
        return struct.unpack("@"+fmt, self.data[offset:offset+size])[0]

    def read_str(self, offset, size):
        binary_string = (self.get_repr(offset+i, 1, "s") for i in range(size))
        return "".join(c.decode() for c in binary_string if c!=b"\x00") # ignore all NULL's

    def read_float(self, offset):
        return self.get_repr(offset, 4, "f")

    def read_int32(self, offset):
        return self.get_repr(offset, 4, "i")

def read_header(data):
    # header ID: all .Lis header has header ID = -13.
    print(data.read_int32(0), "=-13;")

    # list data style
    list_data_style_ind = data.read_int32(4) # header
    list_data_style = {
        1:"digiBASE",
        2:"PRO List",
        3:"Not Used",
        4:"digiBASE-E",
    }[list_data_style_ind]
    print("list data style =",list_data_style)

    # start time
    start_OLE_time = data.get_repr(8, 8, 'd')
    offset_days = math.floor(start_OLE_time) if start_OLE_time>0 else math.ceil(start_OLE_time)
    acquisition_start = (dt.datetime(1899, 12, 30) +
                        dt.timedelta(days=offset_days)+
                        dt.timedelta(days=start_OLE_time - offset_days))
    print("acquisition started at", acquisition_start)

    print("device address =", data.read_str(16, 80))

    print("MCB Type string =", data.read_str(96, 9))

    print("device serial =", data.read_str(105, 16))

    print("text description =", data.read_str(121, 80))

    print("Energy calibration (below) is", 'not' if data.get_repr(201, 1, 's')==b'\x00' else '', "valid.")

    print("energy units =", data.read_str(202, 4))

    E_cal = data.read_float(206), data.read_float(210), data.read_float(214)
    print("For energy calibration, (p0, p1, p2) =", E_cal)

    print("Shape calibration (below) is", 'not' if data.get_repr(218, 1, 's')==b'\x00' else '', "valid.")

    shape_cal = data.read_float(219), data.read_float(223), data.read_float(227)
    print("For shape calibation, (p0, p1, p2) =", shape_cal)

    print("conversion gain", data.read_int32(231))

    print("Detector ID", data.read_int32(235))

    print("Real Time", data.read_float(239))

    print("Live Time", data.read_float(243))

    return

def read_PRO(data):
    pass

if __name__=="__main__":
    with open("TBMD1.Lis", 'br') as f:
        data = BinaryData(f.read())
    read_header(data)

