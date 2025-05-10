def read_ints_binary(file_path, byteorder='little', signed=False, max_count=None):
    ints = []
    with open(file_path, 'rb') as f:
        idx = 0
        while True:
            b = f.read(4)
            if len(b) < 4:
                break
            val = int.from_bytes(b, byteorder=byteorder, signed=signed)
            ints.append(val)
            idx += 1
            if max_count is not None and idx >= max_count:
                break
    return ints