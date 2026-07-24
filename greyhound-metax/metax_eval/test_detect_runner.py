import unittest
from unittest import mock

import numpy as np

import detect_runner


class FakeSharedMemory:
    def __init__(self, values):
        self.storage = bytearray(values.tobytes())
        self.buf = memoryview(self.storage)
        self.closed = False

    def close(self):
        self.buf.release()
        self.closed = True


class ReadShmRecordsTest(unittest.TestCase):
    def test_copies_wrapped_ring_and_closes_mapping(self):
        config = {"NUM_FIELDS": 2, "BUFFER_SIZE": 3, "METADATA_FIELDS": 7}
        values = np.zeros(13, dtype=np.int64)
        values[:7] = [2, 3, 3, 1, 1, 0xDEADBEEF, 8]
        values[7:] = [10, 11, 20, 21, 30, 31]
        fake = FakeSharedMemory(values)

        with mock.patch(
            "multiprocessing.shared_memory.SharedMemory", return_value=fake
        ):
            records = detect_runner.read_shm_records(config)

        np.testing.assert_array_equal(
            records, np.array([[20, 21], [30, 31], [10, 11]])
        )
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
