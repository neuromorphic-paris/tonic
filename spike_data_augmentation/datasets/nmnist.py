import os
import os.path
import numpy as np
from .dataset import Dataset
from .utils import check_integrity, download_and_extract_archive


class NMNIST(Dataset):
    """NMNIST <https://www.garrickorchard.com/datasets/n-mnist> data set.

    arguments:
        train: choose training or test set
        save_to: location to save files to on disk
        transform: list of transforms to apply to the data
        download: choose to download data or not
    """

    base_url = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/"
    test_zip = base_url + "AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1"
    train_zip = base_url + "AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1"
    test_md5 = "69CA8762B2FE404D9B9BAD1103E97832"
    train_md5 = "20959B8E626244A1B502305A9E6E2031"
    test_filename = "nmnist_test.zip"
    train_filename = "nmnist_train.zip"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    sensor_size = (34, 34)
    ordering = "xytp"

    def __init__(
        self, save_to, train=True, transform=None, download=False, num_events=-1
    ):
        super(NMNIST, self).__init__(save_to, transform=transform)

        self.train = train
        self.location_on_system = os.path.join(save_to, "NMNIST")

        if train:
            self.url = self.train_zip
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "Train"
        else:
            self.url = self.test_zip
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "Test"

        file_path = self.location_on_system + "/" + self.folder_name

        numpy_cache_path = file_path + ".npz"

        if os.path.exists(numpy_cache_path):
            cache = np.load(numpy_cache_path)
            self.data_sizes = cache["data_sizes"]
            self.data_locations = cache["data_locations"]
            self.data = cache["data"]
            self.targets = cache["targets"]
        else:
            if download:
                self.download()

            if not self._check_integrity():
                raise RuntimeError(
                    "Dataset not found or corrupted."
                    + " You can use download=True to download it"
                )

            for path, dirs, files in os.walk(file_path):
                dirs.sort()
                for file in files:
                    if file.endswith("bin"):
                        events = self._read_dataset_file(path + "/" + file)
                        self.data.append(events)
                        label_number = int(path[-1])
                        self.targets.append(label_number)

            self.data_sizes = np.array([d.shape[0] for d in self.data])
            self.data_locations = np.cumsum(self.data_sizes)
            self.data = np.concatenate(self.data)
            self.targets = np.array(self.targets)

            print("Saving numpy cache %s" % numpy_cache_path)
            np.savez(
                numpy_cache_path,
                data_sizes=self.data_sizes,
                data_locations=self.data_locations,
                data=self.data,
                targets=self.targets,
            )

        self.num_events = num_events

    def __getitem__(self, index):
        target = self.targets[index]

        start_loc = self.data_locations[index - 1] if index - 1 > -1 else 0
        end_loc = self.data_locations[index]

        events = self.data[start_loc:end_loc, :]

        if self.num_events > 0:
            start_ind = max(
                0, int((events.shape[0] - self.num_events) * np.random.rand())
            )

            end_ind = min(start_ind + self.num_events, events.shape[0])
            events = events[start_ind:end_ind, :]

        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)

        return events, target

    def __len__(self):
        return self.targets.shape[0]

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )

    def _check_integrity(self):
        root = self.location_on_system
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.file_md5):
            return False
        return True

    def _read_dataset_file(self, filename):
        f = open(filename, "rb")
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = (
            ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        )

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        td = np.empty([td_indices.size, 4], dtype=np.int32)
        td[:, 0] = all_x[td_indices]
        td[:, 1] = all_y[td_indices]
        td[:, 2] = all_ts[td_indices]
        td[:, 3] = all_p[td_indices]

        return td
