import h5py
import numpy as np
import torch


class CalorimeterJetDataset(torch.utils.data.Dataset):

    def __init__(self, hdf5_dataset=None, return_pt=False):
        """Generator for calorimeter and jet data"""

        self.hdf5_dataset = hdf5_dataset
        self.return_pt = return_pt

        self.channels = 2  # Number of color channels of the input images
        self.height = 360  # Height of the input images
        self.width = 340  # Width of the input images

        self.labels = self.hdf5_dataset['labels']

        self.ecal_energy = self.hdf5_dataset['ecal_energy']
        self.ecal_phi = self.hdf5_dataset['ecal_phi']
        self.ecal_eta = self.hdf5_dataset['ecal_eta']

        self.hcal_energy = self.hdf5_dataset['hcal_energy']
        self.hcal_phi = self.hdf5_dataset['hcal_phi']
        self.hcal_eta = self.hdf5_dataset['hcal_eta']

        self.dataset_size = len(self.labels)

    def process_labels(self, labels_raw, scaler_mass):
        labels_reshaped = labels_raw.reshape(-1, 5)
        labels = torch.empty_like(labels_reshaped, dtype=torch.float32)

        # Set fractional coordinates
        labels[:, 0] = (labels_reshaped[:, 1] - 23) / float(self.width)
        labels[:, 1] = (labels_reshaped[:, 2] - 23) / float(self.height)
        labels[:, 2] = (labels_reshaped[:, 1] + 23) / float(self.width)
        labels[:, 3] = (labels_reshaped[:, 2] + 23) / float(self.height)

        # Set class label
        labels[:, 4] = labels_reshaped[:, 0]

        # Concatinate auxilary labels
        labels_reshaped[:, 4] = labels_reshaped[:, 4] / scaler_mass
        labels = torch.cat((labels, labels_reshaped[:, 4].unsqueeze(1)), 1)

        if self.return_pt:
            labels = torch.cat((labels, labels_reshaped[:, 3].unsqueeze(1)), 1)

        return labels

    def process_images(self,
                       indices_ecal_phi, indices_hcal_phi,
                       indices_ecal_eta, indices_hcal_eta,
                       ecal_energy, hcal_energy):

        indices_channels = np.append(np.zeros(len(indices_ecal_phi)),
                                     np.ones(len(indices_hcal_phi)))
        indices_phi = np.append(indices_ecal_phi, indices_hcal_phi)
        indices_eta = np.append(indices_ecal_eta, indices_hcal_eta)
        energy = np.append(ecal_energy, hcal_energy)
        scaler = np.max(energy)
        energy = energy / scaler

        i = torch.LongTensor([indices_channels, indices_eta, indices_phi])
        v = torch.FloatTensor(energy)
        pixels = torch.sparse.FloatTensor(i, v, torch.Size([self.channels,
                                                            self.width,
                                                            self.height]))

        return pixels.to_dense(), scaler

    def __getitem__(self, index):

        # Repeat for events with no jets: Causes problems during training
        repeat = True

        while repeat:

            # Load calorimeter
            indices_ecal_phi = np.asarray(self.ecal_phi[index], dtype=np.int16)
            indices_ecal_eta = np.asarray(self.ecal_eta[index], dtype=np.int16)
            ecal_energy = np.asarray(self.ecal_energy[index], dtype=np.float32)

            indices_hcal_phi = np.asarray(self.hcal_phi[index], dtype=np.int16)
            indices_hcal_eta = np.asarray(self.hcal_eta[index], dtype=np.int16)
            hcal_energy = np.asarray(self.hcal_energy[index], dtype=np.float32)

            calorimeter, scaler = self.process_images(indices_ecal_phi,
                                                      indices_hcal_phi,
                                                      indices_ecal_eta,
                                                      indices_hcal_eta,
                                                      ecal_energy,
                                                      hcal_energy)

            # Set labels
            labels_raw = np.asarray([self.labels[index]], dtype=np.float32)
            labels_raw = torch.FloatTensor(labels_raw)
            labels_processed = self.process_labels(labels_raw, scaler)

            if labels_processed.shape[0]:
                repeat = False
            else:
                index += 1

        return calorimeter, labels_processed

    def __len__(self):
        return self.dataset_size
