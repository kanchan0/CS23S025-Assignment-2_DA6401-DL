import lightning.pytorch as L
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class iNaturalistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, img_size, data_augmentation):
        super().__init__()

        self.train_path = data_dir / "train"
        self.test_path = data_dir / "val"
        self.batch_size = batch_size
        self.num_workers = num_workers

        if data_augmentation == "Y":

            self.data_transform = transforms.Compose(
                [
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.AutoAugment(),  # This is the data augmentation method chosen
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4712, 0.4600, 0.3896], std=[0.2034, 0.1981, 0.1948]
                    ),  # These values are calculated for our dataset
                ]
            )

        elif data_augmentation == "N":

            self.data_transform = transforms.Compose(
                [
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4712, 0.4600, 0.3896], std=[0.2034, 0.1981, 0.1948]
                    ),
                ]
            )

        self.test_data_transform = transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4748, 0.4645, 0.3965], std=[0.2004, 0.1954, 0.1923]
                ),
            ]
        )

    def setup(self, stage):
        if stage == "fit":

            # First load all the training data
            train_data_full = datasets.ImageFolder(
                root=self.train_path,
                transform=self.data_transform,
                target_transform=None,
            )

            # Decide the number of validation samples required
            validation_samples_per_class = int(0.2 * 1000)

            # These lists will hold the indices of training and validation data samples
            train_indices = []
            val_indices = []

            for class_idx in range(len(train_data_full.classes)):

                # Obtain the indices of each class
                class_indices = [
                    idx
                    for idx, (_, label) in enumerate(train_data_full.imgs)
                    if label == class_idx
                ]

                # Split and add the indices to the respective lists
                val_indices.extend(class_indices[:validation_samples_per_class])
                train_indices.extend(class_indices[validation_samples_per_class:])

            # Create a two subsets of the initially loaded training data as training data and validation data
            self.train_data = Subset(train_data_full, train_indices)
            self.val_data = Subset(train_data_full, val_indices)

        if stage == "test":
            # Load the test data
            self.test_data = datasets.ImageFolder(
                root=self.test_path, transform=self.test_data_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
