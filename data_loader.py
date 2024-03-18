import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
#from mtcnn import MTCNN

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, csv_name, phase, num_triplets, epoch, triplet_batch_size,num_human_identities_per_batch=32,
                  training_triplets_path=None, transform=None):
        """
        Args:

        root_dir: Absolute path to dataset.
        training_dataset_csv_path: Path to csv file containing the image paths inside the training dataset folder.
        num_triplets: Number of triplets required to be generated.
        epoch: Current epoch number (used for saving the generated triplet list for this epoch).
        num_generate_triplets_processes: Number of separate Python processes to be created for the triplet generation
                                          process. A value of 0 would generate a number of processes equal to the
                                          number of available CPU cores.
        num_human_identities_per_batch: Number of set human identities per batch size.
        triplet_batch_size: Required number of triplets in a batch.
        training_triplets_path: Path to a pre-generated triplet numpy file to skip the triplet generation process (Only
                                 will be used for one epoch).
        transform: Required image transformation (augmentation) settings.
        """

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #  VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #  forcing the 'name' column as being of type 'int' instead of type 'object')
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': object})
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        self.epoch = epoch
        self.transform = transform
        self.phase=phase

        # Modified here to bypass having to use pandas.dataframe.loc for retrieving the class name
        #  and using dataframe.iloc for creating the face_classes dictionary
        df_dict = self.df.to_dict()
        self.df_dict_class_name = df_dict["name"]
        self.df_dict_id = df_dict["id"]
        self.df_dict_class_reversed = {value: key for (key, value) in df_dict["class"].items()}

        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets()
        else:
            print("Loading pre-generated triplets file ...")
            self.training_triplets = np.load(training_triplets_path)

    def make_dictionary_for_face_class(self):
        """
            face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        """
        face_classes = dict()
        for idx, label in enumerate(self.df['class']):
            if label not in face_classes:
                face_classes[label] = []
            # Instead of utilizing the computationally intensive pandas.dataframe.iloc() operation
            face_classes[label].append(self.df_dict_class_name[idx])
            
        return face_classes

    def generate_triplets(self):
        triplets = []
        classes = self.df['class'].unique()
        face_classes = self.make_dictionary_for_face_class()

        print("\nGenerating {} triplets ...".format(self.num_triplets))
        num_training_iterations_per_process = self.num_triplets / self.triplet_batch_size
        progress_bar = tqdm(range(int(num_training_iterations_per_process)))  # tqdm progress bar does not iterate through float numbers

        for training_iteration in progress_bar:

            """
            For each batch: 
                - Randomly choose set amount of human identities (classes) for each batch
            
                  - For triplet in batch:
                      - Randomly choose anchor, positive and negative images for triplet loss
                      - Anchor and positive images in pos_class
                      - Negative image in neg_class
                      - At least, two images needed for anchor and positive images in pos_class
                      - Negative image should have different class as anchor and positive images by definition
            """
            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)

            for triplet in range(self.triplet_batch_size):

                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)

                while len(face_classes[pos_class]) < 2:
                    pos_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)

                # Instead of utilizing the computationally intensive pandas.dataframe.loc() operation
                pos_name_index = self.df_dict_class_reversed[pos_class]
                pos_name = self.df_dict_class_name[pos_name_index]

                neg_name_index = self.df_dict_class_reversed[neg_class]
                neg_name = self.df_dict_class_name[neg_name_index]

                if len(face_classes[pos_class]) == 2:
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                else:
                    ianc = np.random.randint(0, len(face_classes[pos_class]))
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

                    while ianc == ipos:
                        ipos = np.random.randint(0, len(face_classes[pos_class]))

                ineg = np.random.randint(0, len(face_classes[neg_class]))

                triplets.append(
                    [
                        face_classes[pos_class][ianc],
                        face_classes[pos_class][ipos],
                        face_classes[neg_class][ineg],
                        pos_class,
                        neg_class,
                        pos_name,
                        neg_name
                    ]
                )

        print("Saving training triplets list in 'datasets/generated_triplets' directory ...")
        np.save('files/triplets/{}/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                self.phase,self.epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            ),
            triplets
        )
        
        print("Training triplets' list Saved!\n")

        return triplets



    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_class), str(anc_id)+'.jpg')
        pos_img = os.path.join(self.root_dir, str(pos_class), str(pos_id)+'.jpg')
        neg_img = os.path.join(self.root_dir, str(neg_class), str(neg_id)+'.jpg')

        # Modified to open as PIL image in the first place
        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)



def get_dataloader(train_root_dir, valid_root_dir,
                   train_csv_name, valid_csv_name,
                   num_train_triplets, num_valid_triplets,
                   batch_size, num_workers,epoch):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(299),
            #transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])}

    face_dataset = {
        'train': TripletFaceDataset(root_dir=train_root_dir,
                                    csv_name=train_csv_name,
                                    phase='train',
                                    num_triplets=num_train_triplets,
                                    epoch=epoch,
                                    triplet_batch_size=batch_size,
                                    transform=data_transforms['train']
                                    ),
        'valid': TripletFaceDataset(root_dir=valid_root_dir, 
                                    csv_name=valid_csv_name,
                                    phase='valid',
                                    num_triplets=num_valid_triplets,
                                    epoch=epoch,
                                    triplet_batch_size=batch_size,
                                    transform=data_transforms['valid'])
        
        }
        

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x in ['train', 'valid']}
        #for x in ['train']}
    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}
    #data_size={x:len(face_dataset[x]) for x in ['train']} 
    return dataloaders, data_size
