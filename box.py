import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau




#!pip install boxsdk


os.environ['BOX_CLIENT_ID'] = '' #put id
os.environ['BOX_CLIENT_SECRET'] = '' #secret id
os.environ['BOX_DEVELOPER_TOKEN'] = '' #annoying token that expires in 60 minutes, rerun when hitting limit to continue.

from boxsdk import OAuth2, Client

oauth = OAuth2(
    client_id=os.environ['BOX_CLIENT_ID'],
    client_secret=os.environ['BOX_CLIENT_SECRET'],
    access_token=os.environ['BOX_DEVELOPER_TOKEN'],
)
client = Client(oauth)
def get_folder_id_by_name(parent_folder_id: str, folder_name: str) -> str:
    items = client.folder(folder_id=parent_folder_id).get_items(limit=1000, offset=0)
    for item in items:
        if item.type == 'folder' and item.name.lower() == folder_name.lower():
            return item.id
    raise ValueError(f"No folder named '{folder_name}' in folder {parent_folder_id}")

def download_folder(folder_id: str, local_path: str):
    os.makedirs(local_path, exist_ok=True)
    for item in client.folder(folder_id=folder_id).get_items(limit=1000, offset=0):
        if item.type == 'file':
            dest_path = os.path.join(local_path, item.name)
            print(f'Downloading file: {item.name}')
            with open(dest_path, 'wb') as output_file:
                client.file(item.id).download_to(output_file)
        elif item.type == 'folder':
            subdir = os.path.join(local_path, item.name)
            print(f'Entering subfolder: {item.name}')
            download_folder(item.id, subdir)



project1_id = '' #put id here (in the link)
code_id      = get_folder_id_by_name(project1_id, 'code')
processed_id = get_folder_id_by_name(code_id, 'processed')


download_folder(processed_id, 'processed')

print("Check ./processed/")