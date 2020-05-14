import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from app.utils.utils import InvalidValueError

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class ApparelDataset(Dataset):
    """Apparel dataset

       Data loader class
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.apparel_meta = pd.read_csv(csv_file, error_bad_lines=False)
        self.apparel_meta['image'] = self.apparel_meta.apply(lambda row: str(row['id']) + ".jpg", axis=1)
        self.apparel_meta['genderArticle'] = self.apparel_meta.apply(
            lambda row: row['gender'] + '-' + row['articleType'], axis=1
        )
        self.apparel_meta.fillna('', inplace=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Calculate and return length of the dataset

        :return: length of the dataset
        """
        return len(self.apparel_meta)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset by index

        :param idx: index of the expected item
        :return: item at the index idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.apparel_meta.iloc[idx]

    def get_image(self, image_id):
        """
        Get an image from the dataset by it's id

        :param image_id: Id of the image
        :return: loaded image after applying transformation if there is any
        """
        if torch.is_tensor(image_id):
            image_id = image_id.tolist()
        img_name = os.path.join(self.root_dir, str(image_id) + ".jpg")
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image

    def get_candidate_meta(self, min_count=500, per_class=1000):
        """
        Group the metadata by sub-category and return only those which have minimum <min_count> in their group
        selecting top <per_class> from them

        :param min_count: minimum amount for the group to be eligible
        :param per_class: maximum per-group amount of metadata to take
        :return: modified dataset containing the specific amount of meta from the eligible groups
        """
        modified_image_meta = self.apparel_meta.groupby('subCategory').filter(lambda x: len(x) >= min_count)
        return modified_image_meta.groupby('subCategory').head(per_class)

    def get_all_meta(self):
        """
        Return all metadata without filtering or any modification

        :return: dataframe containing all metadata
        """
        return self.apparel_meta

    def filter_by_id(self, product_id):
        """
        Filter the metadata by image id. Return all metadata those have matching gender and articleType with the given
        product

        :param product_id: id of the expected product
        :return: dataframe of filtered metadata with matching gender and articleType
        """
        try:
            product_meta = self.apparel_meta[self.apparel_meta['id'] == product_id]
            if len(product_meta) != 1:
                raise InvalidValueError('Invalid value', 'Unknown image id')

            filtered_product_meta = self.apparel_meta[self.apparel_meta['gender'] == product_meta['gender'].iloc[0]]
            filtered_product_meta = filtered_product_meta[
                filtered_product_meta['articleType'] == product_meta['articleType'].iloc[0]]

        except InvalidValueError as invalid_id_except:
            print(invalid_id_except.errors)
            return pd.DataFrame()

        except Exception as exception:
            print(exception)
            return pd.DataFrame()

        finally:
            print('Done....')

        return filtered_product_meta

    def filter_by_ids(self, product_ids):
        """
        Select all products where id is in product_ids

        :param product_ids: list of product ids
        :return: dataframe of all selected products metadata
        """
        try:
            product_meta = self.apparel_meta[self.apparel_meta['id'].isin(product_ids)]
            if len(product_meta) <= 0:
                raise InvalidValueError('Invalid value', 'Unknown image id')

        except InvalidValueError as invalid_id_except:
            print(invalid_id_except.errors)
            return pd.DataFrame()

        except Exception as exception:
            print(exception)
            return pd.DataFrame()

        finally:
            print('Done....')

        return product_meta

    def filter_by_sub_categories(self, sub_categories):
        """
        Select all metadata where subCategory is in sub_categories

        :param sub_categories: list of sub_categories
        :return: dataframe of all selected products
        """
        try:
            if type(sub_categories) is not list:
                raise InvalidValueError('Invalid value', 'Sub-categories should be a list')

            filtered_product_meta = self.apparel_meta[self.apparel_meta['subCategory'].isin(sub_categories)]

        except InvalidValueError as invalid_value_exception:
            print(invalid_value_exception.errors)
            return pd.DataFrame()

        finally:
            print('Done...')

        return filtered_product_meta
