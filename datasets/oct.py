import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits

CLASSES = ['Normal', 'AMD', 'CSC', 'DR', 'RVO', 'DME', 'MEM', 'MH']
# CLASSES = ['Normal', 'age-related macular degeneration', 
# 'central serous chorioretinopathy', 'diabetic retinopathy', 
# 'retinal vein occlusion', 'Diabetic macular edema', 'macular epiretinal membrane', 'macular hole']


class OCT(ImageList):

    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "O": "image_list/Optovue_OCTA500_test.txt",
                "S": "image_list/Spectralis_Rasti_test.txt",
                "V": "image_list/Velite_WR_test.txt",
                "B": "image_list/Bioptigen_test.txt",
            }
        elif split == 'train':
            self.image_list = {
                "O": "image_list/Optovue_OCTA500_train.txt",
                "S": "image_list/Spectralis_Rasti_train.txt",
                "V": "image_list/Velite_WR_train.txt",
                "B": "image_list/Bioptigen_train.txt",
            }
        elif split == 'val':
            self.image_list = {
                "O": "image_list/Optovue_OCTA500_train.txt",
                "S": "image_list/Spectralis_Rasti_train.txt",
                "V": "image_list/Velite_WR_train.txt",
                "B": "image_list/Bioptigen_train.txt",
            }

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        class_names = [CLASSES[i] for i in filter_class]

        super(OCT, self).__init__(root, num_classes=len(filter_class), class_names=class_names, data_list_file=data_list_file,
                                       filter_class=filter_class, **kwargs)


