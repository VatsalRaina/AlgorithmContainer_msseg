
import SimpleITK
import numpy as np
import torch
from scipy import ndimage
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from uncertainty import ensemble_uncertainties_classification
from pathlib import Path

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        # print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a 
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2
    

class Baseline(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self._segmentation_output_path = Path("/output/segmentation/")
        self._uncertainty_output_path = Path("/output/uncertainty/")

        self.device = get_default_device()

        K = 3
        models = []
        for i in range(K):
            models.append(UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=0).to(self.device)
            )

        for i, model in enumerate(models):
            model.load_state_dict(torch.load('./model'+str(i+1)+'.pth', map_location=self.device))
            model.eval()

        self.models = models
        self.act = torch.nn.Softmax(dim=1)
        self.th = 0.35
        self.roi_size = (96, 96, 96)
        self.sw_batch_size = 4


    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_map, uncertainty_map = self.predict(input_image=input_image)

        # Write resulting segmentation to output location
        segmentation_path = self._segmentation_output_path / input_image_file_path.name
        if not self._segmentation_output_path.exists():
            self._segmentation_output_path.mkdir()
        SimpleITK.WriteImage(segmented_map, str(segmentation_path), True)

        # Write resulting uncertainty map to output location
        uncertainty_path = self._uncertainty_output_path / input_image_file_path.name
        if not self._uncertainty_output_path.exists():
            self._uncertainty_output_path.mkdir()
        SimpleITK.WriteImage(uncertainty_map, str(uncertainty_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "segmentation": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "uncertainty": [
                dict(type="metaio_image", filename=uncertainty_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }


    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:

        image = SimpleITK.GetArrayFromImage(input_image)
        image = np.array(image)

        with torch.no_grad():

            image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image).to(self.device), axis=0), axis=0)

            all_outputs = []
            for model in self.models:
                outputs = sliding_window_inference(image, self.roi_size, self.sw_batch_size, model, mode='gaussian')
                outputs = self.act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
        all_outputs = np.asarray(all_outputs)

        seg = np.mean(all_outputs, axis=0)
        seg[seg>self.th]=1
        seg[seg<=self.th]=0
        seg = np.squeeze(seg)
        seg = remove_connected_components(seg)

        uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )
        unc_rmi = uncs["reverse_mutual_information"]

        out_seg = SimpleITK.GetImageFromArray(seg)
        out_unc = SimpleITK.GetImageFromArray(unc_rmi)
        return out_seg, out_unc





if __name__ == "__main__":
    Baseline().process()
