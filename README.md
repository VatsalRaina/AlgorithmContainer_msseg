#  Algorithm Container Example - Multiple Sclerosis White Matter Lesion Segmentation

This repository is an example of the algorithm container with baseline models that can be submitted to the [Track 1 of Shifts Challenge](https://shifts.grand-challenge.org/medical-dataset/) for the task of segmentation of white matter Multiple Sclerosis (MS) lesions on 3D FLAIR scans. The purpose of an aglorithm container is to wrap together your models and the inference code for your models such that they can be evaluated automatically on unseen inputs in the evaluation set on Grand Challenge.

## Quick instructions

This section outlines how to prepare the baseline submission for Grand Challenge. Please ensure you have the authority to build Docker images on your local system and then follow these instructions which show you how to BUILD, TEST and then EXPORT the container:
1. Clone this repository (or a forked version of it) onto your local system.
2. Navigate to `./Baseline/`
3. Ensure you can BUILD the container by running `./build.sh`.
4. TEST the container by running `./test.sh` and confirm you see the message *Tests successfully passed...*.
5. EXPORT the container by running `./export.sh`.
6. You should see a new file created called `Baseline.tar.gz` - submit this file on the submission page of the MS Lesion Segmentation track of Grand Challenge.

## Detailed description

This section aims to explain each of the files in the `./Baseline/` directory of the repository in more detail.
* `model1.pth`, `model2.pth` and `model3.pth` are baseline models provided to you and trained following the official instructions at [Shifts mswml](https://github.com/Shifts-Project/shifts/tree/main/mswml).
* `Dockerfile` points to locations of all the models being used, ensures all requirements are in place and then calls `process.py`.
* `requirements.txt` specifies all dependencies for the models to be used at inference time; no need to specify pytorch here as it is specified in `Dockerfile`.
* `uncertainty.py` is a module containing implementations of uncertainty measures computed based on deep ensembles to be used by `process.py`.
* `process.py` specifies the inference of the models on each image and saves the predictions appropriately.
* `build.sh` checks whether your system is able to build dockers.
* `test.sh` checks if `process.py` is able to read in a sample image from `./test/` and generate the appropriate outputs (a continuous prediction of probabilities and an equivalent uncertainty map of the same size as the input image) to match the expected output `./test/expected_output.json`.
* `export.sh` wraps the container together into a single `.tar.gz` file that can then be used for submission.


## Modifying for your own submission

If you are on this section, we assume you have trained your own models and ensured they perform well locally based on the instructions at [Shifts mswml](https://github.com/Shifts-Project/shifts/tree/main/mswml). Now, this section explains how you can edit this example algorithm container to make your own algorithm container for your models and their inference.
1. Delete the existing models `model1.pth`, `model2.pth` and `model3.pth` and add your own models instead.
2. Edit `Dockerfile` to ensure the new models you have added are correctly linked based on the names you have used for them (see how the baseline models are linked in the file e.g.  `COPY --chown=algorithm:algorithm model1.pth /opt/algorithm/model1.pth`).
3. Update `requirements.txt` with any additional libraries needed for the inference of your models.
4. Now you must add your own inference code in `process.py`. You only need to edit the function `def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:` which takes in a single input image and returns a segmentation map of probabilities and an uncertainty map.
5. You can now check that your model operates correctly using the BUILD, TEST and EXPORT commands.

## Contact

If you are struggling with any of the above steps or need clarifications on how to use this repository, please contact Vatsal Raina (vr311@cam.ac.uk)
