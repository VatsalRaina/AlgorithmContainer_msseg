# MSSEG Algorithm Container Example

This repository is an example of the algorithm container with baseline models that can be submitted to the MS Lesion Segmentation track of the Shifts Challenge 2022 hosted on Grand Challenge.

## Quick instructions

This section outlines how to prepare the baseline submission for Grand Challenge. Please ensure you have the authority to build Docker images on your local system and then follow these instructions which show you how to BUILD, TEST and then EXPORT the container:
1. Clone this repository (or a forked version of it) onto your local system.
2. Navigate to `./Baseline/`
3. Ensure you can BUILD the container by running `./build.sh`.
4. TEST the container by running `./test.sh` and confirm you see the message *Tests successfully passed...*.
5. EXPORT the container by running `./export.sh`.
6. You should see a new file created called `Baseline.tar.gz` - submit this file on the submission page of the MSSEG track of Grand Challenge.

## Detailed description


## Modifying for your own submission
