# Extension of mv3dpose

## Abstract
This project addresses the problem of markerless motion capture of multiple persons in different scenes.
The goal is to estimate the 3D poses of each person in multi-view videos depicting multiple persons performing various tasks and interacting with each other.
To solve this problem, we created a pipeline that generates sequences of 3D LISST body parameters that correspond to individual persons.
In our pipeline, we used OpenPose to recover 2D poses from individual frames and match them across views and over time with a greedy matching algorithm available in [mv3dpose github repository](https://github.com/jutanke/mv3dpose), explained deeply in the paper [Multiple Person Multiple View 3D Pose Estimation](http://pages.iai.uni-bonn.de/gall_juergen/download/jgall_mvpose_gcpr19.pdf).
We included a track stitching algorithm to reduce the number of falsely detected tracks in *mv3dpose* and used motion priors to achieve more accurate and smooth tracking even with many persons in the scene.

## Install

Clone this repository with its submodules as follows:
```bash
git clone --recursive git@github.com:GemCat/lisst_mv3dpose.git
```

## Usage

Your dataset must reside in a pre-defined folder structure:

* dataset
  * dataset.json
  * cameras
    * camera00_calibration.json
    * camera01_calibration.json
    * ...
  * openpose_keypoints
    * xxxxx_00_json
      * xxxxx_00_000000000000.json
      * xxxxx_00_000000000001.json
      * ...
    * xxxxx_01_json
      * xxxxx_01_000000000000.json
      * xxxxx_01_000000000001.json
      * ...
    * ...
  * img
    * camera00
      * frame0_00.jpg
      * frame1_00.jpg
      * ...
      * frame100_00.jpg
      * ...
    * camera01
      * frame0_01.jpg
      * frame1_01.jpg
      * ...
      * frame100_01.jpg
      * ...
    * ...

The file names per frame utilize the following schema: 
```python
"frame%09d.json"
```

The camera json files follow the structure below: 
```javascript
{
  "name": str(name),
  "type": str(camera_type),
  "resolution": [1 x 2],
  "panel": int(panel_nr),
  "node": int(node_nr),
  "K" : [ 3 x 3 ], /* intrinsic paramters */
  "discCoef": [ 1 x 5 ], /* distortion coefficient */
  "R": [ 3 x 3 ], /* rotation matrix */
  "t": [ 3 x 1 ], /* translation vector */
}
```

The system expects a camera for each view at each point in time. If your dataset uses fixed cameras you will need to simply repeat them for all frames.

The _dataset.json_ file contains general information for the model and the optional parameters:
```javascript
{
  "n_cameras": int(#cameras), /* number of cameras */
  "scale_to_mm": 1, /* scales the calibration to mm */
  "output_dir": str(output_directory),
  "keypoints_dir": str(2D_keypoints_directory),
  "camera_calib_dir": str(camera_calibrations_directory),
  "vis_dir": str(visualization_directory),
  "img_dir": str(frame_images_directory),
  "img_file_type": str(image_type),
  "json_poses_prefix": str(json_poses_prefix),

  "valid_frames": int(#valid_frames), /*if frames do not start at 0 and/or are not continious you can set a list of frames here*/
  "vis_frames": list(frames_ids),
  "plot3d": bool(plotting_3D),

  "epi_threshold": int(epipolar_distance_thr), /*epipolar line distance threshold in PIXEL*/
  "max_distance_between_tracks": int(dist_between_tracks), /*maximal distance in [mm] between tracks so that they can be associated*/
  "min_track_length": int(min_track_length), /*drop any track which is shorter than _min_track_length_ frames*/
  "merge_distance": double(merge_dist),
  "last_seen_delay": int(delay), /*allow to skip _last_seen_delay_ frames for connecting a lost track*/
  "smoothing_sigma": double(sigma), /*sigma value for Gaussian smoothing of tracks*/
  "smoothing_interpolation_range": int(max_interpolation), /*define how far fill-ins should be reaching*/
  "do_smoothing": bool(smoothing) /*should smoothing be done at all*/
}
```

The variable __scale_to_mm__ is needed as we operate in [mm] but calibrations might be in other metrics. For example, when the calibration is done in meters, _scale_to_mm_ must be set to _1000_.


### Run the system

Change the path to your dataset in bash script files and run:
```bash
./mvpose.sh
```

The resulting tracks will be in your dataset folder under __output_dir__, each track represents a single person. 
The files are organised as follows:
```javascript
{
  "J": int(joint number), /* number of joints */
  "frames": [int, int], /* ordered list of the frames where this track is residing */
  "poses": [ n_frames x J x 3 ] /* 3D poses, 3d location OR None, if joint is missing */
}
```
