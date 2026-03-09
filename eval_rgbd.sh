# test VOT22RGBD
cd VOT22RGBD_workspace
vot evaluate --workspace ./ DMTrack
vot analysis --nocache --name DMTrack

# test DepthTrack
cd Depthtrack_workspace
vot evaluate --workspace ./ DMTrack
vot analysis --name DMTrack
vot report --workspace ./ --format html
