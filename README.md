# Simple Facial Recognition

A Python-based app that uses OpenCV to perform basic facial detection
and to demonstrate C++ language binding from Python.

![app](https://i.imgur.com/qp5dQ7V.jpg)
## Setup

0. Install Xcode and Xcode's Command Line tools with `xcode-select --install`
1. Install [Homebrew](https://brew.sh/) with `/usr/bin/ruby -e "$(curl -fsSL
   https://raw.githubusercontent.com/Homebrew/install/master/install)"`

Run the following:

```
# Install prereqs for Kivy and pybind11 as per
# https://kivy.org/docs/installation/installation-osx.html and
# http://pybind11.readthedocs.io/en/master/
brew install python3 cmake pkg-config sdl2 sdl2_image sdl2_ttf sdl2_mixer gstreamer

# Install Python package requirements
pip3 install -r requirements.txt

# We have to install the latest Kivy tarball to prevent an error on
# OSX High Sierra related to compilation using Cython
pip3 install https://github.com/kivy/kivy/archive/master.zip

# Compile the C++ / pybind11 file
# Why `sudo`? Unfortunately, we need to modify our Python path to include the
# path to the compile binary, and that requires sudo perms (at least locally).
sudo python3 setup.py install
```

After that, you should be able to run the application with `python
src/camera_app.py`.

## Using the App

You should see a video from your webcam in the top portion of the application.
Beneath, you can toggle on and off a number of features including:

* Face detection
* Green channel only mode
* Display FPS

And, at the bottom, you have the option to capture a timestamped PNG of the
viewer-area's current output. It will be deposited into the directory
from which you've run the application, most likely in the same directory as this
README.

## Integrations

The "Green Channel" mode uses the slice notation offered by numpy to efficiently
remove all non-green-channel data. All arrays generated by the OpenCV
bindings are
[numpy-based](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_intro/py_intro.html#opencv-python)
arrays.

We use the
[opencv-contrib-python](https://pypi.python.org/pypi/opencv-contrib-python)
package
because it includes pre-built Python wheels,
allowing us to skip installing OpenCV from scratch and instead relying upon
pre-built binaries.

We use pybind11 to compile the helper `pybind_example.cpp` file, which we use to
calculate a set of rolling window statistics on the frames-per-second rendered
/ processed by the app.

## References

* https://github.com/pybind/python_example
* https://ep2017.europython.eu/media/conference/slides/pybind11-seamless-operability-between-c11-and-python.pdf
* https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
* https://kivy.org/docs/examples/gen__camera__main__py.html
* https://gist.github.com/ExpandOcean/de261e66949009f44ad2
