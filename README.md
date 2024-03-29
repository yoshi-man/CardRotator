# Card Rotator - Generating 3D Rotation Animation GIFs

This package helps users generate 3D rotating animation cards based on image pairs of front and back. This is similar to what 3dgifmaker (https://www.3dgifmaker.com/) can do, but couldn't really generate them in bulk. This module provides an alternative solution with bulk-generating capabilities.


## Installation

Run the following to install the python dependencies:

```
pip install opencv-python numpy Pillow
```

To install this package via pip, run:
```
pip install card-rotator
```

## Usage

### Basic Usage
```Python
from CardRotator import CardRotator

rotator = CardRotator(input_folder)

rotator.run(output_path)
```

## Parameters
To vary parameters of the animations, `CardRotator` class has arguments shown below:

```Python
rotator = CardRotator(input_folder: str, frames: int = 240, speed: int = 60, 
                      buffer_px: int = 100, zoom_factor: int = 50, verbose: bool = True)
```

| Name           |   Description                                                       | Default setting  |
|  -------  | ----- |   ------   |   
|   input_folder      | absolute path with image file pairs, where each image pair should be formatted exactly as `[file_name]_front.jpg` and `[file_name]_back.jpg` for it to be successfully detected                                       | n/a     |                                           
|   frames     |   number of frames, controls the smoothness                          | 240 |
|   speed  |   speed of rotation, play around with this number                           | 60 |
|   buffer_px      |   the black frame to extend the image by in pixels                 | 100 | 
|   zoom_factor    |  affects the field of view, play around with this                     | 50 | 
|   verbose    |  if you'd like to have printed statements                   | True | 

```Python
rotator.run(output_path)
```
| Name           |   Description                                                       | Default setting  |
|  -------  | ----- |   ------   |   
|   output_path      | absolute path of where the output GIFs should be located                                        | n/a     | 


## Demo

### Examples
![1_front](https://github.com/yoshi-man/CardRotator/blob/main/examples/1_front.jpg)
![1_back](https://github.com/yoshi-man/CardRotator/blob/main/examples/1_back.jpg)

![gif](https://github.com/yoshi-man/CardRotator/blob/main/examples/1.gif)

