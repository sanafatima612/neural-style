# Convolutional Neural Network (CNN) Implementation of Artistic Style Transfer

This project is a PyTorch implementation of the research paper “A Neural Algorithm of Artistic Style” by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The paper introduces a novel algorithm that utilizes convolutional neural networks (CNNs) to blend the content of one image with the artistic style of another.

## Overview
The algorithm effectively separates and recombines content and style from two distinct images. This approach allows for the creation of visually stunning artwork by transferring the stylistic features of famous paintings onto photographs, preserving the original content.



## Medium Article
For a deeper understanding of this project and insights from the associated research paper, check out my article on Medium: [Exploring Artistic Style Transfer with CNNs](<link_to_article>).

## Requirements
- Python 3.x
- PyTorch

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/sanafatima612/Artistic_neural_style_transfer/


<div align="center">
 <img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night_google.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/hoovertowernight.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/starry_stanford_bigger.png" width="710px">
</div>

Applying the style of different images to the same content image gives interesting results.
Here we reproduce Figure 2 from the paper, which renders a photograph of the Tubingen in Germany in a
variety of styles:

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/tubingen.jpg" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_shipwreck.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_seated_nude.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_composition_vii.png" height="250px">
</div>

Here are the results of applying the style of various pieces of artwork to this photograph of the
golden gate bridge:


<div align="center"
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/golden_gate.jpg" height="200px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/frida_kahlo.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_kahlo.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/escher_sphere.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_escher.png" height="160px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/woman-with-hat-matisse.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_matisse.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/the_scream.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_scream.png" height="160px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night_crop.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/seated-nude.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_seated.png" height="160px">
</div>

### Content / Style Tradeoff

The algorithm allows the user to trade-off the relative weight of the style and content reconstruction terms,
as shown in this example where we port the style of [Picasso's 1907 self-portrait](http://www.wikiart.org/en/pablo-picasso/self-portrait-1907) onto Brad Pitt:

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/picasso_selfport1907.jpg" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/brad_pitt.jpg" height="220px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_5_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_1_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_01_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_0025_style_10.png" height="220px">
</div>

### Style Scale

By resizing the style image before extracting style features, we can control the types of artistic
features that are transfered from the style image; you can control this behavior with the `-style_scale` flag.
Below we see three examples of rendering the Golden Gate Bridge in the style of The Starry Night.
From left to right, `-style_scale` is 2.0, 1.0, and 0.5.

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale2.png" height=175px>
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale1.png" height=175px>
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale05.png" height=175px>
</div>

### Multiple Style Images
You can use more than one style image to blend multiple artistic styles.

Clockwise from upper left: "The Starry Night" + "The Scream", "The Scream" + "Composition VII",
"Seated Nude" + "Composition VII", and "Seated Nude" + "The Starry Night"

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry_scream.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream_composition_vii.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry_seated.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_seated_nude_composition_vii.png" height="250px">
</div>


### Style Interpolation
When using multiple style images, you can control the degree to which they are blended:

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_3_7.png" height="175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_5_5.png" height="175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_7_3.png" height="175px">
</div>


### Transfer style but not color
If you add the flag `-original_colors 1` then the output image will retain the colors of the original image;
this is similar to [the recent blog post by deepart.io](http://blog.deepart.io/2016/06/04/color-independent-style-transfer/).

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_composition_vii.png" height="185px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/original_color/tubingen_starry.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/original_color/tubingen_scream.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/original_color/tubingen_composition_vii.png" height="185px">
</div>

## Setup:

Dependencies:
* [torch7](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)

Optional dependencies:
* For CUDA backend:
  * CUDA 6.5+
  * [cunn](https://github.com/torch/cunn)
* For cuDNN backend:
  * [cudnn.torch](https://github.com/soumith/cudnn.torch)
* For OpenCL backend:
  * [cltorch](https://github.com/hughperkins/cltorch)
  * [clnn](https://github.com/hughperkins/clnn)

After installing dependencies, you'll need to run the following script to download the VGG model:
```
sh models/download_models.sh
```
This will download the original [VGG-19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).
Leon Gatys has graciously provided the modified version of the VGG-19 model that was used in their paper;
this will also be downloaded. By default the original VGG-19 model is used.

If you have a smaller memory GPU then using NIN Imagenet model will be better and gives slightly worse yet comparable results. You can get the details on the model from [BVLC Caffe ModelZoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and can download the files from [NIN-Imagenet Download Link](https://drive.google.com/folderview?id=0B0IedYUunOQINEFtUi1QNWVhVVU&usp=drive_web)

You can find detailed installation instructions for Ubuntu in the [installation guide](INSTALL.md).


## Implementation details
Images are initialized with white noise and optimized using L-BFGS.

We perform style reconstructions using the `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1` layers
and content reconstructions using the `conv4_2` layer. As in the paper, the five style reconstruction losses have
equal weights.

