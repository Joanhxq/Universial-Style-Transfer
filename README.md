# Universal Style Transfer via Feature Transforms

This is a Pytorch implementaion of the [Universial Style Transfer via Feature Transforms](https://arxiv.org/abs/1705.08086).

Given **a content image** and **an arbitrary style image**, the program attempts to transfer the visual style characteristics extracted from the style image to the content image generating stylized output.

The core architecture is VGG19 Convolutional Autoencoder performing Whitening and Coloring Transfermation on the content and style features in the bottleneck layer.

## Installation

Need Python packages can be installed using `conda` package manager bu running

`conda env create -f environment.yaml`

## Usage

`python main.py ARGS`

Possible ARGS are:

- `-h, --help` Show this help message and exit;
- `--content CONTENT` Path of content image(or directory containing images) to be transformed;
- `--style STYLE` Path of style image(or directory containing images) to be transformed;
- `--contentSize CONTENTSIZE` Reshape the content image to have new specified maximum size (keeping aspect ratio);
- `--styleSize STYLESIZE` Reshape the style image to have new specified maximum size (keeping aspect ratio);
- `--mask MASK` Path of the binary mask image (white on black) to transfer the style pair in the corrisponding areas;
- `--synthesis SYNTHESIS` Flag to syntesize a new texture. Must provide a texture style image;
- `--stylePair STYLEPAIR` Path of two style images(separated by ",") to use in combination;
- `--gpu GPU` Flag to enables GPU to accelerated computations(default is `True`);
- `--beta BETA` Hyperparameter balancing the interpolation between the two style images in stylePair(default is `0.5`);
- `--alpha ALPHA` Hyperparameter balancing the original content features and WCT-transformed features(default is `0.2`);
- `--outDir OUTDIR` Path of the directory store stylized images(default is `outputs`);
- `--outPrefix OUTPREFIX` Name prefixd in the saved stylized images;
- `--singleLevel SINGLELEVEL` Flag to switch to single level stylization(default is `False`)

Supported image file formats are: __jpg__, __jpeg__, __png__.

### Examples:

- <u>STYLE TRANSFER:</u>

```
python main.py --content ./inputs/contents/in1.jpg --style ./inputs/styles/tiger.jpg --contentSize 512 --styleSize 512

python main.py --content ./inputs/contents --style ./inputs/styles/tiger.jpg --contentSize 512 --styleSize 512 --outPrefix allcontents

python main.py --content ./inputs/contents/in1.jpg --style ./inputs/styles --contentSize 512 --styleSize 512 --outPrefix allstyles

python main.py --content ./inputs/contents --style ./inputs/styles --contentSize 512 --styleSize 512 --outPrefix total
```

- <u>MASK:</u>

```
python main.py --content ./inputs/contents/face.jpg --stylePair ./inputs/styles/cubism.jpg,./inputs/styles/draft.jpg --mask ./inputs/masks/glasses_mask.jpg --contentSize 512 --styleSize 512 --outPrefix mask
```

- <u>SYNTHESIS:</u>

```
python main.py --stylePair ./inputs/styles/tiger.jpg,./inputs/styles/in1.jpg --synthesis True --contentSize 512 --styleSize 512
```

## Result:

<figure class="half">
​    <img src="outputs\total_in1_stylized_by_tiger_alpha_0.2.png" width="250px" height="250px"><img src="outputs\multiLevel_mask_face_stylized_by_cubism_and_draft_alpha_0.2.png" width="250px" height="250px"><img src="outputs\multiLevel_texture_iter3_stylized_by_tiger_and_in1_alpha_1.png" width="250px" height="250px">

</figure>



