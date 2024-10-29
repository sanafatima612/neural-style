# Ensure you have the ultra-high resolution scan of Starry Night from the Google Art Project.
# Download it from: https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg

# Define the style and content images.
STYLE_IMAGE="starry_night_gigapixel.jpg"
CONTENT_IMAGE="examples/inputs/hoovertowernight.jpg"

# Set the parameters for style transfer.
STYLE_WEIGHT=500    # Weight for style loss; higher values prioritize style over content.
STYLE_SCALE=1.0     # Scale factor for the style image.
IMAGE_SIZES=(256 512 1024 2048 3620)  # Different sizes for image generation.
NUM_ITERATIONS=(1 500 200 100 50)      # Number of iterations for each corresponding image size.
OUTPUT_IMAGES=("out1.png" "out2.png" "out3.png" "out4.png" "out5.png")  # Output filenames.

# Loop through the sizes, generating images at increasing resolutions.
for i in "${!IMAGE_SIZES[@]}"; do
    # Set parameters for current iteration.
    CURRENT_SIZE=${IMAGE_SIZES[$i]}
    CURRENT_OUTPUT=${OUTPUT_IMAGES[$i]}
    CURRENT_ITERATIONS=${NUM_ITERATIONS[$i]}

    # If this is the first iteration, initialize with the content image.
    if [ $i -eq 0 ]; then
        th neural_style.lua \
          -content_image $CONTENT_IMAGE \
          -style_image $STYLE_IMAGE \
          -style_scale $STYLE_SCALE \
          -print_iter 1 \
          -style_weight $STYLE_WEIGHT \
          -image_size $CURRENT_SIZE \
          -output_image $CURRENT_OUTPUT \
          -tv_weight 0 \
          -backend cudnn -cudnn_autotune
    else
        # For subsequent iterations, use the previous output as the initialization.
        th neural_style.lua \
          -content_image $CONTENT_IMAGE \
          -style_image $STYLE_IMAGE \
          -init image -init_image ${OUTPUT_IMAGES[$((i-1))]} \
          -style_scale $STYLE_SCALE \
          -print_iter 1 \
          -style_weight $STYLE_WEIGHT \
          -image_size $CURRENT_SIZE \
          -num_iterations $CURRENT_ITERATIONS \
          -output_image $CURRENT_OUTPUT \
          -tv_weight 0 \
          -backend cudnn -cudnn_autotune \
          # Additional GPU settings for larger images.
          $(if [ $CURRENT_SIZE -ge 2048 ]; then echo "-gpu 0,1"; fi)
    fi
done

# The last iteration includes options for multi-GPU usage and saves iterations every 25 steps.
th neural_style.lua \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out4.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 3620 \
  -num_iterations 50 \
  -save_iter 25 \
  -output_image out5.png \
  -tv_weight 0 \
  -lbfgs_num_correction 5 \
  -gpu 0,1,2,3 \
  -multigpu_strategy 3,6,12 \
  -backend cudnn
