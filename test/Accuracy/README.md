# Accuracy under Bit Flip (Section 5.4)

This directory includes the test code for obtaining Figures 13 and 14 except for ECC for ImageNet-1k dataset (ViT and Resnet). 

For ECC (Error-Correcting Code), when an NVIDIA 4090 graphics card encounters a 2-bit error, it results in a GPU reset that terminates all running programs. 
Thus, it will drop to -100% when encountering for 2 bit flips.
Consequently, ECC is designed to handle only a 1-bit error across all scenarios. 

In our provided test environment, `root_dir` has been set to ImageNet-1k dataset on the provided device.


## Fast execution
To quickly demonstrate the results, we have provided a "5-minute" fast execution mode that tests two specific scenarios: one without any bit flips and the other with 4,096 bit flips.
```bash
export BASE_PYTHON=$(which python)
export SAVE_PYTHON=$(path/to/save/installed/torch)
make virtFast
make phyFast
```

The results will be printed in `stdout`. One possible expected output is shown below:

```bash
Cnt     vit_direct.csv          vit_rednet.csv          vit_save.csv            resnet_direct.csv       resnet_rednet.csv       resnet_save.csv     
0       40533                   40533                   40533                   24625                   24625                   24625               
4095    51                      50                      40501                   50                      50                      24618 
```

**Claim:** The expected result is that the accuracy with 4,096 bits flipped is significantly lower than without any bit flips. SAVE can ensure accuracy is maintained.

## Full execution
For full execution, each test is estimated to require 307 GPU hours.
```bash
export BASE_PYTHON=$(which python)
export SAVE_PYTHON=$(path/to/save/installed/torch)
make virt
make phy
```

The results will be printed in `stdout`. One possible expected output is shown below:

```bash
Cnt     vit_direct.csv          vit_rednet.csv          vit_save.csv            resnet_direct.csv       resnet_rednet.csv       resnet_save.csv     
0       40533                   40533                   40533                   24625                   24625                   24625               
(Omit all other lines)
4095    51                      50                      40501                   50                      50                      24618 
```

**Claim:** The expected outcome is a noticeable decrease in accuracy when up to 64 bits are flipped. This degradation in performance is influenced by different memory placement on the GPU, which may not exactly match the decline points described in the paper. 
