# Accuracy under Bit Flip (Section 5.4)

This directory includes the test code for obtaining Figures 13 and 14 except for ECC for ImageNet-1k dataset (ViT and Resnet). 

For ECC (Error-Correcting Code), when an NVIDIA 4090 graphics card encounters a 2-bit error, it results in a GPU reset that terminates all running programs. 
Thus, it will drop to -100% when encountering for 2 bit flips.
Consequently, ECC is designed to handle only a 1-bit error across all scenarios. 

In our provided test environment, `root_dir` has been set to ImageNet-1k dataset on the provided device. The output numbers in accuracy are all the numbers of correct test points. 
There are 50000 points for each test.


## Fast execution
To quickly demonstrate the results, we have provided a "5-minute" fast execution mode that tests two specific scenarios: one without any bit flips and the other with 4,096 bit flips.

First, you need to enter the directory. In our given code, you should `cd` into `test/Accuracy` to continue the tests.
```bash
conda activate base_environment # if your environment with base pytorch is not `base_environment`, change to your environment name.
export BASE_PYTHON=$(which python)
conda activate save_environment # if your environment with save-modified pytorch is not `base_environment`, change to your environment name.
export SAVE_PYTHON=$(which python)
rm -rf *.csv # make sure no other csv file is in the folder
make virtFast
make phyFast
```

The results will be printed in `stdout`. One possible expected output is shown below:

```bash
Cnt     vit_direct.csv          vit_rednet.csv          vit_save.csv            resnet_direct.csv       resnet_rednet.csv       resnet_save.csv     
0       40533                   40533                   40533                   24625                   24625                   24625               
4095    51                      50                      40501                   50                      50                      24618 
```

> Explanation of example output: The first three columns show the number of correct test points for the model under vit. There is a significant decrease in the number of correct test points for both "no protection" and the rednet method, indicating that the model's capability is compromised. With the save protection method, the number of correct test points remains nearly unchanged, suggesting that the model's capability is unaffected even when faced with thousands of bit flips. The last three columns represent the effectiveness of resnet, and the data is read in a similar manner.

 
**Claim:** The expected result is that the accuracy with 4,096 bits flipped is significantly lower than without any bit flips. SAVE can ensure accuracy is maintained.

## Full execution
For full execution, each test is estimated to require 307 GPU hours.

In our given code, you should `cd` into `test/Accuracy` to continue the tests.
```bash
conda activate base_environment # if your environment with base pytorch is not `base_environment`, change to your environment name.
export BASE_PYTHON=$(which python)
conda activate save_environment # if your environment with save-modified pytorch is not `base_environment`, change to your environment name.
export SAVE_PYTHON=$(which python)
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

> Explanation of example output: Similar to fast execution, except that you can find the detailed degrading process for different count of bits.


**Claim:** The expected outcome is a noticeable decrease in accuracy when up to 64 bits are flipped. This degradation in performance is influenced by different memory placement on the GPU, which may not exactly match the decline points described in the paper. 

## Q&A
1. What is the difference between virt and phy?
Answer: According to the trace provided in the RedNet paper, real-world bit flips can be categorized as either contiguous in physical space (similar to being in the same row or column) or contiguous in virtual address space. In the former case, the flips appear as bit flips that are very far apart in the virtual address space. To distinguish between these two scenarios, we conducted separate tests.
