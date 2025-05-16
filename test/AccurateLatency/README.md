# AccurateLatency for end-to-end performance (Section 5.3)

The directory includes the test code for the two tests referenced in Figure 12. 

To simplify the overhead of switching environments, we implemented RedNet and DrDNA in Python, which differs from what we described in the paper (where our baselines were implemented in a PyTorch environment). 
As a result, the outputs may differ from the data in the paper, but the performance is still comparable.



## Execution
To execute this test, you can use the command `make all` under this folder `test/AccurateLatency`. 

The results will be printed in `stdout`. One possible expected output is shown below:

```bash
python scripts/conclude.py vit_drdna.csv vit_rednet.csv vit_save.csv
Base    vit_drdna.csv           vit_rednet.csv          vit_save.csv        
1       3.1568627450980393      3.4239130434782608      1.08955223880597014 
python scripts/conclude.py resnet_drdna.csv resnet_rednet.csv resnet_save.csv
Base    resnet_drdna.csv        resnet_rednet.csv       resnet_save.csv     
1       3.2777777777777777      2.9482758620689653      1.0666666666666666 
```

**Claim:** The expected outcome is that the relative execution time of SAVE is better than both DrDNA and RedNet.

## Q&A
### 1. How should I interpret the output giving a smaller number (less than 1) when testing save compared to the base?
This is because our test takes the average of the top 95% of the time across all test points to calculate the time (to avoid instability caused by factors such as frequency in the initial cycles of the GPU). During the test, it's possible that more unstable times appeared in the initial cycles, as shown in the following output example:

```
[+] Current size: 256, elapsed time: 629.3094482421875ms
[+] Current size: 256, elapsed time: 496.74530029296875ms
[+] Current size: 256, elapsed time: 530.3079833984375ms
[+] Current size: 256, elapsed time: 493.9627380371094ms
[+] Current size: 256, elapsed time: 536.3916625976562ms
[+] Current size: 256, elapsed time: 486.76214599609375ms
[+] Current size: 256, elapsed time: 538.4450073242188ms
[+] Current size: 256, elapsed time: 486.4649353027344ms
[+] Current size: 256, elapsed time: 539.446044921875ms
[+] Current size: 256, elapsed time: 488.7890319824219ms
[+] Current size: 256, elapsed time: 539.5068969726562ms
[+] Current size: 256, elapsed time: 538.5414428710938ms
[+] Current size: 256, elapsed time: 538.6681518554688ms
[+] Current size: 256, elapsed time: 536.5517578125ms
[+] Current size: 256, elapsed time: 535.073486328125ms
[+] Current size: 256, elapsed time: 533.783935546875ms
[+] Current size: 256, elapsed time: 537.0277099609375ms
[+] Current size: 256, elapsed time: 325.0585632324219ms
[+] Current size: 256, elapsed time: 288.6312561035156ms
[+] Current size: 256, elapsed time: 288.9085388183594ms
[+] Current size: 256, elapsed time: 289.0149841308594ms
[+] Current size: 256, elapsed time: 288.96807861328125ms
[+] Current size: 256, elapsed time: 289.0125732421875ms
[+] Current size: 256, elapsed time: 288.93328857421875ms
[+] Current size: 256, elapsed time: 288.99261474609375ms
[+] Current size: 256, elapsed time: 288.94818115234375ms
[+] Current size: 256, elapsed time: 289.0245056152344ms
```

The instability in the time from lines 1 to 18 of the above output caused an issue with the base setting.
