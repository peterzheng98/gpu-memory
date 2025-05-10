# AccurateLatency for end-to-end performance (Section 5.3)

The directory includes the test code for the two tests referenced in Figure 12. 

To simplify the overhead of switching environments, we implemented RedNet and DrDNA in Python, which differs from what we described in the paper (where our baselines were implemented in a PyTorch environment). 
As a result, the outputs may differ from the data in the paper, but the performance is still comparable.



## Execution
To execute this test, you can use the command `make all`. 

The results will be printed in `stdout`. One possible expected ooutput is shown below:

```bash
python scripts/conclude.py vit_drdna.csv vit_rednet.csv vit_save.csv
Base    vit_drdna.csv           vit_rednet.csv          vit_save.csv        
1       3.1568627450980393      3.4239130434782608      1.08955223880597014 
python scripts/conclude.py resnet_drdna.csv resnet_rednet.csv resnet_save.csv
Base    resnet_drdna.csv        resnet_rednet.csv       resnet_save.csv     
1       3.2777777777777777      2.9482758620689653      1.0666666666666666 
```

**Claim:** The expected outcome is that the relative execution time of SAVE is better than both DrDNA and RedNet.