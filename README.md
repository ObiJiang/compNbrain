# compNbrain
Binary classification for breast cancer detection using only local hebbian rule

Import stuff: use tanh as activation function to have negative responese

Import stuff: adaptive bias term

# How to Run
```
python model_cancer.py
```

# Sample Result
## Hebb (with LTD)
```
Tranning Error at Epochs 0:0.4375
Tranning Error at Epochs 1:0.11500000208616257
Tranning Error at Epochs 2:0.10499999672174454
Tranning Error at Epochs 3:0.0949999988079071
Tranning Error at Epochs 4:0.0949999988079071
Tranning Error at Epochs 5:0.09000000357627869
Tranning Error at Epochs 6:0.09000000357627869
Tranning Error at Epochs 7:0.0925000011920929
Tranning Error at Epochs 8:0.08749999850988388
Tranning Error at Epochs 9:0.08749999850988388
Tranning Error at Epochs 10:0.08749999850988388
...
Test Error Rate:0.11999999731779099
```

## BDM
```
Tranning Error at Epochs 0:0.47749996185302734
Tranning Error at Epochs 1:0.7649999856948853
Tranning Error at Epochs 2:0.7649999856948853
Tranning Error at Epochs 3:0.5049999952316284
Tranning Error at Epochs 4:0.30000001192092896
Tranning Error at Epochs 5:0.29750001430511475
Tranning Error at Epochs 6:0.10999999940395355
Tranning Error at Epochs 7:0.11500000208616257
Tranning Error at Epochs 8:0.0949999988079071
Tranning Error at Epochs 9:0.08749999105930328
Tranning Error at Epochs 10:0.09749999642372131
Test Error Rate:0.14000000059604645
```