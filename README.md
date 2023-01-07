[MobileNetV3](https://arxiv.org/abs/1905.02244) implementation using PyTorch

#### Steps

* Configure `imagenet` path by changing `data_dir` in `main.py`
* `python main.py --benchmark` for model information
* `bash ./main.sh $ --train` for training model, `$` is number of GPUs
* `python main.py --test` for testing

#### Note

* MobileNetV3-Large achieved  75.6 % top-1 and  92.4 % top-5 after 450 epochs

```
Number of parameters: 5470832
Time per operator type:
        162.927 ms.    88.5142%. Conv
        7.23122 ms.    3.92853%. Add
        4.38072 ms.    2.37993%. Div
        3.71718 ms.    2.01945%. Mul
        2.62309 ms.    1.42506%. Relu
        1.49218 ms.   0.810663%. Clip
        1.22271 ms.   0.664268%. FC
       0.469844 ms.   0.255254%. AveragePool
     0.00485016 ms. 0.00263497%. Flatten
        184.069 ms in Total
FLOP per operator type:
       0.428162 GFLOP.     97.588%. Conv
     0.00501988 GFLOP.    1.14415%. FC
     0.00210867 GFLOP.   0.480615%. Mul
     0.00193868 GFLOP.    0.44187%. Add
     0.00151532 GFLOP.   0.345376%. Div
              0 GFLOP.          0%. Relu
       0.438744 GFLOP in Total
Feature Memory Read per operator type:
        29.8725 MB.    37.7212%. Conv
         14.496 MB.    18.3046%. Mul
        10.0533 MB.    12.6947%. FC
        9.44828 MB.    11.9307%. Add
        9.26157 MB.    11.6949%. Relu
         6.0614 MB.    7.65395%. Div
         79.193 MB in Total
Feature Memory Written per operator type:
        17.6196 MB.    35.8551%. Conv
        9.26157 MB.     18.847%. Relu
        8.43469 MB.    17.1643%. Mul
        7.75472 MB.    15.7806%. Add
        6.06128 MB.    12.3345%. Div
        0.00912 MB.  0.0185589%. FC
        49.1409 MB in Total
Parameter Memory per operator type:
        11.7699 MB.    53.9552%. Conv
        10.0443 MB.    46.0449%. FC
              0 MB.          0%. Add
              0 MB.          0%. Div
              0 MB.          0%. Mul
              0 MB.          0%. Relu
        21.8142 MB in Total
```

