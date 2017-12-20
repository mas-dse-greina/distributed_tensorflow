# distributed_tensorflow
Sample Implementation of Distributed TensorFlow

This should work for TF >= 1.4.

It implements either synchronous and asynchronous parameter updates.

The "data" is just a simple line with random noise. You can change the slope and intercept of the line by editing those variables at the top of the [test_dist.py](https://github.com/mas-dse-greina/distributed_tensorflow/blob/master/test_dist.py) file.

```
slope = 5
intercept = 13
```

To run the script:
1. First you'll need to install a TensorFlow conda environment on all of your machines.
+ Install [Miniconda](https://conda.io/miniconda.html)
+ `conda create -n tf -c intel python=2 pip numpy`
+ `source activate tf`
+ `pip install https://anaconda.org/intel/tensorflow/1.4.0/download/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl`
2. Edit [test_dist.py](https://github.com/mas-dse-greina/distributed_tensorflow/blob/master/test_dist.py) to specify which machines will be parameter servers and which will be worker nodes.
```
ps_hosts = ["22.100.38.245"]
ps_ports = ["2222"]
worker_hosts = ["22.100.38.193","22.100.38.183"] #,"22.100.38.185","22.100.38.187"]
worker_ports = ["2222", "2222"] #, "2222", "2222"]
```
3. Run `python test_dist.py --is_sync=1` on every machine. You can specify the IP address (--ip) if you want to use OPA instead of the usual IPv4/6 address.

Here's a sample output from one of the worker nodes:
```
(tf) [tony@tensor816 distributed_tensorflow]$ python test_dist.py --ip=22.100.38.193 --is_sync=1
Distributed TensorFlow training
Parameter server nodes are: ['22.100.38.245:2222']
Worker nodes are ['22.100.38.193:2222', '22.100.38.183:2222']
E1220 11:01:55.676897759   27162 ev_epoll1_linux.c:1051]     grpc epoll fd: 3
[step: 0 of 50,000] Predicted Slope: 2.674 (True slope = 5), Predicted Intercept: 2.156 (True intercept = 13), loss: 82.6484
[step: 0 of 50,000] Predicted Slope: 2.690 (True slope = 5), Predicted Intercept: 2.159 (True intercept = 13), loss: 643.8683
[step: 1,000 of 50,000] Predicted Slope: 4.994 (True slope = 5), Predicted Intercept: 4.088 (True intercept = 13), loss: 80.9343
[step: 2,000 of 50,000] Predicted Slope: 5.063 (True slope = 5), Predicted Intercept: 5.704 (True intercept = 13), loss: 58.0817
[step: 3,000 of 50,000] Predicted Slope: 5.057 (True slope = 5), Predicted Intercept: 7.026 (True intercept = 13), loss: 41.1207
[step: 4,000 of 50,000] Predicted Slope: 5.019 (True slope = 5), Predicted Intercept: 8.106 (True intercept = 13), loss: 23.3233
[step: 5,000 of 50,000] Predicted Slope: 4.978 (True slope = 5), Predicted Intercept: 8.992 (True intercept = 13), loss: 18.7609
[step: 6,000 of 50,000] Predicted Slope: 4.994 (True slope = 5), Predicted Intercept: 9.721 (True intercept = 13), loss: 9.4690
[step: 7,000 of 50,000] Predicted Slope: 4.984 (True slope = 5), Predicted Intercept: 10.317 (True intercept = 13), loss: 8.5107
[step: 8,000 of 50,000] Predicted Slope: 4.986 (True slope = 5), Predicted Intercept: 10.806 (True intercept = 13), loss: 3.7054
[step: 9,000 of 50,000] Predicted Slope: 4.988 (True slope = 5), Predicted Intercept: 11.205 (True intercept = 13), loss: 1.5718
[step: 10,000 of 50,000] Predicted Slope: 4.993 (True slope = 5), Predicted Intercept: 11.528 (True intercept = 13), loss: 2.7668
[step: 11,000 of 50,000] Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 11.797 (True intercept = 13), loss: 1.1680
[step: 12,000 of 50,000] Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 12.016 (True intercept = 13), loss: 0.9956
[step: 13,000 of 50,000] Predicted Slope: 4.995 (True slope = 5), Predicted Intercept: 12.193 (True intercept = 13), loss: 0.0016
[step: 14,000 of 50,000] Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.337 (True intercept = 13), loss: 0.1792
[step: 15,000 of 50,000] Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.457 (True intercept = 13), loss: 0.0713
[step: 16,000 of 50,000] Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.556 (True intercept = 13), loss: 0.1906
[step: 17,000 of 50,000] Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.637 (True intercept = 13), loss: 0.8171
[step: 18,000 of 50,000] Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 12.704 (True intercept = 13), loss: 0.1772
[step: 19,000 of 50,000] Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.756 (True intercept = 13), loss: 0.0997
[step: 20,000 of 50,000] Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.800 (True intercept = 13), loss: 0.0682
[step: 21,000 of 50,000] Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 12.835 (True intercept = 13), loss: 0.1465
[step: 22,000 of 50,000] Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 12.864 (True intercept = 13), loss: 0.0883
[step: 23,000 of 50,000] Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 12.889 (True intercept = 13), loss: 0.0335
[step: 24,000 of 50,000] Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 12.911 (True intercept = 13), loss: 0.1060
[step: 25,000 of 50,000] Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 12.927 (True intercept = 13), loss: 0.0068
[step: 26,000 of 50,000] Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 12.938 (True intercept = 13), loss: 0.0042
[step: 27,000 of 50,000] Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 12.948 (True intercept = 13), loss: 0.5640
[step: 28,000 of 50,000] Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 12.955 (True intercept = 13), loss: 0.2997
[step: 29,000 of 50,000] Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 12.963 (True intercept = 13), loss: 0.1205
[step: 30,000 of 50,000] Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.970 (True intercept = 13), loss: 0.4628
[step: 31,000 of 50,000] Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 12.975 (True intercept = 13), loss: 0.0000
[step: 32,000 of 50,000] Predicted Slope: 4.997 (True slope = 5), Predicted Intercept: 12.980 (True intercept = 13), loss: 0.0186
[step: 33,000 of 50,000] Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 12.985 (True intercept = 13), loss: 0.0345
[step: 34,000 of 50,000] Predicted Slope: 4.996 (True slope = 5), Predicted Intercept: 12.987 (True intercept = 13), loss: 0.0264
[step: 35,000 of 50,000] Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.991 (True intercept = 13), loss: 0.2470
[step: 36,000 of 50,000] Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.993 (True intercept = 13), loss: 0.1550
[step: 37,000 of 50,000] Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 12.994 (True intercept = 13), loss: 0.0397
[step: 38,000 of 50,000] Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.993 (True intercept = 13), loss: 0.0038
[step: 39,000 of 50,000] Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 12.993 (True intercept = 13), loss: 0.7138
[step: 40,000 of 50,000] Predicted Slope: 5.004 (True slope = 5), Predicted Intercept: 12.995 (True intercept = 13), loss: 0.0000
[step: 41,000 of 50,000] Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.996 (True intercept = 13), loss: 0.0693
[step: 42,000 of 50,000] Predicted Slope: 4.996 (True slope = 5), Predicted Intercept: 12.998 (True intercept = 13), loss: 0.0001
[step: 43,000 of 50,000] Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.998 (True intercept = 13), loss: 0.4081
[step: 44,000 of 50,000] Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 12.999 (True intercept = 13), loss: 0.0072
[step: 45,000 of 50,000] Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 12.999 (True intercept = 13), loss: 0.0679
[step: 46,000 of 50,000] Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 12.999 (True intercept = 13), loss: 0.2819
[step: 47,000 of 50,000] Predicted Slope: 4.997 (True slope = 5), Predicted Intercept: 12.998 (True intercept = 13), loss: 0.1404
[step: 48,000 of 50,000] Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 12.999 (True intercept = 13), loss: 0.3086
[step: 49,000 of 50,000] Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 13.000 (True intercept = 13), loss: 0.0982
[step: 50,000 of 50,000] Predicted Slope: 4.996 (True slope = 5), Predicted Intercept: 13.000 (True intercept = 13), loss: 0.1591
```
