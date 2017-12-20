# distributed_tensorflow
Sample Implementation of Distributed TensorFlow

This should work for TF >= 1.4.

It implements either synchronous and asynchronous parameter updates.

The "data" is just a simple line with random noise. You can change the slope and intercept of the line by editing those variables at the top of the test_dist.py file.

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
2. Edit test_dist.py to specify which machines will be parameter servers and which will be worker nodes.
```
ps_hosts = ["22.100.68.245"]
ps_ports = ["2222"]
worker_hosts = ["22.100.28.193","22.100.28.183"] #,"22.100.28.185","22.100.28.187"]
worker_ports = ["2222", "2222"] #, "2222", "2222"]
```
3. Run `python test_dist.py --issync=1` on every machine. You can specify the IP address (--ip) if you want to use OPA instead of the usual IPv4/6 address.

Here's a sample output from one of the worker nodes:
```
(tf) [tony@tensor816 distributed_tensorflow]$ python test_dist.py --ip=10.100.68.193 --issync=0
Distributed TensorFlow training
Parameter server nodes are: ['22.100.28.245:2222']
Worker nodes are ['22.100.28.193:2222', '22.100.28.183:2222']
E1219 17:21:59.704264874  284893 ev_epoll1_linux.c:1051]     grpc epoll fd: 3
(step: 5,000 of 50,000) Predicted Slope: 5.061 (True slope = 5), Predicted Intercept: 3.491 (True intercept = 13, loss: 78.1103
(step: 7,000 of 50,000) Predicted Slope: 4.988 (True slope = 5), Predicted Intercept: 4.564 (True intercept = 13, loss: 76.8142
(step: 10,000 of 50,000) Predicted Slope: 5.011 (True slope = 5), Predicted Intercept: 5.955 (True intercept = 13, loss: 54.2730
(step: 12,000 of 50,000) Predicted Slope: 4.977 (True slope = 5), Predicted Intercept: 6.749 (True intercept = 13, loss: 34.2588
(step: 16,000 of 50,000) Predicted Slope: 5.017 (True slope = 5), Predicted Intercept: 8.083 (True intercept = 13, loss: 23.1863
(step: 19,000 of 50,000) Predicted Slope: 4.964 (True slope = 5), Predicted Intercept: 8.893 (True intercept = 13, loss: 14.5385
(step: 21,000 of 50,000) Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 9.357 (True intercept = 13, loss: 16.4815
(step: 22,000 of 50,000) Predicted Slope: 5.020 (True slope = 5), Predicted Intercept: 9.569 (True intercept = 13, loss: 12.6421
(step: 23,000 of 50,000) Predicted Slope: 5.008 (True slope = 5), Predicted Intercept: 9.770 (True intercept = 13, loss: 14.0352
(step: 26,000 of 50,000) Predicted Slope: 4.989 (True slope = 5), Predicted Intercept: 10.302 (True intercept = 13, loss: 7.6165
(step: 27,000 of 50,000) Predicted Slope: 5.020 (True slope = 5), Predicted Intercept: 10.460 (True intercept = 13, loss: 5.3987
(step: 29,000 of 50,000) Predicted Slope: 5.013 (True slope = 5), Predicted Intercept: 10.745 (True intercept = 13, loss: 4.5019
(step: 30,000 of 50,000) Predicted Slope: 4.997 (True slope = 5), Predicted Intercept: 10.876 (True intercept = 13, loss: 2.3906
(step: 33,000 of 50,000) Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 11.226 (True intercept = 13, loss: 2.3224
(step: 35,000 of 50,000) Predicted Slope: 4.991 (True slope = 5), Predicted Intercept: 11.427 (True intercept = 13, loss: 3.5785
(step: 39,000 of 50,000) Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 11.763 (True intercept = 13, loss: 1.5105
(step: 40,000 of 50,000) Predicted Slope: 4.996 (True slope = 5), Predicted Intercept: 11.836 (True intercept = 13, loss: 2.3410
(step: 42,000 of 50,000) Predicted Slope: 4.994 (True slope = 5), Predicted Intercept: 11.968 (True intercept = 13, loss: 2.0370
(step: 43,000 of 50,000) Predicted Slope: 5.012 (True slope = 5), Predicted Intercept: 12.029 (True intercept = 13, loss: 0.9397
(step: 44,000 of 50,000) Predicted Slope: 4.994 (True slope = 5), Predicted Intercept: 12.085 (True intercept = 13, loss: 1.8277
(step: 46,000 of 50,000) Predicted Slope: 4.997 (True slope = 5), Predicted Intercept: 12.188 (True intercept = 13, loss: 2.0670
(step: 47,000 of 50,000) Predicted Slope: 4.999 (True slope = 5), Predicted Intercept: 12.236 (True intercept = 13, loss: 0.2624
(step: 48,000 of 50,000) Predicted Slope: 5.000 (True slope = 5), Predicted Intercept: 12.280 (True intercept = 13, loss: 0.4610
(step: 49,000 of 50,000) Predicted Slope: 4.993 (True slope = 5), Predicted Intercept: 12.322 (True intercept = 13, loss: 0.1421
(step: 50,000 of 50,000) Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 12.361 (True intercept = 13, loss: 0.5046
```
