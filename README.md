# distributed_tensorflow
Sample Implementation of Distributed TensorFlow

This should work for TF > 1.4.

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
ps_hosts = ["10.100.68.245"]
ps_ports = ["2222"]
worker_hosts = ["10.100.68.193","10.100.68.183"] #,"10.100.68.185","10.100.68.187"]
worker_ports = ["2222", "2222"] #, "2222", "2222"]
```
3. Run `python test_dist.py --issync=1` on every machine. You can specify the IP address (--ip) if you want to use OPA instead of the usual IPv4/6 address.

Here's a sample output from one of the worker nodes:
```
(tf) [tony@tensor816 distributed_tensorflow]$ python test_dist.py --ip=10.100.68.193 --issync=0
Distributed TensorFlow training
Parameter server nodes are: ['10.100.68.245:2222']
Worker nodes are ['10.100.68.193:2222', '10.100.68.183:2222']
E1219 17:16:00.594426061  256441 ev_epoll1_linux.c:1051]     grpc epoll fd: 3
2017-12-19 17:16:00.606638: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job ps -> {0 -> 10.100.68.245:2222}
2017-12-19 17:16:00.606825: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job worker -> {0 -> localhost:2222, 1 -> 10.100.68.183:2222}
2017-12-19 17:16:00.621637: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:324] Started server with target: grpc://localhost:2222
2017-12-19 17:16:05.214110: I tensorflow/core/distributed_runtime/master_session.cc:1004] Start master session bbb6329477418b8d with config:
step: 0 of 50,000, Predicted Slope: -0.605 (True slope = 5), Predicted Intercept: -0.452 (True intercept = 13, loss: 8838.7070
step: 1,000 of 50,000, Predicted Slope: 4.995 (True slope = 5), Predicted Intercept: 0.367 (True intercept = 13, loss: 176.1432
step: 2,000 of 50,000, Predicted Slope: 5.027 (True slope = 5), Predicted Intercept: 1.103 (True intercept = 13, loss: 162.4184
step: 3,000 of 50,000, Predicted Slope: 4.974 (True slope = 5), Predicted Intercept: 1.796 (True intercept = 13, loss: 118.7577
step: 4,000 of 50,000, Predicted Slope: 5.042 (True slope = 5), Predicted Intercept: 2.448 (True intercept = 13, loss: 101.8099
step: 5,000 of 50,000, Predicted Slope: 4.888 (True slope = 5), Predicted Intercept: 3.061 (True intercept = 13, loss: 141.2458
step: 6,000 of 50,000, Predicted Slope: 5.006 (True slope = 5), Predicted Intercept: 3.641 (True intercept = 13, loss: 92.9185
step: 8,000 of 50,000, Predicted Slope: 5.004 (True slope = 5), Predicted Intercept: 4.700 (True intercept = 13, loss: 67.9143
step: 10,000 of 50,000, Predicted Slope: 4.982 (True slope = 5), Predicted Intercept: 5.636 (True intercept = 13, loss: 56.2651
step: 12,000 of 50,000, Predicted Slope: 5.020 (True slope = 5), Predicted Intercept: 6.470 (True intercept = 13, loss: 39.8159
step: 13,000 of 50,000, Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 6.850 (True intercept = 13, loss: 32.2074
step: 14,000 of 50,000, Predicted Slope: 5.007 (True slope = 5), Predicted Intercept: 7.208 (True intercept = 13, loss: 37.4388
step: 15,000 of 50,000, Predicted Slope: 5.016 (True slope = 5), Predicted Intercept: 7.545 (True intercept = 13, loss: 37.0500
step: 17,000 of 50,000, Predicted Slope: 5.036 (True slope = 5), Predicted Intercept: 8.164 (True intercept = 13, loss: 18.4807
step: 18,000 of 50,000, Predicted Slope: 4.995 (True slope = 5), Predicted Intercept: 8.447 (True intercept = 13, loss: 20.1930
step: 21,000 of 50,000, Predicted Slope: 4.979 (True slope = 5), Predicted Intercept: 9.197 (True intercept = 13, loss: 14.3947
step: 23,000 of 50,000, Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 9.628 (True intercept = 13, loss: 7.4217
step: 24,000 of 50,000, Predicted Slope: 5.002 (True slope = 5), Predicted Intercept: 9.824 (True intercept = 13, loss: 9.0870
step: 26,000 of 50,000, Predicted Slope: 5.008 (True slope = 5), Predicted Intercept: 10.184 (True intercept = 13, loss: 8.6674
step: 27,000 of 50,000, Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 10.347 (True intercept = 13, loss: 4.7181
step: 29,000 of 50,000, Predicted Slope: 5.007 (True slope = 5), Predicted Intercept: 10.648 (True intercept = 13, loss: 5.4148
step: 31,000 of 50,000, Predicted Slope: 5.005 (True slope = 5), Predicted Intercept: 10.914 (True intercept = 13, loss: 8.8434
step: 33,000 of 50,000, Predicted Slope: 5.012 (True slope = 5), Predicted Intercept: 11.149 (True intercept = 13, loss: 4.0885
step: 34,000 of 50,000, Predicted Slope: 5.003 (True slope = 5), Predicted Intercept: 11.257 (True intercept = 13, loss: 1.0327
step: 35,000 of 50,000, Predicted Slope: 4.989 (True slope = 5), Predicted Intercept: 11.359 (True intercept = 13, loss: 5.2579
step: 36,000 of 50,000, Predicted Slope: 4.997 (True slope = 5), Predicted Intercept: 11.455 (True intercept = 13, loss: 2.6394
step: 37,000 of 50,000, Predicted Slope: 5.004 (True slope = 5), Predicted Intercept: 11.546 (True intercept = 13, loss: 0.9799
step: 39,000 of 50,000, Predicted Slope: 5.011 (True slope = 5), Predicted Intercept: 11.710 (True intercept = 13, loss: 3.1933
step: 41,000 of 50,000, Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 11.856 (True intercept = 13, loss: 0.8543
step: 42,000 of 50,000, Predicted Slope: 5.001 (True slope = 5), Predicted Intercept: 11.922 (True intercept = 13, loss: 3.1766
step: 43,000 of 50,000, Predicted Slope: 4.997 (True slope = 5), Predicted Intercept: 11.984 (True intercept = 13, loss: 0.9390
step: 44,000 of 50,000, Predicted Slope: 4.998 (True slope = 5), Predicted Intercept: 12.043 (True intercept = 13, loss: 0.9113
step: 45,000 of 50,000, Predicted Slope: 4.992 (True slope = 5), Predicted Intercept: 12.099 (True intercept = 13, loss: 1.2349
step: 47,000 of 50,000, Predicted Slope: 4.995 (True slope = 5), Predicted Intercept: 12.201 (True intercept = 13, loss: 0.9698
step: 48,000 of 50,000, Predicted Slope: 4.995 (True slope = 5), Predicted Intercept: 12.248 (True intercept = 13, loss: 1.7298
step: 49,000 of 50,000, Predicted Slope: 4.994 (True slope = 5), Predicted Intercept: 12.292 (True intercept = 13, loss: 0.9635
```
