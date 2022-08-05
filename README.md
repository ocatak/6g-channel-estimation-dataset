# 6G Channel Estimation Dataset and Defensive Distillation Based Adversarial Security
## Defensive Distillation based Adversarial Attacks Mitigation Method for Channel Estimation using Deep Learning  Models in Next-Generation Wireless Networks

This research paper focuses on the security concerns of using artificial intelligence in future wireless networks (5G, 6G, 7G and beyond), also known as Next Generation or NextG. It is crucial to protect the next-generation cellular networks against cyber attacks, especially adversarial attacks. The paper proposes a comprehensive vulnerability analysis of deep learning (DL)-based channel estimation models trained with the dataset obtained from MATLAB’s 5G toolbox for adversarial attacks and defensive distillation-based mitigation methods. The adversarial attacks produce faulty results by manipulating trained DL-based models for channel estimation in NextG networks while making models more robust against any attacks through mitigation methods. The paper also presents the performance of the proposed defensive distillation mitigation method for each adversarial attack against the channel estimation model. The results indicated that the proposed mitigation method could defend the DL-based channel estimation models against adversarial attacks in NextG networks.

## Implementation
- [Jupyter Notebook](https://github.com/ocatak/6g-channel-estimation-dataset/blob/main/Channel_Estimation_Attacks_Github.ipynb)
- [Dataset](https://github.com/ocatak/6g-channel-estimation-dataset/blob/main/data.mat)
- [Defensive Distillation Implementation](https://github.com/ocatak/6g-channel-estimation-dataset/blob/main/util_defdistill.py)

## Paper
* Catak, F. O., Kuzlu, M., Catak, E., Cali, U., Guler (2022). Defensive Distillation based Adversarial Attacks Mitigation Method for Channel Estimation using Deep Learning  Models in Next-Generation Wireless Networks (*Under Review*).
## People
- Ferhat Ozgur Catak https://www.uis.no/nb/profile/ferhat-ozgur-catak-0
- Murat Kuzlu https://www.odu.edu/directory/people/m/mkuzlu
- Evren Catak https://www.linkedin.com/in/evren-catak/
- Umit Cali https://www.ntnu.edu/employees/umit.cali
- Ozgur Guler https://www.linkedin.com/in/ozgurgulerphd


## Typical adversarial ML-based adversarial sample generation
![Adversarial Example](https://github.com/ocatak/6g-channel-estimation-dataset/raw/main/typical_adv.png)

## Defensive Distillation

In computer security, defensive distillation is a technique for transforming a machine learning model to make it more robust to adversarial examples. The goal is to distil the knowledge from a complex model into a simpler one that is less susceptible to being fooled by adversarial examples. There are a few different ways to do defensive distillation, but the most common is to train a new, smaller model using the predictions of the original model as labels. Carlini and Wagner first proposed this technique in their 2016 paper "Towards Evaluating the Robustness of Neural Networks".

![Defensive Distillation](https://github.com/ocatak/6g-channel-estimation-dataset/raw/main/6g-defense-channel_estimation_distill.png)


## Dataset Description

In this study, the dataset used to train the DL-based channel estimation models is generated through a reference example in MATLAB 5G Toolbox, i.e, "Deep Learning Data Synthesis for 5G Channel Estimation". In the example, a convolutional neural network (CNN) is used for channel estimation. Single-input single-output (SISO) antenna method is also used by utilizing the physical downlink shared channel (PDSCH) and demodulation reference signal (DM-RS) to create the channel estimation model.  The reference example in the toolbox generates 256 training datasets, i.e., transmit/receive the signal 256 times, for DL-based channel estimation model. Each dataset consists of 8568 data points, i.e., 612 subcarriers, 14 OFDM symbols,  1 antenna. However, each data point of training dataset is converted from a complex (real and imaginary) 612-14 matrix into a real-valued 612-14-2 matrix for providing inputs separately into the neural network during the training process. This is because the resource grids consisting of complex data points with real and imaginary parts in the channel estimation scenario, but CNN model manages the resource grids as 2-D images with real numbers. In this example, the training dataset is converted into 4-D arrays, i.e., 612-14-1-2N, where N presents the number of training examples, i.e., 256.  For each set of the training dataset, a new channel characteristic is generated based on various channel parameters, such as delay profiles (TDL-A, TDL-B, TDL-C, TDL-D, TDL-E), delay spreads (1-300 nanosecond), doppler shifts (5-400 Hz), and Signal-to-noise ratio (SNR or S/N) changes between 0 and 10 dB. Each transmitted waveform with the DM-RS symbols is stored in the train dataset, and the perfect channel values in train labels. The CNN-based channel estimation based is trained with the generated dataset. MATLAB 5G toolbox also allows tuning several communication channel parameters, such as the frequency, subcarrier spacing, number of subcarriers, cyclic prefix type, antennas, channel paths, bandwidth,  code rate, modulation, etc. The channel estimation scenario parameters with values are given for each in the following table.

![Defensive Distillation](https://github.com/ocatak/6g-channel-estimation-dataset/raw/main/channel_est_param.png)


## The vulnerable CNN model

![Defensive Distillation](https://github.com/ocatak/6g-channel-estimation-dataset/raw/main/model_plot.png)





