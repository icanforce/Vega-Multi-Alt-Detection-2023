# Vega: Drone-based Multi-Altitude Target Detection for Autonomous Surveillance

This repository contains the supplementary material for our paper Vega. The paper appeared in the International Conference on Distributed Computing in Sensor Systems 2023 held in Pafos, Cyprus.

The repository contains the following material.

1. The dataset of aerial images taken in a parking lot at different drone altitudes ranging from h=100ft to h=350ft. 
2. The controller for Vega written for DJI drones using the API exposed by DJI. The controller allows the drone to perform cycles at a fixed altitude. 
3. Code to transform any vision dataset to a smaller resolution. We use this code to check the object detector's performance on target objects of much smaller size. 
4. EfficientDet d0 and d3 model weights. The model was trained on the UAVDT dataset, VisDrone dataset, and the Carpk dataset.

 
