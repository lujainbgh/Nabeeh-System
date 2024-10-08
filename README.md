# Nabeeh-System
### Capstone Project at Data Science and AI Bootcamp at SDAIA
This project implements a real-time traffic accident detection system using advanced computer vision techniques, particularly leveraging YOLO models. The system aims to improve traffic safety and efficiency by detecting accidents in real time and providing alerts to nearby drivers through dynamic signage.

## Problem Statement
- Traffic congestion and accidents are significant issues in Riyadh, affecting urban mobility, safety, and quality of life.
- Accidents in Riyadh contribute to increased travel times, higher risk of subsequent accidents, and environmental pollution.
- Slow detection of accidents and the lack of real-time updates for drivers worsen traffic conditions.
- Quick response to accidents is critical in Riyadh to prevent additional incidents and alleviate congestion.

## Recent statistics in Saudi Arabia

## Project Overview
This project is focused on enhancing road safety by analyzing video streams from traffic cameras to detect traffic incidents. Upon detecting an accident, the system triggers a digital display to warn drivers, suggesting alternative routes, providing traffic density information, and estimating wait times.
The project is aligned with the objectives of Saudi Vision 2030, aiming to contribute to smart city initiatives by improving transportation infrastructure and road safety.

## Features
Accident Detection: Uses YOLOv9m models to detect traffic accidents from live video feeds.
Traffic Density Analysis: YOLOv8m is employed to assess traffic density and provide estimated waiting times.
Smart Traffic Alerts: Digital signs display warnings and alternative routes upon detecting an accident.

## Technologies Used
YOLOv9m and YOLOv8m: Deep learning models for real-time object detection.
Traffic Surveillance Cameras and Dash Cameras: Data sources for accident detection.
Python & OpenCV: Used for video processing and model integration.

## Dataset
The dataset consists of approximately 3,500 video clips from traffic and dash cameras. Videos are categorized into various accident types such as rear-end, front-end, and side-impact collisions. Data preprocessing involved frame extraction, manual labeling, and augmentation techniques like horizontal flipping, Gaussian blur, and noise addition to simulate real-world conditions.

## Methodology
YOLOv9m Model: Transitioned to this model for improved performance with 50 and then 100 epochs.
Traffic Density Calculation: Pretrained YOLOv8m model used for object detection and density classification.

## Results
The system successfully detects accidents with high accuracy and provides real-time warnings to drivers. The smart screen displays alternative routes and traffic conditions, helping to reduce congestion and improve road safety.

## Contributors
Lujain Alghamdi
Nouf Almutairi
Manar Altalhi
Maryam Alsulami

## Acknowledgments
We would like to thank Tuwaiq Academy and SDAIA for their support and the opportunity to work on this project.
