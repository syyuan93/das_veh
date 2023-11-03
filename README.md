# Automated and Continuous Near-surface characterization Using Vehicle-induced DAS signals

Virtual shot gathers with one car signal | Virtual shot gathers with 236 car signals   
:-------------------------:|:-------------------------:
<img src="figures/shot_gather_one.png" width="324">  |  <img src="figures/20221223/vs_time_540m_nCars_236.png" width="324">

Disperson image with one car signal | Disperson image with 236 car signals   
:-------------------------:|:-------------------------:
<img src="figures/dispersion_one.png" width="324">  |  <img src="figures/20221223/disp_time_540m_nCars_236.png" width="324">

This is the repository for the following paper. If you use this implementation, please cite our papers:

>* Yuan, S., Liu, J., Noh, H. Y., Clapp, R., & Biondi, B.(2023). Using Vehicle-induced DAS Signals for Near-surface Characterization with High Spatiotemporal Resolution. JGR: Solid Earth, in preparation.

[[paper]]()

and

>* Jingxiao Liu, Siyuan Yuan, Yiwen Dong, Biondo Biondi, and Hae Young Noh. 2023. TelecomTM: A Fine-Grained and Ubiquitous Traffic Monitoring System Using Pre-Existing Telecommunication Fiber-Optic Cables as Sensors. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 7, 2, Article 64 (June 2023), 24 pages. https://doi.org/10.1145/3596262

[[paper]](https://dl.acm.org/doi/10.1145/3596262) [[code]](https://github.com/jingxiaoliu/TelecomTM)

### Description

![flowchart.](figures/flowchart.png)

This study proposes a novel method for detecting spatial subsurface heterogeneity and rain-induced soil saturation changes in the San Francisco Bay Area. Our approach utilizes vehicles as cost-effective surface-wave sources that excite wavefield recorded by a roadside Distributed Acoustic Sensing (DAS) array. Leveraging a Kalman filter vehicle-tracking algorithm, we can automatically track hundreds of vehicles each day, allowing us to extract space-time windows of high-quality surface waves. By constructing highly accurate virtual shot gathers from these waves, we can perform time-lapse surface-wave analyses with high temporal and spatial resolutions.

### Code Usage
```
git clone https://github.com/das_veh

```
- Run the demo example with
```
jupyter notebook demo.ipynb
```

### Contact
Feel free to send any questions to:
- [Siyuan Yuan](mailto:syyuan@stanford.edu)

Note: The telecommunication cable DAS data used to support the fndings of this study are available from the corresponding author upon request. 
