# Third-party notices

BUSS-Net uses or derives from the following external projects. Their source trees, datasets, and pretrained weights are not included in this repository.

- **EVSSM** — <https://github.com/kkkls/EVSSM>  
  The GeoFSS implementation was adapted from commit `5098a5276640694a39a941119f9e8bcc3ece9fb0`. EVSSM is distributed under the MIT License; retain its copyright and license notices when redistributing derived portions.
- **nnU-Net** — <https://github.com/MIC-DKFZ/nnUNet>  
  BUSS-Net uses the fixed two-dimensional `PlainConvUNet` architecture through the installed `nnunetv2` package.
- **dynamic-network-architectures** — <https://github.com/MIC-DKFZ/dynamic-network-architectures>  
  Provides the `PlainConvUNet` building blocks used by the model.
- **Mamba / mamba-ssm** — <https://github.com/state-spaces/mamba>  
  Provides the selective-scan implementation used by GeoFSS.
- **PraNet** — <https://github.com/DengPingFan/PraNet>  
  The public polyp dataset split and download references follow its established protocol. Dataset rights remain with their respective owners.
