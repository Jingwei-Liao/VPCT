# The code for "Viewport-based Neural 360° Image Compression"

This repository contains the code for our paper "Viewport-based Neural 360° Image Compression". 

## Pipeline Comparison

![ReferenceWVPCT](./assets/into_1.5.png)


## Neural Viewport Codec & VPCT Module

<p align="center">
  <img src="./assets/vpct-General.png" width="30%" />
  <img src="./assets/overall_structure_of_vpct.png" width="48%" />
</p>

## Performance Comparison

<table width="100%">
  <tr>
    <td width="4%"></td>
    <td align="center"><b>LIC360 (1K)</b></td>
    <td align="center"><b>Flickr360 (2K)</b></td>
    <td align="center"><b>CVIQ (4K)</b></td>
    <td align="center"><b>SaliencyVR (8K)</b></td>
  </tr>
  <tr>
    <td align="center" width="4%">
      <div style="writing-mode: vertical-rl; transform: rotate(180deg); font-size: 12px; font-weight: bold; white-space: nowrap;">V-PSNR</div>
    </td>
    <td><img src="./assets/vpsnr_1k_new.png" alt="VPSNR 1K" width="220" height="162" /></td>
    <td><img src="./assets/vpsnr_2k_new.png" alt="VPSNR 2K" width="220" height="162" /></td>
    <td><img src="./assets/vpsnr_4k_new.png" alt="VPSNR 4K" width="220" height="162" /></td>
    <td><img src="./assets/vpsnr_8k_new.png" alt="VPSNR 8K" width="220" height="162" /></td>
  </tr>
  <tr>
    <td align="center" width="4%">
      <div style="writing-mode: vertical-rl; transform: rotate(180deg); font-size: 12px; font-weight: bold; white-space: nowrap;">V-SSIM</div>
    </td>
    <td><img src="./assets/vssim_1k_new.png" alt="VSSIM 1K" width="220" height="162" /></td>
    <td><img src="./assets/vssim_2k_new.png" alt="VSSIM 2K" width="220" height="162" /></td>
    <td><img src="./assets/vssim_4k_new.png" alt="VSSIM 4K" width="220" height="162" /></td>
    <td><img src="./assets/vssim_8k_new.png" alt="VSSIM 8K" width="220" height="162" /></td>
  </tr>
  <tr>
    <td align="center" width="4%">
      <div style="writing-mode: vertical-rl; transform: rotate(180deg); font-size: 12px; font-weight: bold; white-space: nowrap;">V-LPIPS</div>
    </td>
    <td><img src="./assets/vlpips_1k_new.png" alt="VLPIPS 1K" width="220" height="162" /></td>
    <td><img src="./assets/vlpips_2k_new.png" alt="VLPIPS 2K" width="220" height="162" /></td>
    <td><img src="./assets/vlpips_4k_new.png" alt="VLPIPS 4K" width="220" height="162" /></td>
    <td><img src="./assets/vlpips_8k_new.png" alt="VLPIPS 8K" width="220" height="162" /></td>
  </tr>
</table>
