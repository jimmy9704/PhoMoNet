# PhoMoNet: Monocular Depth Estimation Network with Single-Pixel Depth Guidance
Official implementation of PhoMoNet: Monocular Depth Estimation Network with Single-Pixel Depth Guidance

##### visualization results:
<img src="https://github.com/jimmy9704/PhoMoNet/blob/main/image/Result2.png" width="800"/>

#### Results
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Sched</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">FPS</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<tr><td align="left">PanopticFCN</td>
<td align="center">R50</td>
<td align="center">1x</td>
<td align="center"> 41.1 </td>
<td align="center"> 79.8 </td>
<td align="center"> 49.9 </td>
<td align="center"> 30.2 </td>
<td align="center"> 41.4 </td>
<td align="center"> 13.6 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1tD1A5Zwbtri5OejlIz9MLKwzOzjtIMHQ/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1NeUO9EWtkZE0M5NrEpZ8uFqOX3vQg3Lx/view?usp=sharing">metrics</a> </td>
</tr>
<tr><td align="left">PanopticFCN-400</td>
<td align="center">R50</td>
<td align="center">3x</td>
<td align="center"> 40.8 </td>
<td align="center"> 81.1 </td>
<td align="center"> 49.4 </td>
<td align="center"> 28.9 </td>
<td align="center"> 43.5 </td>
<td align="center"> 26.1 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1QBYMAznZDDX7A0Mnaq3euB23rTBzwUCf/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1QOwbA9KRIvDN8PKh10aCQhf1jpykKwbB/view?usp=sharing">metrics</a> </td>
</tr>
<tr><td align="left">PanopticFCN-512</td>
<td align="center">R50</td>
<td align="center">3x</td>
<td align="center"> 42.3 </td>
<td align="center"> 81.1 </td>
<td align="center"> 51.2 </td>
<td align="center"> 30.7 </td>
<td align="center"> 43.2 </td>
<td align="center"> 22.0 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1QBYMAznZDDX7A0Mnaq3euB23rTBzwUCf/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1QOwbA9KRIvDN8PKh10aCQhf1jpykKwbB/view?usp=sharing">metrics</a> </td>
</tr>
<tr><td align="left">PanopticFCN-600</td>
<td align="center">R50</td>
<td align="center">3x</td>
<td align="center"> 42.7 </td>
<td align="center"> 80.8 </td>
<td align="center"> 51.4 </td>
<td align="center"> 31.6 </td>
<td align="center"> 43.9 </td>
<td align="center"> 19.1 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1gIUxy1DJ_V91IwL5_jHQDMOIgHoWn_O1/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1OfbyJWIVfdGQ0C-JNUnXoocHXdILnIkf/view?usp=sharing">metrics</a> </td>
</tr>
<tr><td align="left">PanopticFCN</td>
<td align="center">R50</td>
<td align="center">3x</td>
<td align="center"> 43.6 </td>
<td align="center"> 81.4 </td>
<td align="center"> 52.5 </td>
<td align="center"> 32.4 </td>
<td align="center"> 43.6 </td>
<td align="center"> 13.5 </td>
<td align="center"> <a href="https://drive.google.com/file/d/18Re3keEkIiy7EVS-uFCNPBfT1BfT8Ng3/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1ACrIJ_AZCW3fD7jcipdya3-ixVBojnFO/view?usp=sharing">metrics</a> </td>
</tr>
<tr><td align="left">PanopticFCN*</td>
<td align="center">R50</td>
<td align="center">3x</td>
<td align="center"> 44.2 </td>
<td align="center"> 81.7 </td>
<td align="center"> 52.9 </td>
<td align="center"> 33.4 </td>
<td align="center"> 43.9 </td>
<td align="center"> 9.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1_VkJIhbQg9uqN49L3cDAW66zZKJE0fkI/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1uulb8PATBy1dF2VhlgQYQlo7gEdKRMK1/view?usp=sharing">metrics</a> </td>
</tr>
</tbody></table>

## Web demo
[Here](https://74c7-163-152-183-111.jp.ngrok.io) you can simulate simple SPAD guidance and visualize the results.

## Datasets
**NYU-Depth-v2**: Please follow the instructions in [BTS](https://github.com/cleinc/bts) to download the training/test set.

**RGB-SPAD**: Please follow the instructions in [DMD](https://github.com/computational-imaging/single_spad_depth) to download the test set.

## Testing
Testing code will be available upon acceptance of our manuscript.

## Training 
Training code will be available upon acceptance of our manuscript.

## Acknowledgments
The code is based on [Adabins](https://github.com/shariqfarooq123/AdaBins) and [DMD](https://github.com/computational-imaging/single_spad_depth).

## Related works
* [DMD (single spad depth)](https://github.com/computational-imaging/single_spad_depth)
* [Adabins](https://github.com/shariqfarooq123/AdaBins)
* [BTS](https://github.com/cleinc/bts)
