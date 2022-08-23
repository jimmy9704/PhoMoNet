# PhoMoNet: Monocular Depth Estimation Network with Single-Pixel Depth Guidance
Official implementation of PhoMoNet: Monocular Depth Estimation Network with Single-Pixel Depth Guidance

## Results

##### visualization results:
<img src="https://github.com/jimmy9704/PhoMoNet/blob/main/image/Result2.png" width="800"/>

##### Results on the NYU-Depth-v2 dataset with different SBRs
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">SBR</th>
<th valign="bottom">Result Rel</th>

<!-- TABLE BODY -->
<tr><td align="left">PhoMoNet</td>
<td align="center">100</td>
<td align="center">download</td>
</tr>
<tr><td align="left">PhoMoNet</td>
<td align="center">50</td>
<td align="center">download</td>
</tr>
<tr><td align="left">PhoMoNet</td>
<td align="center">10</td>
<td align="center">download</td>
</tr>
<tr><td align="left">PhoMoNet</td>
<td align="center">5</td>
<td align="center">download</td>
</tr>
</tbody></table>

## Web demo
[Here](https://c9d6-163-152-183-111.jp.ngrok.io/) you can simulate simple SPAD guidance and visualize the results.

## Datasets
**NYU-Depth-v2**: Please follow the instructions in [BTS](https://github.com/cleinc/bts) to download the training/test set.

**RGB-SPAD**: Please follow the instructions in [single spad depth](https://github.com/computational-imaging/single_spad_depth) to download the test set.

## Testing
Testing code will be available upon acceptance of our manuscript.

## Training 
Training code will be available upon acceptance of our manuscript.

## Acknowledgments
The code is based on [Adabins](https://github.com/shariqfarooq123/AdaBins) and [single spad depth](https://github.com/computational-imaging/single_spad_depth).

## Related works
* [single spad depth](https://github.com/computational-imaging/single_spad_depth)
* [Adabins](https://github.com/shariqfarooq123/AdaBins)
* [BTS](https://github.com/cleinc/bts)
