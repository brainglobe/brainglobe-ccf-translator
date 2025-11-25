When ingesting new transforms please do the following. 
1. Ensure the source and target atlas are in the [BrainGlobe atlasapi](https://github.com/brainglobe/brainglobe-atlasapi)
2. Create a new script in ingesting_new_deformations. Usually the name of this script should be the name of the new atlas space being added (if you are adding two totally new spaces not connected to existing graph then just try your best and I will help out).
3. Create a new script in metadata generation. For the format of this script please refer to the other existing integration scripts. 

Notes:
    * Usually we should use the same name in CCF translator as in the atlas api. exceptions to this are when there are alternative versions of the same atlas in the api but they exist in the same space. An example of this is DeMBA is called demba_allen_seg_dev_mouse in the atlas api but simply demba_dev_mouse in CCF translator as there are plans for a future demba_kim_seg_dev_mouse.  