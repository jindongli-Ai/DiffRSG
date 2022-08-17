### Basic Information:
This code is released for the papers: 

AAAI_2023, DiffRSG: Rules-based Skip-GCN in Neural Latent Information Diffusion Network for Social Recommendation

The article is in the period of review. Please do not quote or use it for other purposes


### Usage:
1. Environment: I have tested this code with python3.6, tensorflow-gpu-1.12.0 
3. Run DiffNet: 
   1. cd the sub-directory diffnet and execute the command `python entry.py --data_name=<data_name> --model_name=diffnet --gpu=<gpu id>` 
4. Run DiffNet++:
   1. cd the sub-directory diffnet++ and execute the command `python entry.py --data_name=<data_name> --model_name=diffnetplus --gpu=<gpu id>` 
5. If you have any available gpu device, you can specify the gpu id, or you can just ignore the gpu id. 

Following are the command examples:  
`python entry.py --data_name=yelp --model_name=DiffRSG --gpu=1`  
`python entry.py --data_name=flickr --model_name=DiffRSG --gpu=2`

