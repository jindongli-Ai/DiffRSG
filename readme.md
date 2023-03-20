### Basic Information:
This code is released for the papers(underview): 

ECMLPKDD_2023, DiffRSG: Rules-based Skip-GCN in Neural Latent Information Diffusion Network for Social Recommendation

The article is in the period of review. Please do not quote or use it for other purposes


### Usage:
1. Environment: I have tested this code with python3.6, tensorflow-gpu-1.12.0 
2. Run DiffRSG:
   (1) Dataset
   Download datasets from this [link](https://drive.google.com/drive/folders/1YAJvgsCJLKDFPVFMX3OG7v3m1LAYZD5R?usp=sharing), and just put the downloaded folder 'data' in the sub-directory named DiffRSG of your local clone repository.
   (2) cd DiffRSG directory execute the command `python entry.py --data_name=<data_name> --model_name=DiffRSG --gpu=<gpu id>` 
3. If you have any available gpu device, you can specify the gpu id, or you can just ignore the gpu id. 

Following are the command examples:  
`python entry.py --data_name=yelp --model_name=DiffRSG --gpu=1`  
`python entry.py --data_name=flickr --model_name=DiffRSG --gpu=2`

